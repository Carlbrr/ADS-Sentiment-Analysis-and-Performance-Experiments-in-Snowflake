--ALTER SESSION SET USE_CACHED_RESULT=FALSE;

-- SETUP -- (combine tables for simplicity) ---------------------------------------------------------------------------------------------
Create or replace table yelp_training_udtf(data variant);
Create or replace table yelp_testing_udtf(data variant);

COPY INTO yelp_training_udtf FROM @bullfrog_stage/train_data/train-00000-of-00001.parquet FILE_FORMAT = training_db.TPCH_SF1.MYPARQUETFORMAT;
COPY INTO yelp_testing_udtf FROM @bullfrog_stage/test_data/test-00000-of-00001.parquet FILE_FORMAT = training_db.TPCH_SF1.MYPARQUETFORMAT;

select * from yelp_training_udtf limit 5;

CREATE OR REPLACE TABLE yelp_combined_data_udtf ( -- it gets out of hand to make this in two different udtfs because we give input row-wise, so the first udtf would have to get priors and likelihood, and the next should grab then likelihood, priors as well as the test data..
    data_mode STRING,          -- "train" or "predict"
    label NUMBER,         -- score (0 or 4)
    review VARCHAR(16777216) -- Review text
);

INSERT INTO yelp_combined_data_udtf (data_mode, label, review) --260k rows
SELECT 'train', data:"label"::INT, data:"text"::STRING
FROM yelp_training_udtf
WHERE data:"label" IN (0, 4);

INSERT INTO yelp_combined_data_udtf (data_mode, label, review) --20k rows
SELECT 'test', data:"label"::INT, data:"text"::STRING
FROM yelp_testing_udtf
WHERE data:"label" IN (0, 4);

-- Snowflake’s vectorized UDTFs handle batches through Pandas DataFrames directly, with the process method accepting columns as Pandas Series rather than handling one row at a time.
-- This is 10000x easier than the previous method, where we had to handle each row individually and then aggregate the results
-------------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION BULLFROG_DB.PUBLIC.UDTF_SENTIMENT_ANALYSIS("MODE" STRING, "LABEL" NUMBER, "REVIEW" VARCHAR(16777216))
RETURNS TABLE("ACTUAL_SCORE" NUMBER, "PREDICTED_SCORE" NUMBER)
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('pandas')
HANDLER = 'SentimentAnalysisVectorizedUDTF'
AS $$
import math
import pandas as pd
from _snowflake import vectorized

class SentimentAnalysisVectorizedUDTF:

    def __init__(self):
        # Initialize stopwords, word counts, total reviews, priors, and likelihoods
        self.stopwords = { #same stopwords as before
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "he", "him", "his", "himself",
            "she", "her", "hers", "it", "its", "itself", "they",
            "them", "their", "theirs", "themselves", "what", "which", "who",
            "whom", "this", "that", "these", "those", "am", "is", "are", "was",
            "were", "be", "been", "being", "have", "has", "had", "having",
            "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
            "or", "because", "as", "until", "while", "of", "at", "by", "for",
            "with", "about", "against", "between", "into", "through", "during",
            "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under"
        }
        self.word_counts = {0: {}, 4: {}}  # Word counts by score
        self.total_reviews = {0: 0, 4: 0}  # Review counts by score
        self.prior_probs = {}  # Prior probabilities by score
        self.likelihoods = {}  # Likelihood of each word for each score

    def clean_text(self, text_series):
        # Tcleancleanclean again
        lowercase_text = text_series.str.lower()
        cleaned_text = lowercase_text.str.replace(r"(?<![a-z])\'|\'(?![a-z])|[^a-z\s\']", " ", regex=True)
        words_series = cleaned_text.str.split()
        # Remove stopwords from each row in the series. we need to use apply because its a pandas Series (so like a column in a table)
        return words_series.apply(lambda words: [word for word in words if word not in self.stopwords])

    @vectorized(input=pd.DataFrame)
    def end_partition(self, df):
        # Rename columns for Snowflake UDTF requirements - had an error otherwise
        df.columns = ["MODE", "LABEL", "REVIEW"]
        
        # Clean text by splitting reviews into words
        df["words"] = self.clean_text(df["REVIEW"])

        # Separate data into training and testing sets - now we have everything we need and can just follow what we did for SQL version
        train_data = df[df["MODE"] == "train"]
        test_data = df[df["MODE"] == "test"]

        # Count word occurrences by label on training data for likelihoods and priors
        for _, row in train_data.iterrows():
            label = row["LABEL"]
            self.total_reviews[label] += 1
            #This is a single row, so one review_id so to say, where we have split the reviews and now count the words for that score
             # (just like the bag of words model we did before)
            for word in row["words"]:
                if word not in self.word_counts[label]:
                    self.word_counts[label][word] = 0
                self.word_counts[label][word] += 1

        # Calculate priors and likelihoods after training
        self.calculate_probabilities()

        # Prediction phase for each test review
        results = []
        for _, row in test_data.iterrows():
            words = row["words"]
            actual_label = row["LABEL"]

            # Start with log of prior probabilities for each score
            score_probabilities = {score: math.log(self.prior_probs[score]) for score in self.prior_probs}

            # POSTERIOR = PRIOR * LIKELIHOOD (or log(P) + log(L) = log(P*L))
            # Sum the log likelihoods for each word in the test review with the prior probabilities
            # Its basically just, for each word in the review, get the likelihood of that word for each score and add it to the score probability (including the prior)
            # this gives each score a posterior prob and the "prediction" is the max of these two
            for word in words:
                for score in score_probabilities: 
                    #This is how we handle it in sql as well
                    likelihood = math.log(self.likelihoods.get((word, score), 1e-10))
                    score_probabilities[score] += likelihood

            # Predict the score with the highest posterior probability
            predicted_score = max(score_probabilities, key=score_probabilities.get)
            results.append((actual_label, predicted_score))

        # Return predictions as an iterable of tuples
        return pd.DataFrame(results, columns=["ACTUAL_SCORE", "PREDICTED_SCORE"])

    def calculate_probabilities(self):
        # Compute priors based on training review counts
        total_reviews_all = sum(self.total_reviews.values())
        self.prior_probs = {score: count / total_reviews_all for score, count in self.total_reviews.items()}

        # Create a set of unique words for Laplace smoothing
        unique_words = set(word for words in self.word_counts.values() for word in words)
        vocabulary_size = len(unique_words)

        # Calculate likelihoods for each word given each score with Laplace smoothing
        for score, word_counts in self.word_counts.items():
            total_words_for_score = sum(word_counts.values())
            for word, count in word_counts.items():
                # Calculate likelihood: P(word|score) = (count + 1) / (total_words_for_score + vocabulary_size)
                self.likelihoods[(word, score)] = float((count + 1)) / float((total_words_for_score + vocabulary_size))
$$;


WITH prediction_results AS (
    SELECT 
        ACTUAL_SCORE, 
        PREDICTED_SCORE
    FROM 
        yelp_combined_data_udtf,
        TABLE(BULLFROG_DB.PUBLIC.UDTF_SENTIMENT_ANALYSIS(data_mode, label, review) 
              OVER (PARTITION BY 1))
)
SELECT 
    COUNT(*) AS total_reviews,
    (SUM(CASE WHEN ACTUAL_SCORE = PREDICTED_SCORE THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0)) * 100 AS accuracy_percentage
FROM 
    prediction_results;
-- The accuracy percentage here is 92.235% for 20k reviews in test and took 54s

