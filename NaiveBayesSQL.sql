-- COPY INTO yelp_training from @bullfrog_stage/train_data/train-00000-of-00001.parquet FILE_FORMAT = training_db.TPCH_SF1.MYPARQUETFORMAT;
--create or replace table yelp_testing (data variant)
-- COPY INTO yelp_testing from @bullfrog_stage/test_data/test-00000-of-00001.parquet FILE_FORMAT = training_db.TPCH_SF1.MYPARQUETFORMAT;

CREATE OR REPLACE FUNCTION BULLFROG_DB.PUBLIC.CLEAN_TEXT("TEXT_INPUT" VARCHAR(16777216))
RETURNS ARRAY
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
HANDLER = 'clean_text'
AS '
import re
# Just a list of Pronouns, helping verbs, articles, conjunctions and prepositions 
stopwords = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they",
    "them", "their", "theirs", "themselves", "what", "which", "who",
    "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under"
}

def clean_text(text_input: str):
    # Convert text to lowercase
    lowercase_text = text_input.lower()
    # Regex to remove anything not a letter, space, or apostrophess inside words
    cleaned_text = re.sub(r"(?<![a-z])\'|\'(?![a-z])|[^a-z\s\']", " ", lowercase_text)
    # Split into words
    words = cleaned_text.split()
    # Filter out stopwords
    filtered_words = [word for word in words if word not in stopwords]
    return filtered_words
';


-- Now, create view of all cleaned reviews.. we need an id consistent with the review to keep track of word relations
-- use row_number to generates a unique sequential number for each row... the over null creates a row number without any specific ordering
CREATE OR REPLACE VIEW train_yelp_format AS
SELECT ROW_NUMBER() OVER (ORDER BY NULL) as review_id, data:"text"::STRING AS review_text, data:"label"::Integer as score
FROM yelp_training
WHERE 
    data:"label" IN (0, 4);  -- Filter for scores 0 and 4


-- We want to clean (and split) all the reviews we have collected, as well as aggregate same word occurences (bag of words representation)
-- Create a view to clean the reviews and flatten the words by (maintaining the ID)
CREATE OR REPLACE VIEW train_yelp_cleaned AS
SELECT
    review_id,
    f.value::STRING AS word,
    score
FROM
    train_yelp_format,
    LATERAL FLATTEN(input => BULLFROG_DB.PUBLIC.CLEAN_TEXT(review_text)) f;

-- Now we want to group the words by id and word, such that we can show a count of those words for each review
CREATE OR REPLACE VIEW train_yelp_cleaned_count AS
SELECT
    review_id,
    word,
    COUNT(*) as word_count,
    score
FROM
    train_yelp_cleaned
GROUP BY review_id, word, score;


-- PERFECT -> Now we are ready to move on to the simple/naive bayes, as we have our "bag of words" representation they state in the stanford slides.
-- First we need the prior probabilities of each class (review score).. so the prob of each score occurring in the data
-- just the amount of reviews in a given score grp divided by the count of all reviews

-- PRIOR PROB AND LIKELIHOOD ARE SLIDE 30 (LEARNING), SO Calculate P(c) with c being the "term"/score and P(w|c) So word w given score c
-- This is exactly the prior probability (so what is the prob for a score for all reviews) and likelihood (then what is the prob this score and word together)
-- also it is for likelihood we use laplace smoothing (+1) as to accomodate not having seen some word x with score y, and thus not having a 0 prob as they "cannot be conditioned away" slide 28/29

create or replace table prior_prob as 
SELECT
    score,
    COUNT(DISTINCT review_id) / (SELECT COUNT(DISTINCT review_id) FROM train_yelp_cleaned_count) AS prob_of_score
FROM
    train_yelp_cleaned_count
GROUP BY
    score;

-- Now we know the prob of a score without seeing a word, now we want to include the word and get the likelihood of a score given a word.
-- that is, how likely is this word for each score? or in other words, for which score is this word most likely.

CREATE or replace table word_count_by_score as -- Calculates the total number of words for each score
SELECT
        score,
        SUM(word_count) AS total_word_count
    FROM
        train_yelp_cleaned_count
    GROUP BY
        score;

CREATE or replace table vocabulary_size as -- Counts the total number of distinct words across the entire dataset    
    SELECT COUNT(DISTINCT word) AS V
    FROM train_yelp_cleaned_count;

CREATE OR REPLACE TABLE word_likelihoods AS -- what is the prob of score 4 and word "good" for example
SELECT
    r.score,
    r.word, --sum(r.word_count) is the count of that word for all reviews in the score group (0 or 4) then divide by total number of words for each score
    ((SUM(r.word_count) + 1)::double) / ((t.total_word_count + v.V)::double) AS P_word_given_score -- LAPLACE SMOOTHING
FROM
    train_yelp_cleaned_count r
JOIN
    word_count_by_score t ON r.score = t.score
JOIN
    vocabulary_size v
GROUP BY
    r.score, r.word, total_word_count, v.V;



-- We now have the prior probabilities and the likelihood of each word given a score, the next step is to combine these probabilities to compute the posterior probabilities.
-- So, predict the score based on the words in the review. So posterior is like "prob of P(word|score) given we now have some knowledge", which we sum for all words in the review
-------------------------------------------------------------------------------------------------------------------------------------------

-- so we just need to sum the log of probabilities/likelihoods... 
-- or well multiply the likelihoods by the prior probabilities of each score to get the posterior probability and pick the highest


-- Step 1: Format, then clean and split words from Yelp testing reviews
-- format test data with review_id, review_text, and score
CREATE OR REPLACE VIEW yelp_testing_format AS
SELECT 
    ROW_NUMBER() OVER (ORDER BY NULL) AS review_id,
    data:"text"::STRING AS review_text,
    data:"label"::Integer AS score
FROM 
    yelp_testing
WHERE 
    data:"label" IN (0, 4);

CREATE OR REPLACE VIEW yelp_testing_cleaned AS
SELECT
    review_id,
    f.value::STRING AS word,
    score
FROM
    yelp_testing_format,
    LATERAL FLATTEN(input => BULLFROG_DB.PUBLIC.CLEAN_TEXT(review_text)) f;
    
-- Step 2: Count word occurrences for each review in the testing data
CREATE OR REPLACE VIEW yelp_testing_cleaned_count AS
SELECT
    review_id,
    word,
    COUNT(*) AS word_count,
    score
FROM
    yelp_testing_cleaned
GROUP BY
    review_id, word, score;


-- Step 3 and 4 would then include our prior probabilities and likelihoods but we already have these from the training!
SELECT * FROM prior_prob;  -- TRAINING VIEW
SELECT * FROM word_likelihoods; -- TRAINING VIEW

-- Step 5: Calculate the posterior probabilities for each score in the testing data
--Why log? Logarithms convert multiplication into addition, which helps avoid underflow issues when dealing with very small numbers (like probabilities).
--adding sum of log probs and log of prior prob gives the total log probability of observing that review given that score.

--ok, we need test data, priors and likelihoods
CREATE OR REPLACE table POSTERIOR_PROB AS
SELECT 
    t.review_id,
    t.score AS actual_score,
    0 AS predicted_score,
    -- Sum of the log of the likelihoods of words given score 0
    -- Setting COALESCE to zero for unseen words essentially gave those words an infinite negative log-probability (ln(0) is undefined), which could dominate the overall score for reviews with new words and lead to near-zero likelihoods for those scores. By using a small positive value like 1e-10, we assigned unseen words a very low but non-zero probability, preserving score calculations without overpowering them – effectively reducing their influence without nullifying them. This went from 87% to 92.2796 %
    SUM(COALESCE(LN(l.P_word_given_score), LN(1e-10))) + LN(p.prob_of_score) AS log_posterior_score_0,
    -- Repeat for score 4
    SUM(COALESCE(LN(l2.P_word_given_score), LN(1e-10))) + LN(p2.prob_of_score) AS log_posterior_score_4
FROM 
    yelp_testing_cleaned_count t
-- Join with likelihoods for score 0
LEFT JOIN 
    word_likelihoods l ON t.word = l.word AND l.score = 0
-- Join with likelihoods for score 4
LEFT JOIN 
    word_likelihoods l2 ON t.word = l2.word AND l2.score = 4
-- Join with prior probabilities
JOIN 
    prior_prob p ON p.score = 0
JOIN 
    prior_prob p2 ON p2.score = 4
GROUP BY 
    t.review_id, t.score, p.prob_of_score, p2.prob_of_score;


-- We need to use coalsce anyway even with laplace smoothing as we can encounter a word in the testing data that we didnt see in training data.


-- Predict score based on the max posterior prop for 0 and 4 
CREATE OR REPLACE TABLE predicted_scores AS
SELECT
    review_id,
    actual_score,
    CASE
        WHEN log_posterior_score_0 > log_posterior_score_4 THEN 0
        ELSE 4
    END AS predicted_score
FROM 
    POSTERIOR_PROB;

-- Evaluate the accuracy of the predictions
SELECT 
    COUNT(*) AS total_reviews,
    (SUM(CASE WHEN actual_score = predicted_score THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS accuracy_percentage
FROM 
    predicted_scores; -- 92.27960
