import math
import pandas as pd

class SentimentAnalysisVectorizedUDTF:

    def __init__(self):
        # Our clean up stopwords like before
        self.stopwords = {
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
        self.word_counts = {0: {}, 4: {}}  # Word counts for each score
        self.total_reviews = {0: 0, 4: 0}  # Review counts for each score
        self.prior_probs = {}  # Prior probabilities for each score
        self.likelihoods = {}  # Likelihood of each word for each score
        #self.test_data = []  # Hold test data for later predictions (will not happen in the process method but in end_partition as its a batch process/vectorized UDTF)

    def clean_text(self, text_series):
        # Clean and split for pandas Series of text
        lowercase_text = text_series.str.lower()
        cleaned_text = lowercase_text.str.replace(r"(?<![a-z])\'|\'(?![a-z])|[^a-z\s\']", " ", regex=True)
        words_series = cleaned_text.str.split()
        # We need to "apply" the stopwords removal to each row because its a pandas Series (like a column in a table)
        return words_series.apply(lambda words: [word for word in words if word not in self.stopwords])

    def process(self, modes, labels, reviews):
        # Convert inputs to Series if not already - i got an error in snowflake that it didnt. 
        # modes, labels, and reviews are read as scalar values rather than as lists or Pandas Series - so its not vectorized?
        if not isinstance(modes, pd.Series):
            modes = pd.Series(modes)
        if not isinstance(labels, pd.Series):
            labels = pd.Series(labels)
        if not isinstance(reviews, pd.Series):
            reviews = pd.Series(reviews)

        # Convert inputs to DataFrame
        df = pd.DataFrame({"mode": modes, "label": labels, "review": reviews})
        
        # Clean text and split for word counts and prediction
        df["words"] = self.clean_text(df["review"])

        # Split data based on mode
        train_data = df[df["mode"] == 'train']
        self.test_data = df[df["mode"] == 'test']

        # Accumulate word counts for training data
        for _, row in train_data.iterrows():
            label = row["label"]
            self.total_reviews[label] += 1
            #This is a single row, so one review_id so to say, where we have split the reviews on line 41 and now count the words for that score
            # (just like the bag of words model we did before)
            for word in row["words"]:
                if word not in self.word_counts[label]:
                    self.word_counts[label][word] = 0
                self.word_counts[label][word] += 1
    
    def end_partition(self, df):
        # Calculate priors and likelihoods after processing all training data
        self.calculate_probabilities()

        # Prediction/test phase
        results = []
        for _, row in self.test_data.iterrows():
            words = row["words"]
            actual_label = row["label"]

            # Log probabilities with prior probabilities, so this will hold the posterior prob of each score for this review
            score_probabilities = {score: math.log(self.prior_probs[score]) for score in self.prior_probs}

            # Sum the log likelihoods for each word in the review with the prior probabilities
            # Its basically just, for each word in the review, get the likelihood of that word for each score and add it to the score probability (including the prior)
            # this gives each score a posterior prob and the "prediction" is the max of these two
            for word in words:
                for score in score_probabilities:
                    likelihood = self.likelihoods.get((word, score), 1 / (1000 + len(self.likelihoods)))
                    score_probabilities[score] += math.log(likelihood)

            # Predict the score with the highest posterior probability
            predicted_score = max(score_probabilities, key=score_probabilities.get)
            results.append((actual_label, predicted_score))

        # Return predictions for evaluation
        predicted_score = max(score_probabilities, key=score_probabilities.get)
        results.append((actual_label, predicted_score))

        # Return the tuples
        return iter(results)

    # Helper method to separate the "training" part of the classifier
    def calculate_probabilities(self):
        # Calculate priors and likelihoods after we have procesed all training data
        total_reviews_all = sum(self.total_reviews.values())
        self.prior_probs = {score: count / total_reviews_all for score, count in self.total_reviews.items()}

        # Calculate vocabulary size for Laplace smoothing. so distinct words, which we can just a set for
        unique_words = set()
        for words in self.word_counts.values():
            #words.keys() returns the words (keys) in each dictionary (so for both 0 and 4 and unions them into one set - ez pez)
            unique_words.update(words.keys())

        vocabulary_size = len(unique_words)

        # Calculate likelihood for each word given each label with Laplace smoothing (+1 like we did before)
        for score, word_counts in self.word_counts.items():
            #First the total words for that score
            total_words_for_score = sum(word_counts.values())
            #Then we iterate over all the words for that score and calculate the likelihood, so P(word|score) = (count (for that word for that score) + 1) / (total_words_for_score + vocabulary_size)
            for word, count in word_counts.items():
                self.likelihoods[(word, score)] = (count + 1) / (total_words_for_score + vocabulary_size)


# Create an instance of the UDTF
# Sample data simulating yelp_combined_data_udtf
data = {
    "mode": ["train", "train", "train", "test", "test"],
    "label": [0, 4, 0, 4, 0],
    "review": [
        "I love this place, it’s wonderful and fantastic!",
        "Awful experience, would not recommend.",
        "Excellent service, very satisfied.",
        "Terrible food, never coming back.",
        "Best restaurant in town, absolutely amazing."
    ]
}

# Create DataFrame for testing
df = pd.DataFrame(data)

# Initialize and run the UDTF
udtf = SentimentAnalysisVectorizedUDTF()
udtf.process(df["mode"], df["label"], df["review"])
results = udtf.end_partition()

# Display results
print(results)