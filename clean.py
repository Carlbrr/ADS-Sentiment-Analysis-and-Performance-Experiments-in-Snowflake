import re
#obs. when creating the UDF in Snowflake, we must care for the ' character - we must use the double ' to escape it in the code, 
#as all the python code is just seen as a string literal.


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

# Example usage
sentence = "YOYO. This is a small test sentece. We even see some punctuation! it's it' i"
cleaned_words = clean_text(sentence)
print(cleaned_words)