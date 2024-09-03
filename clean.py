import re
#obs. when creating the UDF in Snowflake, we must care for the ' character - we must use the double ' to escape it in the code, 
#as all the python code is just seen as a string literal.

def clean_text(text_input):
    # Make lowercase
    lowercase_text = text_input.lower()
    # Regex to remove anything not a letter or space
    cleaned_text = re.sub(r'[^a-z\s]', '', lowercase_text)
    # Split to words (we retained whitespace)
    return cleaned_text.split()

# Example usage
sentence = "YOYO. This is a small test sentece. We even see some punctuation!"
cleaned_words = clean_text(sentence)
print(cleaned_words)