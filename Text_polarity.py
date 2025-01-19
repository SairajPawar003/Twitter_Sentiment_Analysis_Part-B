from textblob import TextBlob

# Example sentences
text_positive = "I love programming. It's so enjoyable and rewarding!"
text_negative = "I hate bugs in the code. They make me frustrated."
text_neutral = "I am writing some Python code."

# Create TextBlob objects
blob_positive = TextBlob(text_positive)
blob_negative = TextBlob(text_negative)
blob_neutral = TextBlob(text_neutral)

# Get sentiment polarity
print(f"Positive Text Polarity: {blob_positive.sentiment.polarity}")  # > 0, e.g., 0.85
print(f"Negative Text Polarity: {blob_negative.sentiment.polarity}")  # < 0, e.g., -0.6
print(f"Neutral Text Polarity: {blob_neutral.sentiment.polarity}")    # ~ 0, e.g., 0.0
