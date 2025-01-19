from textblob import TextBlob

# Example tweet
tweet = "I hate the new feature in the app! It's amazing."

# Analyze sentiment
analysis = TextBlob(tweet)
polarity = analysis.sentiment.polarity

# Classify sentiment based on polarity
if polarity > 0:
    sentiment = "Positive"
elif polarity < 0:
    sentiment = "Negative"
else:
    sentiment = "Neutral"

print(f"Tweet: {tweet}")
print(f"Polarity: {polarity}")
print(f"Sentiment: {sentiment}")
