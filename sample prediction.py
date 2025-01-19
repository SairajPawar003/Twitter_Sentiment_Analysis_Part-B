import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json

# Load your JSON file containing the Twitter data (Replace with your file path)
file_path = 'tweets.20150430-223406.json/tweets.20150430-223406.json'
twt = pd.read_json(file_path, lines=True)

# Check the structure of the data (e.g., if there is a 'text' column)
print(twt.columns)

# Filter necessary columns (assuming 'text' is the column containing tweet text)
twt = twt[['created_at', 'text']]  # Use only the columns that are useful for analysis

# Clean and preprocess the text data
# Remove any unwanted characters (like numbers, URLs, etc.)
twt['text'] = twt['text'].str.replace('[^\w\s]', '', regex=True)  # Remove punctuation
twt['text'] = twt['text'].str.lower()  # Convert to lowercase

# Optional: Remove stopwords (if needed, you can use NLTK stopwords)

# Sentiment Analysis with TextBlob
def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    score = analysis.sentiment.polarity  # Polarity score: -1 (negative) to 1 (positive)
    
    if score > 0:
        return 'positive'
    elif score == 0:
        return 'neutral'
    else:
        return 'negative'

# Apply sentiment function to tweets
twt['Sentiment'] = twt['text'].apply(get_sentiment)

# Sample the dataset if it's too large (optional)
# twt = twt.sample(n=1000, random_state=42)

# Split the data into features (X) and labels (y)
X = twt['text']
y = twt['Sentiment']

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')  # You can adjust stop words here
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Show a few predictions with actual and predicted sentiment
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(predictions.head())
