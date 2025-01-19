from nltk.corpus import stopwords 
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd 

def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    score = analysis.sentiment.polarity  # Polarity score: -1 (negative) to 1 (positive)
    
    if score > 0:
        return 'positive'
    elif score == 0:
        return 'neutral'
    else:
        return 'negative'



# **********************1) DATA LOADING PROCESS*****************************************
twt = pd.read_json('tweets.20150430-223406.json/tweets.20150430-223406.json', lines=True)
# print(twt.shape)
# print(twt.info)
# print(twt.dtypes)
# print(twt.describe())

# ********************** 2) DATA SELECTION AND CLEANING,FILTERING PROCESS ***********************************
#we are only selecting dates and comments which are used for forther analysis
twt = twt[['created_at','text']]
# print(twt)
'''Following Steps are used to remove the noise present in the twt file '''
#text process for twitter data 
twt['text'] = twt['text'].str.lstrip('0123456789')
#Lowercase conversion
twt['text'] = twt['text'].apply(lambda a: " ".join(a.lower() for a in a.split()))
# #remove pantuation 
# twt['text'] = twt['text'].str.replace('[^\w\s]','')
twt['text'] = twt['text'].str.replace('[^\w\s]', '', regex=True)
#remove stopwords
stop = stopwords.words('english')
twt['text'] = twt['text'].apply(lambda a: " ".join(a for a in a.split() if a not in stop))
# #Spelling Correction 
# twt['text'].apply(lambda a: str(TextBlob(a).correct())) #This par of code take tooo much time so i am skipping this for some time 
# print(twt.tail())


# **************************** PREDICTING THE EMOTION *****************************************
# befor this i created the def get_sentiment function 
twt['Sentiment'] = twt['text'].apply(get_sentiment)
# Split the data into features (X) and labels (y)
X = twt['text']
y = twt['Sentiment']
# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#word level tf-idf 
tv = TfidfVectorizer()
X_train_tfidf = tv.fit_transform(X_train)
X_test_tfidf = tv.transform(X_test)

# Train a Logistic Regression classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_tfidf, y_train)

#Get Predicted Emotion 
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# ************************** PREDICTING THE SENTIMENT *****************************************
# Show a few predictions with actual and predicted sentiment
# Create predictions DataFrame using X_test for the text
predictions = pd.DataFrame({'text': X_test, 'Actual': y_test, 'Predicted': y_pred})
print(predictions.head(10))






