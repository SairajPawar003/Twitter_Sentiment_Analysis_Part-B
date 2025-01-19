import pandas as pd 
t = pd.read_json('tweets.20150430-223406.json/tweets.20150430-223406.json', lines=True)
print(t.dtypes)