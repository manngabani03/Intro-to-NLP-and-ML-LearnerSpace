import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import gensim.downloader as api
from nltk.tokenize import word_tokenize
import os
import re
from nltk.stem import WordNetLemmatizer

# Again same as problem1, i saved the csv file locally on my PC 
base_dir = r'C:\\Users\\Mann\Desktop\\Intro to ML and NLP\Week2'  # Change this to your actual path
file_name = 'Tweets.csv'                                          # Ensure the file exists in the specified path
file_path = os.path.join(base_dir, file_name)
df = pd.read_csv(file_path, encoding='latin-1')

df = df[['text', 'airline_sentiment']]
# Preprocessing text function 

def preprocess_text(text):
    regex = re.sub(r'@\w+|http\S+|#\w+|', '', text)
    regex = re.sub(r"\bdon't\b", "do not", regex)
    regex = re.sub(r"\bdidn't\b", "did not", regex)
    regex = re.sub(r'[^\w\s]', '', regex)
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(regex.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    regex = ' '.join(lemmatized_tokens)
    return regex


df['processed_text'] = df['text'].apply(preprocess_text)

model = api.load('word2vec-google-news-300')

def avg_word2vec(tokens, model):
    valid_vectors = []
    for word in tokens:
        if word in model:
            valid_vectors.append(model[word])
    
    if not valid_vectors:
        return np.zeros(model.vector_size)
    return np.mean(valid_vectors, axis=0)

X =np.array([avg_word2vec(word_tokenize(text), model) for text in df['processed_text']])
y = df['airline_sentiment']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression(max_iter=1000, multi_class='multinomial')
lgr.fit(X_train, y_train)

y_pred = lgr.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}%")


def predict_tweet_sentiment(model, w2v_model, tweet):
    return model.predict([avg_word2vec(word_tokenize(preprocess_text(tweet)), w2v_model)])

# Example
tweet = "The flight was delayed and the staff was rude."
print(predict_tweet_sentiment(lgr, model, tweet))




    
