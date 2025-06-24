import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

base_dir = r'C:\\Users\\Mann\Desktop\\Intro to ML and NLP\Week2'   #Change this to your actual path
file_name = 'spam.csv'                                             # Ensure the file exists in the specified path
file_path = os.path.join(base_dir, file_name)
df = pd.read_csv(file_path, encoding='latin-1')
stop_words = set(stopwords.words('english'))

df = df.iloc[:, :2]
df.columns = ['label', 'message']

# Preprocessing function
def preprocess_message(message):
    tokens = word_tokenize(message.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return filtered_tokens

# Apply preprocessing to the 'message' column
df['processed_message'] = df['message'].apply(preprocess_message)

model = api.load('word2vec-google-news-300')  
def avg_word2vec(tokens, model):
    valid_vectors = []
    for word in tokens:
        if word in model:
            valid_vectors.append(model[word])
    
    if not valid_vectors:
        return np.zeros(model.vector_size)
    return np.mean(valid_vectors, axis=0)

X = np.array([avg_word2vec(tokens, model) for tokens in df['processed_message']])
y = df['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
lgr = LogisticRegression(max_iter=1000)
lgr.fit(X_train, y_train)

# Calculate accuracy
y_pred = lgr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Logistic Regression model: {accuracy * 100}%")


def predict_message_class(model,w2v_model,message):
    return model.predict([avg_word2vec(preprocess_message(message), w2v_model)])

#  Example
message = "Congratulations! You won a free ticket to Bahamas!"
print(predict_message_class(lgr, model, message))
    
