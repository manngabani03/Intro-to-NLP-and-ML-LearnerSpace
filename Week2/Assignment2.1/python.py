import math
from math import log

corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]

# Compute document frequency for each word
doc_count = len(corpus)
word_doc_freq = {}
for sentence in corpus:
    unique_words = set(sentence.split())
    for word in unique_words:
        word_doc_freq[word] = word_doc_freq.get(word, 0) + 1

# Compute TF-IDF for each word in each sentence
for idx, sentence in enumerate(corpus):
    words = sentence.split()
    total_words = len(words)
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    print(f"Sentence {idx+1}:")
    for word in set(words):
        tf = word_freq[word] / total_words
        idf = log(doc_count / word_doc_freq[word])
        tfidf = tf * idf
        print(f"  TF-IDF for word '{word}': {tfidf}")


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]

model = Pipeline([
    ('count_vectorizer', CountVectorizer()),
    ('tfidf_transformer', TfidfTransformer())
])

model.fit(corpus)

print("TF-IDF Output using Scikit-Learn:")
print(model.named_steps['count_vectorizer'].get_feature_names_out())
tfidf_matrix = model.transform(corpus).toarray()
print(tfidf_matrix)

