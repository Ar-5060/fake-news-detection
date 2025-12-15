from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


PROC = Path('data/processed')
FEAT = PROC / 'features'
FEAT.mkdir(parents=True, exist_ok=True)


train = pd.read_csv(PROC / 'train.csv')
val = pd.read_csv(PROC / 'val.csv')


# Use title + text combined for stronger signal
train_corpus = (train['title'].fillna('') + ' ' + train['text'].fillna('')).tolist()


vectorizer = TfidfVectorizer(
max_features=50000,
ngram_range=(1,2),
min_df=2,
max_df=0.9,
sublinear_tf=True,
)
X_train = vectorizer.fit_transform(train_corpus)


joblib.dump(vectorizer, FEAT / 'tfidf_vectorizer.joblib')
print('Saved TF-IDF vectorizer →', FEAT / 'tfidf_vectorizer.joblib')