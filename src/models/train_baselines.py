from pathlib import Path
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


PROC = Path('data/processed')
FEAT = PROC / 'features'
MODELS = Path('models/baselines')
MODELS.mkdir(parents=True, exist_ok=True)


# Load data
train = pd.read_csv(PROC / 'train.csv')
val = pd.read_csv(PROC / 'val.csv')


vectorizer = joblib.load(FEAT / 'tfidf_vectorizer.joblib')
X_train = vectorizer.transform((train['title'].fillna('') + ' ' + train['text'].fillna('')).tolist())
y_train = train['label'].values
X_val = vectorizer.transform((val['title'].fillna('') + ' ' + val['text'].fillna('')).tolist())
y_val = val['label'].values


# Train
clf = LogisticRegression(max_iter=2000, n_jobs=None, class_weight='balanced', solver='liblinear')
clf.fit(X_train, y_train)


# Validate
pred = clf.predict(X_val)
acc = accuracy_score(y_val, pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_val, pred, average='binary')
print(f"val: acc={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")


# Save model
joblib.dump(clf, MODELS / 'logreg.joblib')
print('Saved model →', MODELS / 'logreg.joblib')