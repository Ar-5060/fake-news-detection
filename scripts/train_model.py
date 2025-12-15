import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features():
    # Load preprocessed data
    train_data = pd.read_csv('Data/processed/train.csv')
    val_data = pd.read_csv('Data/processed/val.csv')

    # Feature extraction using TF-IDF
    tfidf = TfidfVectorizer(max_features=10000)
    X_train = tfidf.fit_transform(train_data['text'])
    X_val = tfidf.transform(val_data['text'])

    y_train = train_data['label']
    y_val = val_data['label']

    # Save TF-IDF vectorizer for later use
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

    return X_train, X_val, y_train, y_val

def train_model():
    X_train, X_val, y_train, y_val = extract_features()

    # Train Logistic Regression Model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'models/logreg_model.pkl')
    print("Model and TF-IDF vectorizer have been saved.")

if __name__ == '__main__':
    train_model()
