from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = joblib.load('models/logreg_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

# Helper function to generate and encode graphs as base64 for display in the web app
def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the text from the form
        text = request.form["text"]
        
        # Transform the text into the feature vector using TF-IDF
        vectorized_text = tfidf.transform([text])
        
        # Get prediction from the model
        prediction = model.predict(vectorized_text)[0]
        prediction_proba = model.predict_proba(vectorized_text)[:, 1]
        
        # Evaluate model on some test data (for demonstration purposes, we use the validation set)
        val_data = pd.read_csv('Data/processed/val.csv')
        X_val = tfidf.transform(val_data['text'])
        y_val = val_data['label']
        
        # Predict on validation data
        y_pred = model.predict(X_val)
        
        # Calculate accuracy and confusion matrix
        accuracy = accuracy_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_val, prediction_proba)
        auc = roc_auc_score(y_val, prediction_proba)
        
        # Plot confusion matrix with seaborn
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        img_cm_base64 = plot_to_base64(fig_cm)

        # Plot ROC Curve with plotly
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig_roc.update_layout(
            title=f'ROC Curve (AUC = {auc:.2f})',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        img_roc_base64 = fig_roc.to_html(full_html=False)
        
        # Return the results to the HTML template
        return render_template("index.html", 
                               prediction='Fake' if prediction == 0 else 'True', 
                               accuracy=accuracy, 
                               img_cm_base64=img_cm_base64, 
                               img_roc_base64=img_roc_base64)
    
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
