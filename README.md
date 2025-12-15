Fake News Detection
Overview

This project uses machine learning techniques to detect whether a news article is fake or real. The model is trained using Logistic Regression with TF-IDF (Term Frequency-Inverse Document Frequency) for text feature extraction. The model is then deployed through a simple Flask web application that allows users to input news articles and get predictions in real-time.

Project Structure

The project is divided into the following key components:

project/
│
├── data/                        # Raw and processed data
│   ├── processed/               # Preprocessed data (train, validation, test)
│   └── raw/                     # Raw data
│
├── models/                      # Folder for saving models
│   ├── baselines/               # Folder for baseline models
│   │   ├── logreg.joblib        # Trained Logistic Regression model
│   │   └── tfidf_vectorizer.joblib  # TF-IDF Vectorizer
│
├── experiments/                 # Folder for saving evaluation results (plots, metrics)
│   ├── confusion_matrix.png     # Confusion matrix plot
│   └── roc_curve.png            # ROC curve plot
│
├── src/                         # Source code
│   ├── data/                    # Scripts related to data preprocessing
│   ├── eval/                    # Scripts related to evaluation
│   ├── features/                # Scripts related to feature extraction
│   ├── models/                  # Scripts related to model training
│   └── webapp/                  # Web application
│
├── app.py                       # Main entry point for Flask web app
├── requirements.txt             # File listing all required Python packages
└── README.md                    # Project overview and instructions

Installation

Clone the Repository:
Clone this repository to your local machine.

https://github.com/Ar-5060/fake-news-detection

Create a Virtual Environment (optional but recommended):
If you're using a virtual environment to manage your dependencies:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
Install the required Python libraries by running:

pip install -r requirements.txt


Download Data:
You will need to download the raw data files, Fake.csv and True.csv, which contain fake and real news articles, respectively. These files should be placed in the data/raw/ folder.

Usage

Data Preprocessing:
Preprocess the raw data by running the following script. This will clean the data and split it into training, validation, and test sets:

python src/data/make_dataset.py


Feature Extraction:
Build the TF-IDF vectorizer by running the following script. This will save the vectorizer as a .joblib file:

python src/features/build_tfidf.py


Model Training:
Train the Logistic Regression model using the preprocessed data and the TF-IDF vectorizer:

python src/models/train_baselines.py


Model Evaluation:
After training the model, evaluate it on the validation set and print out the performance metrics:

python src/eval/evaluate_baselines.py


Evaluate on Test Set:
To evaluate the model on the test set, generate and save the confusion matrix and ROC curve:

python src/eval/evaluate.py


Run the Flask Web Application:
Start the Flask web app to use the model for predictions. Navigate to the src/webapp/ folder and run:

python app.py


The app will be accessible at http://127.0.0.1:5000 in your browser. You can enter news text to get a prediction (Fake or Real).

Web Application Interface

Home Page: A form where users can enter the news text.

Prediction: The model will classify the entered news article as either Fake or Real.

Evaluation Results: Users can view the evaluation results, such as the confusion matrix and ROC curve.

Folder Structure Explanation

data/: Contains the raw and processed data files. The raw/ folder contains the original Fake.csv and True.csv files, and the processed/ folder contains the preprocessed datasets (train.csv, val.csv, test.csv).

models/: Stores the trained Logistic Regression model (logreg.joblib) and TF-IDF vectorizer (tfidf_vectorizer.joblib).

experiments/: Contains the saved evaluation plots such as confusion matrices and ROC curves.

src/: Contains all the source code files, split into subfolders for data preprocessing, feature extraction, model training, and the web application.

app.py: Main entry point for the Flask web app.

requirements.txt: Lists all required Python packages, including Flask, scikit-learn, pandas, joblib, etc.

README.md: This file, containing instructions and explanations.

Example Output (Command Line):

Training Model:

Training Logistic Regression model...
Model trained and saved as logreg.joblib


Evaluation:

Classification Report:
precision    recall  f1-score   support

0.0       0.98      0.97      0.97      1234
1.0       0.95      0.97      0.96      1234

accuracy                           0.97      2468
macro avg       0.96      0.97      0.96      2468
weighted avg    0.96      0.97      0.96      2468


Flask Web App:
After entering the news text, the result will be shown as:

The news is: Real