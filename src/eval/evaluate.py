from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Define directories for processed data, features, models, and experiment results
PROC = Path('data/processed')
FEAT = PROC / 'features'
MODELS = Path('models/baselines')
EXP = Path('experiments')
EXP.mkdir(parents=True, exist_ok=True)

# Load the test data, TF-IDF vectorizer, and trained model
test = pd.read_csv(PROC / 'test.csv')
vec = joblib.load(FEAT / 'tfidf_vectorizer.joblib')
clf = joblib.load(MODELS / 'logreg.joblib')

# Prepare the test data for prediction
X_test = vec.transform((test['title'].fillna('') + ' ' + test['text'].fillna('')).tolist())
y_test = test['label'].values

# Predict the labels for the test set
pred = clf.predict(X_test)

# Print the classification report
print(classification_report(y_test, pred, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cbar=False, xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(EXP / 'confusion_matrix.png', dpi=160)
print('Saved figure → experiments/confusion_matrix.png')

# ROC Curve and AUC calculation
# Get predicted probabilities for the positive class (Real)
y_prob = clf.predict_proba(X_test)[:, 1]

# Calculate the ROC curve (False Positive Rate, True Positive Rate)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='b', label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random classifier)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()

# Save the ROC curve plot
plt.savefig(EXP / 'roc_curve.png', dpi=160)
plt.show()  # This will display the plot
print('Saved figure → experiments/roc_curve.png')
