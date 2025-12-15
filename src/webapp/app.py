from flask import Flask, render_template, request
import joblib
from pathlib import Path

app = Flask(__name__)

VEC = joblib.load(Path("data/processed/features/tfidf_vectorizer.joblib"))
MODEL = joblib.load(Path("models/baselines/logreg.joblib"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    proba = None
    title_in = ""
    text_in = ""

    if request.method == "POST":
        title_in = request.form["title"]
        text_in = request.form["text"]

        X = VEC.transform([title_in + " " + text_in])
        pred = MODEL.predict(X)[0]
        prob = MODEL.predict_proba(X)[0].max()

        prediction = "Real" if pred == 1 else "Fake"
        proba = f"{prob*100:.2f}%"

    return render_template(
        "index.html",
        prediction=prediction,
        proba=proba,
        title_in=title_in,
        text_in=text_in
    )

if __name__ == "__main__":
    app.run(debug=True)
