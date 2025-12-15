# Fake News Detector
A simple, fast Fake vs Real news detector with a clean Flask UI.


## Quickstart
1. Create venv, install deps (see requirements.txt)
2. Put `Fake.csv` + `True.csv` into `data/raw/`
3. `python src/data/make_dataset.py`
4. `python src/features/build_tfidf.py`
5. `python src/models/train_baselines.py`
6. `flask run` → open http://127.0.0.1:5000