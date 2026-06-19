# ============================================================
# ml_models.py
# Project : Real-Time Social Media Analytics for Sentiment
#           and Trend Detection
# Course  : CSI-630 | Govt. Municipal Graduate College, Faisalabad
# Purpose : Train Random Forest + Logistic Regression on
#           Sentiment140 dataset. Save/load models. Provide
#           evaluation metrics for dashboard display.
# ============================================================

import os
import re
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection    import train_test_split
from sklearn.metrics            import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline           import Pipeline

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "models")
LR_PATH      = os.path.join(MODEL_DIR, "lr_pipeline.pkl")
RF_PATH      = os.path.join(MODEL_DIR, "rf_pipeline.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.pkl")

# Sentiment140 CSV (download separately — see train_models.py)
SENTIMENT140_PATH = os.path.join(BASE_DIR, "data", "sentiment140.csv")


# ── Text cleaning (lightweight, fast) ──────────────────────────────
def _clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+",           " ", text)
    text = re.sub(r"#(\w+)",         r"\1", text)   # keep hashtag word
    text = re.sub(r"[^a-z\s]",       " ", text)
    text = re.sub(r"\s+",            " ", text).strip()
    return text


# ── Pipeline factory ────────────────────────────────────────────────
def _build_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", clf),
    ])


# ── Training ────────────────────────────────────────────────────────
def train_and_save(sample_size: int = 100_000) -> dict:
    """
    Load Sentiment140, train LR + RF, save models and metrics.

    Parameters
    ----------
    sample_size : int
        Total rows to train on (split 50/50 pos/neg for balance).
        Full dataset = 1.6 M rows. 100 k is fast and representative.

    Returns
    -------
    dict  {model_name: {accuracy, f1_score, confusion_matrix}}
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(SENTIMENT140_PATH):
        raise FileNotFoundError(
            f"Sentiment140 CSV not found at:\n  {SENTIMENT140_PATH}\n"
            "Run  python train_models.py  to download it automatically."
        )

    print(f"[ML] Loading Sentiment140 (sample={sample_size:,}) ...")
    cols = ["polarity", "id", "date", "query", "user", "text"]
    df   = pd.read_csv(
        SENTIMENT140_PATH,
        encoding="latin-1",
        header=None,
        names=cols,
        usecols=["polarity", "text"],
    )

    # Balance classes: equal negative (0) and positive (4) samples
    half = sample_size // 2
    neg  = df[df["polarity"] == 0].sample(n=half, random_state=42)
    pos  = df[df["polarity"] == 4].sample(n=half, random_state=42)
    df   = pd.concat([neg, pos]).sample(frac=1, random_state=42).reset_index(drop=True)

    df["label"] = (df["polarity"] == 4).astype(int)   # 1 = positive
    df["clean"] = df["text"].apply(_clean)

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["clean"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"],
    )

    results = {}

    # ── Logistic Regression ─────────────────────────────────────────
    print("[ML] Training Logistic Regression ...")
    lr = _build_pipeline(
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
    )
    lr.fit(X_tr, y_tr)
    lr_pred = lr.predict(X_te)
    results["Logistic Regression"] = {
        "accuracy":         round(accuracy_score(y_te, lr_pred),           4),
        "f1_score":         round(f1_score(y_te, lr_pred, average="weighted"), 4),
        "confusion_matrix": confusion_matrix(y_te, lr_pred).tolist(),
    }
    with open(LR_PATH, "wb") as f:
        pickle.dump(lr, f)
    print(f"[ML] LR done — acc={results['Logistic Regression']['accuracy']}, "
          f"f1={results['Logistic Regression']['f1_score']}")

    # ── Random Forest ───────────────────────────────────────────────
    print("[ML] Training Random Forest (may take 5-10 min) ...")
    rf = _build_pipeline(
        RandomForestClassifier(
            n_estimators=200, max_depth=30,
            min_samples_split=5, n_jobs=-1, random_state=42,
        )
    )
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)
    results["Random Forest"] = {
        "accuracy":         round(accuracy_score(y_te, rf_pred),           4),
        "f1_score":         round(f1_score(y_te, rf_pred, average="weighted"), 4),
        "confusion_matrix": confusion_matrix(y_te, rf_pred).tolist(),
    }
    with open(RF_PATH, "wb") as f:
        pickle.dump(rf, f)
    print(f"[ML] RF done — acc={results['Random Forest']['accuracy']}, "
          f"f1={results['Random Forest']['f1_score']}")

    with open(METRICS_PATH, "wb") as f:
        pickle.dump(results, f)

    print("[ML] Models + metrics saved to  models/")
    return results


# ── Inference helpers ───────────────────────────────────────────────
_cache: dict = {}

def _load(key: str, path: str):
    if key not in _cache and os.path.exists(path):
        with open(path, "rb") as f:
            _cache[key] = pickle.load(f)
    return _cache.get(key)


def predict_sentiment(texts: list[str], model: str = "lr") -> list[str]:
    """
    Predict sentiment using a trained ML model.

    Parameters
    ----------
    texts : list of raw post strings
    model : 'lr'  → Logistic Regression
            'rf'  → Random Forest

    Returns
    -------
    list of 'Positive' | 'Negative'
    """
    pipe = _load("lr", LR_PATH) if model == "lr" else _load("rf", RF_PATH)
    if pipe is None:
        return ["Unknown"] * len(texts)
    cleaned = [_clean(t) for t in texts]
    preds   = pipe.predict(cleaned)
    return ["Positive" if p == 1 else "Negative" for p in preds]


def load_metrics() -> dict | None:
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "rb") as f:
        return pickle.load(f)


def models_exist() -> bool:
    return os.path.exists(LR_PATH) and os.path.exists(RF_PATH)
