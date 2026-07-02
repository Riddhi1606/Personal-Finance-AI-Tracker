"""
categorization.py
------------------
Trains and serves a text-classification model that assigns a spend
category (Food & Dining, Groceries, Transport, ...) to each transaction
based on its cleaned merchant narration.

Model choice: TF-IDF (char n-grams, since merchant names are short and
often mis-spelled/truncated in bank statements) + Multinomial Naive
Bayes. This is deliberately a lightweight, explainable classical ML
model rather than a deep model, since the dataset per user is small
(hundreds-thousands of rows) and interpretability matters for a
finance app.
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODEL_PATH = "models/categorizer.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
        )),
        ("clf", MultinomialNB(alpha=0.3)),
    ])


def train(df: pd.DataFrame, model_path: str = MODEL_PATH):
    """
    df must have 'clean_merchant' (input) and 'true_category' (label).
    Only debit transactions are categorized into spend buckets; credits
    are labelled separately (e.g. 'Income') upstream.
    """
    X = df["clean_merchant"]
    y = df["true_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)

    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    print(f"Test accuracy: {acc:.3f}\n")
    print(report)
    return pipeline, acc


def load_model(model_path: str = MODEL_PATH) -> Pipeline:
    return joblib.load(model_path)


def predict_category(pipeline: Pipeline, clean_merchant: str) -> str:
    return pipeline.predict([clean_merchant])[0]


def predict_batch(pipeline: Pipeline, merchants: pd.Series) -> pd.Series:
    return pd.Series(pipeline.predict(merchants), index=merchants.index)


if __name__ == "__main__":
    from preprocessing import preprocess_transactions

    raw = pd.read_csv("data/transactions.csv")
    df = preprocess_transactions(raw)
    train(df)
