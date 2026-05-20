#!/usr/bin/env python3
"""Run clustering on the feedback CSV and print a short report."""
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

DATA_FILE = Path("7-1-2024-5-19-2026 General Feedback.csv")
TEXT_COLUMN_CANDIDATES = ["Feedback", "Comments", "comment", "text"]


def find_text_column(df: pd.DataFrame) -> str:
    for col in TEXT_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        if df[col].dtype == object:
            return col
    raise ValueError("Could not find a text column in dataset.")


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    text_col = find_text_column(df)

    texts = df[text_col].fillna("").astype(str)
    texts = texts[texts.str.strip() != ""]

    if len(texts) < 5:
        raise ValueError("Need at least 5 non-empty text rows for clustering.")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(texts)

    n_clusters = min(5, max(2, len(texts) // 25))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    terms = vectorizer.get_feature_names_out()
    centers = model.cluster_centers_

    report = {
        "rows_used": int(len(texts)),
        "text_column": text_col,
        "clusters": [],
    }

    for i in range(n_clusters):
        top_idx = centers[i].argsort()[-8:][::-1]
        top_terms = [terms[j] for j in top_idx]
        size = int((labels == i).sum())
        report["clusters"].append({"cluster": i, "size": size, "top_terms": top_terms})

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
