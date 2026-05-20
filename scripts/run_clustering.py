#!/usr/bin/env python3
"""Extract common ideas/phrases from feedback with topic modeling + keyphrases."""
import json
from pathlib import Path

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

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


def top_phrases_from_rows(rows: pd.Series, top_k: int = 10) -> list[str]:
    vec = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(2, 3),
        min_df=2,
        max_df=0.9,
        max_features=3000,
    )
    X = vec.fit_transform(rows)
    scores = X.sum(axis=0).A1
    phrases = vec.get_feature_names_out()
    idx = scores.argsort()[-top_k:][::-1]
    return [phrases[i] for i in idx]


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    text_col = find_text_column(df)
    texts = df[text_col].fillna("").astype(str)
    texts = texts[texts.str.strip() != ""]

    if len(texts) < 10:
        raise ValueError("Need at least 10 non-empty text rows for topic extraction.")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=8000,
    )
    X = vectorizer.fit_transform(texts)

    n_topics = min(8, max(3, len(texts) // 40))
    model = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=500)
    W = model.fit_transform(X)
    H = model.components_

    terms = vectorizer.get_feature_names_out()
    topic_assignments = W.argmax(axis=1)

    report = {
        "rows_used": int(len(texts)),
        "text_column": text_col,
        "model": "NMF(topic modeling) + TF-IDF keyphrase extraction",
        "topics": [],
        "global_top_phrases": top_phrases_from_rows(texts, top_k=15),
    }

    for topic_id in range(n_topics):
        term_idx = H[topic_id].argsort()[-10:][::-1]
        top_terms = [terms[i] for i in term_idx]

        member_mask = topic_assignments == topic_id
        member_rows = texts[member_mask]
        size = int(member_mask.sum())

        phrases = top_phrases_from_rows(member_rows, top_k=8) if size >= 5 else []

        report["topics"].append(
            {
                "topic": int(topic_id),
                "size": size,
                "top_terms": top_terms,
                "top_phrases": phrases,
            }
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
