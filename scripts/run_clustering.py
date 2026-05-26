#!/usr/bin/env python3
"""Extract rating-segmented feedback themes with phrase-first cluster headings."""
import json
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_FILE = Path("7-1-2024 -5-19-2026 Housing Feedback.csv")
TEXT_COLUMN_CANDIDATES = ["Comments", "Feedback", "comment", "text"]
RATING_COLUMN_CANDIDATES = ["Rating", "rating", "Score", "score"]


def find_text_column(df: pd.DataFrame) -> str:
    for col in TEXT_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        if df[col].dtype == object:
            return col
    raise ValueError("Could not find a text column in dataset.")


def find_rating_column(df: pd.DataFrame) -> str:
    for col in RATING_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find a rating column. Tried: {RATING_COLUMN_CANDIDATES}")


def choose_topic_name(top_terms: list[str]) -> str:
    for term in top_terms:
        if " " in term:
            return term
    return top_terms[0] if top_terms else "n/a"


def is_extremely_positive_short_comment(comment: str, max_words: int = 6) -> bool:
    positive_markers = {
        "great", "excellent", "awesome", "amazing", "perfect", "wonderful",
        "fantastic", "love", "loved", "good", "nice", "best", "outstanding",
    }
    words = [w.strip(".,!?;:\"'()[]{}").lower() for w in str(comment).split()]
    words = [w for w in words if w]

    if len(words) > max_words or not words:
        return False

    positive_hits = sum(1 for w in words if w in positive_markers)
    return positive_hits >= 2 or (positive_hits / len(words)) >= 0.5


def cluster_segment(texts: pd.Series, n_clusters: int) -> list[dict]:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.95,
        max_features=8000,
    )
    X = vectorizer.fit_transform(texts)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    terms = vectorizer.get_feature_names_out()
    topics: list[dict] = []
    for cluster_id in range(n_clusters):
        center = model.cluster_centers_[cluster_id]
        term_idx = center.argsort()[-10:][::-1]
        top_terms = [terms[i] for i in term_idx]
        topic_name = choose_topic_name(top_terms)
        size = int((labels == cluster_id).sum())
        sample = texts[labels == cluster_id].iloc[0] if size else ""

        topics.append(
            {
                "topic_name": topic_name,
                "cluster_id": int(cluster_id),
                "size": size,
                "top_terms": top_terms,
                "sample_comment": str(sample)[:180],
            }
        )
    return topics


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    text_col = find_text_column(df)
    rating_col = find_rating_column(df)

    base = df[[text_col, rating_col]].copy()
    base[text_col] = base[text_col].fillna("").astype(str)
    base = base[base[text_col].str.strip() != ""]
    base[rating_col] = pd.to_numeric(base[rating_col], errors="coerce")
    base = base.dropna(subset=[rating_col])

    high = base[base[rating_col] >= 4.0].copy()
    low = base[base[rating_col] < 4.0].copy()

    if len(high) < 5 or len(low) < 5:
        raise ValueError("Need at least 5 comments in each rating segment.")

    high_filtered = high[~high[text_col].apply(is_extremely_positive_short_comment)].copy()
    if len(high_filtered) < 5:
        high_filtered = high

    report = {
        "rows_used": int(len(base)),
        "text_column": text_col,
        "rating_column": rating_col,
        "model": "KMeans + TF-IDF keyphrase extraction",
        "segments": {
            "rating_gte_4": {
                "comments_before_filter": int(len(high)),
                "comments_after_filter": int(len(high_filtered)),
                "clusters": cluster_segment(high_filtered[text_col], n_clusters=5),
            },
            "rating_lt_4": {
                "comments": int(len(low)),
                "clusters": cluster_segment(low[text_col], n_clusters=5),
            },
        },
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
