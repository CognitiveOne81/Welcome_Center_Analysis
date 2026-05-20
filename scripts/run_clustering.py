#!/usr/bin/env python3
"""Extract phrase clusters for specific date/rating buckets."""
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_FILE = Path("7-1-2024-5-19-2026 General Feedback.csv")
DATE_COLUMN = "Date"
RATING_COLUMN = "Rating"
TEXT_COLUMN_CANDIDATES = ["Comments", "Feedback", "comment", "text"]

PHASE_WINDOWS = [
    ("2024-07-01", "2025-06-30", "phase_1_2024-07-01_to_2025-06-30"),
    ("2025-07-01", "2026-05-19", "phase_2_2025-07-01_to_2026-05-19"),
]
RATING_GROUPS = [
    ([4], "rating_4"),
    ([3], "rating_3"),
    ([0, 1, 2], "rating_0_1_2"),
]

SHORT_POSITIVE_MARKERS = {
    "good",
    "great",
    "excellent",
    "awesome",
    "amazing",
    "nice",
    "perfect",
    "helpful",
    "friendly",
    "love",
    "loved",
    "wonderful",
    "fantastic",
    "outstanding",
    "satisfied",
    "happy",
}


def find_text_column(df: pd.DataFrame) -> str:
    for col in TEXT_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find a text column. Tried: {', '.join(TEXT_COLUMN_CANDIDATES)}"
    )

def top_phrases_from_rows(rows: pd.Series, top_k: int = 12) -> list[str]:
    if len(rows) < 2:
        return []

    vec = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(2, 5),
        min_df=1,
        max_df=0.95,
        max_features=8000,
    )

    X = vec.fit_transform(rows)
    if X.shape[1] == 0:
        return []

    scores = X.sum(axis=0).A1
    phrases = vec.get_feature_names_out()

    # Mildly favor longer phrases while preserving TF-IDF signal.
    weighted_scores = []
    for phrase, score in zip(phrases, scores):
        phrase_len = len(phrase.split())
        weighted_scores.append(score * (1 + 0.1 * (phrase_len - 2)))

    idx = sorted(range(len(weighted_scores)), key=lambda i: weighted_scores[i], reverse=True)[:top_k]
    return [phrases[i] for i in idx]


def is_short_positive_comment(comment: str) -> bool:
    normalized = " ".join(comment.strip().lower().split())
    if not normalized:
        return True

    words = normalized.split()
    if len(words) > 7:
        return False

    # Treat short comments with no sentence punctuation as "less than a sentence".
    has_sentence_punctuation = any(p in normalized for p in ".!?")
    if has_sentence_punctuation:
        return False

    if normalized in {"n/a", "na", "none"}:
        return True

    return any(marker in words for marker in SHORT_POSITIVE_MARKERS)


def filter_comments_for_cluster(comments: pd.Series, rating_name: str) -> pd.Series:
    cleaned = comments.fillna("").astype(str)
    cleaned = cleaned[cleaned.str.strip() != ""]

    if rating_name == "rating_4":
        cleaned = cleaned[~cleaned.apply(is_short_positive_comment)]

    return cleaned


def phase_cluster_report(df: pd.DataFrame, text_col: str) -> dict:
    clusters: list[dict] = []

    for start_str, end_str, phase_name in PHASE_WINDOWS:
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        date_mask = (df[DATE_COLUMN] >= start) & (df[DATE_COLUMN] <= end)
        phase_df = df.loc[date_mask]

        for ratings, rating_name in RATING_GROUPS:
            bucket = phase_df[phase_df[RATING_COLUMN].isin(ratings)]
            comments = filter_comments_for_cluster(bucket[text_col], rating_name)

            clusters.append(
                {
                    "phase": phase_name,
                    "date_range": {"start": start_str, "end": end_str},
                    "cluster": rating_name,
                    "ratings_included": ratings,
                    "rows_in_bucket": int(len(bucket)),
                    "non_empty_comments": int(len(comments)),
                    "top_phrases": top_phrases_from_rows(comments, top_k=12),
                }
            )

    return {
        "model": "TF-IDF keyphrase extraction (2-5 gram phrases)",
        "cluster_definition": "Date/rating buckets generated from configured phase and rating groups",
        "clusters": clusters,
    }


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    if DATE_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {DATE_COLUMN}")
    if RATING_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {RATING_COLUMN}")

    text_col = find_text_column(df)

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df[RATING_COLUMN] = pd.to_numeric(df[RATING_COLUMN], errors="coerce")

    df = df.dropna(subset=[DATE_COLUMN, RATING_COLUMN])
    df[RATING_COLUMN] = df[RATING_COLUMN].astype(int)

    report = phase_cluster_report(df, text_col)
    report["rows_used_after_cleaning"] = int(len(df))
    report["text_column"] = text_col

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
