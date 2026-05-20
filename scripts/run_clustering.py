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

TOUR_TYPE_COLUMN = "What type of tour did you take?"
TARGET_TOUR_TYPE = "Weekday campus Tour"
RATING_GROUPS = [
    ([4], "rating_4"),
    ([0, 1, 2, 3], "rating_0_1_2_3"),
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

    weekday_tour_mask = (
        df[TOUR_TYPE_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        == TARGET_TOUR_TYPE.casefold()
    )
    tour_df = df.loc[weekday_tour_mask]

    for ratings, rating_name in RATING_GROUPS:
        bucket = tour_df[tour_df[RATING_COLUMN].isin(ratings)]
        comments = filter_comments_for_cluster(bucket[text_col], rating_name)
        cluster_count = 3 if rating_name == "rating_4" else 2

        clusters.append(
            {
                "tour_type": TARGET_TOUR_TYPE,
                "cluster": rating_name,
                "ratings_included": ratings,
                "rows_in_bucket": int(len(bucket)),
                "non_empty_comments": int(len(comments)),
                "top_phrases": top_phrases_from_rows(comments, top_k=cluster_count),
            }
        )

    return {
        "model": "TF-IDF keyphrase extraction (2-5 gram phrases)",
        "cluster_definition": "Tour-type and rating buckets generated from configured target tour and rating groups",
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
    if TOUR_TYPE_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {TOUR_TYPE_COLUMN}")

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
