#!/usr/bin/env python3
"""Cluster housing feedback comments into themes using TF-IDF + KMeans."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_DATASET = Path("7-1-2024 -5-19-2026 Housing Feedback.csv")


def load_comments_by_rating(
    csv_path: Path,
    comments_column: str = "Comments",
    rating_column: str = "Rating",
    high_rating_threshold: float = 4.0,
) -> tuple[list[str], list[str]]:
    """Load non-empty comments, split into >= threshold and < threshold rating groups."""
    high_rating_comments: list[str] = []
    low_rating_comments: list[str] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []

        if comments_column not in fieldnames:
            raise ValueError(
                f"Column '{comments_column}' not found. Available columns: {fieldnames}"
            )
        if rating_column not in fieldnames:
            raise ValueError(
                f"Column '{rating_column}' not found. Available columns: {fieldnames}"
            )

        for row in reader:
            comment = (row.get(comments_column) or "").strip()
            rating_raw = (row.get(rating_column) or "").strip()

            if not comment or not rating_raw:
                continue

            try:
                rating_value = float(rating_raw)
            except ValueError:
                continue

            if rating_value >= high_rating_threshold:
                high_rating_comments.append(comment)
            else:
                low_rating_comments.append(comment)

    return high_rating_comments, low_rating_comments


def cluster_comments(
    comments: list[str],
    num_clusters: int,
    random_state: int = 42,
    ngram_range: tuple[int, int] = (1, 3),
) -> tuple[list[int], KMeans, TfidfVectorizer]:
    """Cluster comments and return labels + fitted model/vectorizer."""
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=ngram_range)
    matrix = vectorizer.fit_transform(comments)

    model = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(matrix)

    return labels.tolist(), model, vectorizer


def choose_cluster_heading(terms: list[str]) -> str:
    """Prefer a multi-word phrase for the cluster heading when available."""
    for term in terms:
        if " " in term:
            return term
    return terms[0] if terms else "n/a"


def print_cluster_summary(
    segment_title: str,
    comments: list[str],
    labels: list[int],
    vectorizer: TfidfVectorizer,
    model: KMeans,
    top_n_terms: int = 5,
) -> None:
    """Print cluster sizes, phrase-led headings, and sample comments."""
    counts = Counter(labels)
    terms = vectorizer.get_feature_names_out()

    print(f"\n{segment_title}")
    print("=" * 60)

    for cluster_id in sorted(counts):
        center = model.cluster_centers_[cluster_id]
        top_indices = center.argsort()[-top_n_terms:][::-1]
        top_terms = [terms[i] for i in top_indices]

        cluster_comments_list = [c for c, l in zip(comments, labels) if l == cluster_id]
        sample = cluster_comments_list[0] if cluster_comments_list else ""

        heading = choose_cluster_heading(top_terms)

        print(f"Cluster {cluster_id}: {heading}")
        print(f"- Size: {counts[cluster_id]}")
        print(f"- Top phrases/terms: {', '.join(top_terms)}")
        print(f"- Sample comment: {sample[:180]}")
        print("-" * 60)


def is_extremely_positive_short_comment(comment: str, max_words: int = 6) -> bool:
    """Heuristic to remove very short, highly positive comments from high-rating set."""
    positive_markers = {
        "great", "excellent", "awesome", "amazing", "perfect", "wonderful",
        "fantastic", "love", "loved", "good", "nice", "best", "outstanding",
    }

    words = [w.strip(".,!?;:\"'()[]{}").lower() for w in comment.split()]
    words = [w for w in words if w]

    if len(words) > max_words or not words:
        return False

    positive_hits = sum(1 for w in words if w in positive_markers)
    ratio = positive_hits / len(words)

    return positive_hits >= 2 or ratio >= 0.5


def filter_high_rating_comments_for_depth(comments: list[str]) -> list[str]:
    """Keep longer/more descriptive high-rating comments by removing short praise blurbs."""
    filtered = [c for c in comments if not is_extremely_positive_short_comment(c)]
    return filtered or comments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster housing feedback comments using KMeans."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to CSV dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--comments-column",
        default="Comments",
        help="Name of the text column to cluster (default: Comments)",
    )
    parser.add_argument(
        "--rating-column",
        default="Rating",
        help="Name of the rating column used for splitting (default: Rating)",
    )
    parser.add_argument(
        "--high-rating-threshold",
        type=float,
        default=4.0,
        help="Threshold for high rating segment (>= threshold). Default: 4",
    )
    parser.add_argument(
        "--high-rating-clusters",
        type=int,
        default=5,
        help="Number of clusters for high rating segment (default: 5)",
    )
    parser.add_argument(
        "--low-rating-clusters",
        type=int,
        default=5,
        help="Number of clusters for low rating segment (default: 5)",
    )
    parser.add_argument(
        "--top-terms",
        type=int,
        default=5,
        help="Top terms to display per cluster (default: 5)",
    )
    return parser.parse_args()


def validate_cluster_count(cluster_count: int, comments_count: int, label: str) -> None:
    if cluster_count < 2:
        raise ValueError(f"{label} must be at least 2.")
    if cluster_count > comments_count:
        raise ValueError(
            f"{label} ({cluster_count}) cannot exceed number of comments in that segment ({comments_count})."
        )


def main() -> None:
    args = parse_args()

    high_comments, low_comments = load_comments_by_rating(
        args.dataset,
        comments_column=args.comments_column,
        rating_column=args.rating_column,
        high_rating_threshold=args.high_rating_threshold,
    )

    if not high_comments:
        raise ValueError("No non-empty comments found for the high-rating segment.")

    filtered_high_comments = filter_high_rating_comments_for_depth(high_comments)
    if not low_comments:
        raise ValueError("No non-empty comments found for the low-rating segment.")

    validate_cluster_count(
        args.high_rating_clusters,
        len(filtered_high_comments),
        "--high-rating-clusters",
    )
    validate_cluster_count(
        args.low_rating_clusters,
        len(low_comments),
        "--low-rating-clusters",
    )

    removed_short_positive = len(high_comments) - len(filtered_high_comments)
    print(
        f"Loaded high-rating comments (>= {args.high_rating_threshold}): {len(high_comments)} "
        f"(filtered out {removed_short_positive} very short/extremely positive comments)"
    )
    high_labels, high_model, high_vectorizer = cluster_comments(
        filtered_high_comments, args.high_rating_clusters
    )
    print_cluster_summary(
        f"Cluster Summary: Rating >= {args.high_rating_threshold}",
        filtered_high_comments,
        high_labels,
        high_vectorizer,
        high_model,
        args.top_terms,
    )

    print(f"\nLoaded low-rating comments (< {args.high_rating_threshold}): {len(low_comments)}")
    low_labels, low_model, low_vectorizer = cluster_comments(
        low_comments, args.low_rating_clusters
    )
    print_cluster_summary(
        f"Cluster Summary: Rating < {args.high_rating_threshold}",
        low_comments,
        low_labels,
        low_vectorizer,
        low_model,
        args.top_terms,
    )


if __name__ == "__main__":
    main()
