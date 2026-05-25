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


def cluster_comments(comments: list[str], num_clusters: int, random_state: int = 42) -> tuple[list[int], KMeans, TfidfVectorizer]:
    """Cluster comments and return labels + fitted model/vectorizer."""
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    matrix = vectorizer.fit_transform(comments)

    model = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(matrix)

    return labels.tolist(), model, vectorizer


def print_cluster_summary(
    segment_title: str,
    comments: list[str],
    labels: list[int],
    vectorizer: TfidfVectorizer,
    model: KMeans,
    top_n_terms: int = 5,
) -> None:
    """Print cluster sizes, top terms, and sample comments."""
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

        print(f"Cluster {cluster_id}")
        print(f"- Size: {counts[cluster_id]}")
        print(f"- Top terms: {', '.join(top_terms)}")
        print(f"- Sample comment: {sample[:180]}")
        print("-" * 60)


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
        default=3,
        help="Number of clusters for high rating segment (default: 3)",
    )
    parser.add_argument(
        "--low-rating-clusters",
        type=int,
        default=3,
        help="Number of clusters for low rating segment (default: 3)",
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
    if not low_comments:
        raise ValueError("No non-empty comments found for the low-rating segment.")

    validate_cluster_count(
        args.high_rating_clusters,
        len(high_comments),
        "--high-rating-clusters",
    )
    validate_cluster_count(
        args.low_rating_clusters,
        len(low_comments),
        "--low-rating-clusters",
    )

    print(f"Loaded high-rating comments (>= {args.high_rating_threshold}): {len(high_comments)}")
    high_labels, high_model, high_vectorizer = cluster_comments(
        high_comments, args.high_rating_clusters
    )
    print_cluster_summary(
        f"Cluster Summary: Rating >= {args.high_rating_threshold}",
        high_comments,
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
