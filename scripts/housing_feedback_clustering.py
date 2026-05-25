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


def load_comments(csv_path: Path, comments_column: str = "Comments") -> list[str]:
    """Load non-empty comments from CSV."""
    comments: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if comments_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Column '{comments_column}' not found. Available columns: {reader.fieldnames}"
            )

        for row in reader:
            comment = (row.get(comments_column) or "").strip()
            if comment:
                comments.append(comment)

    if not comments:
        raise ValueError("No non-empty comments found in dataset.")

    return comments


def cluster_comments(comments: list[str], num_clusters: int, random_state: int = 42) -> tuple[list[int], KMeans, TfidfVectorizer]:
    """Cluster comments and return labels + fitted model/vectorizer."""
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    matrix = vectorizer.fit_transform(comments)

    model = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(matrix)

    return labels.tolist(), model, vectorizer


def print_cluster_summary(comments: list[str], labels: list[int], vectorizer: TfidfVectorizer, model: KMeans, top_n_terms: int = 5) -> None:
    """Print cluster sizes, top terms, and sample comments."""
    counts = Counter(labels)
    terms = vectorizer.get_feature_names_out()

    print("\nCluster Summary")
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
        "--clusters",
        type=int,
        default=3,
        help="Number of clusters to build (default: 3)",
    )
    parser.add_argument(
        "--top-terms",
        type=int,
        default=5,
        help="Top terms to display per cluster (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    comments = load_comments(args.dataset, args.comments_column)

    if args.clusters < 2:
        raise ValueError("--clusters must be at least 2.")
    if args.clusters > len(comments):
        raise ValueError(
            f"--clusters ({args.clusters}) cannot exceed number of comments ({len(comments)})."
        )

    labels, model, vectorizer = cluster_comments(comments, args.clusters)

    print(f"Loaded {len(comments)} comments from {args.dataset}")
    print_cluster_summary(comments, labels, vectorizer, model, args.top_terms)


if __name__ == "__main__":
    main()
