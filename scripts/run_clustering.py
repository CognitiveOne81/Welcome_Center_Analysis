#!/usr/bin/env python3
"""Cluster housing feedback comments into themes using TF-IDF + KMeans."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Optional

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


DEFAULT_DATASET = Path("7-1-2024 -5-19-2026 Housing Feedback.csv")

GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"


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


def select_cluster_count(
    matrix,
    min_k: int = 2,
    max_k: int = 10,
    random_state: int = 42,
) -> tuple[int, list[dict[str, float]]]:
    """Select cluster count by silhouette score across candidate k values."""
    n_samples = matrix.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 comments to evaluate cluster counts.")

    max_candidate = min(max_k, n_samples - 1)
    min_candidate = max(2, min_k)
    if min_candidate > max_candidate:
        raise ValueError("Not enough comments to evaluate requested k sweep range.")

    best_k: Optional[int] = None
    best_score = float("-inf")
    diagnostics: list[dict[str, float]] = []

    for k in range(min_candidate, max_candidate + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(matrix)
        score = float(silhouette_score(matrix, labels))
        diagnostics.append({"k": float(k), "silhouette": score})

        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        raise ValueError("Unable to select a cluster count from candidate range.")

    return best_k, diagnostics


def cluster_comments(
    comments: list[str],
    num_clusters: Optional[int],
    random_state: int = 42,
    ngram_range: tuple[int, int] = (1, 3),
    k_min: int = 2,
    k_max: int = 10,
) -> tuple[list[int], KMeans, TfidfVectorizer, int, list[dict[str, float]]]:
    """Cluster comments and return labels/model/vectorizer plus k-selection diagnostics."""
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=ngram_range)
    matrix = vectorizer.fit_transform(comments)

    diagnostics: list[dict[str, float]] = []
    selected_k = num_clusters if num_clusters is not None else 0

    if num_clusters is None:
        selected_k, diagnostics = select_cluster_count(
            matrix,
            min_k=k_min,
            max_k=k_max,
            random_state=random_state,
        )

    model = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
    labels = model.fit_predict(matrix)
    return labels.tolist(), model, vectorizer, selected_k, diagnostics


def choose_cluster_heading(terms: list[str]) -> str:
    """Prefer a multi-word phrase for the cluster heading when available."""
    for term in terms:
        if " " in term:
            return term
    return terms[0] if terms else "n/a"


def choose_heading_from_centroid_neighbors(
    comments: list[str],
    labels: list[int],
    cluster_id: int,
    matrix,
    centroid,
    fallback_terms: list[str],
    min_phrase_doc_count: int = 2,
) -> str:
    """Label cluster from phrases shared across comments nearest to cluster centroid."""
    cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
    if not cluster_indices:
        return choose_cluster_heading(fallback_terms)

    cluster_matrix = matrix[cluster_indices]
    similarity = (cluster_matrix @ centroid.reshape(-1, 1)).ravel()
    nearest_local = similarity.argsort()[::-1][: min(10, len(cluster_indices))]
    nearest_comments = [comments[cluster_indices[i]] for i in nearest_local]

    local_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(2, 3),
        min_df=1,
        binary=True,
    )
    try:
        phrase_matrix = local_vectorizer.fit_transform(nearest_comments)
    except ValueError:
        return choose_cluster_heading(fallback_terms)

    phrases = local_vectorizer.get_feature_names_out()
    doc_counts = phrase_matrix.sum(axis=0).A1
    valid_indices = [i for i, c in enumerate(doc_counts) if c >= min_phrase_doc_count]

    if valid_indices:
        ranked = sorted(valid_indices, key=lambda i: doc_counts[i], reverse=True)
        for idx in ranked:
            phrase = phrases[idx].strip()
            if phrase and len(phrase.split()) >= 2:
                return phrase

    return choose_cluster_heading(fallback_terms)


def print_cluster_summary(
    segment_title: str,
    segment_color: str,
    comments: list[str],
    labels: list[int],
    vectorizer: TfidfVectorizer,
    model: KMeans,
    top_n_terms: int = 5,
) -> None:
    """Print cluster sizes, phrase-led headings, and sample comments."""
    counts = Counter(labels)
    terms = vectorizer.get_feature_names_out()
    matrix = vectorizer.transform(comments)

    print(f"\n{segment_color}{segment_title}{RESET}")
    print(f"{segment_color}{"=" * 60}{RESET}")

    for cluster_id in sorted(counts):
        center = model.cluster_centers_[cluster_id]
        top_indices = center.argsort()[-top_n_terms:][::-1]
        top_terms = [terms[i] for i in top_indices]

        cluster_comments_list = [c for c, l in zip(comments, labels) if l == cluster_id]
        sample = cluster_comments_list[0] if cluster_comments_list else ""

        heading = choose_heading_from_centroid_neighbors(
            comments,
            labels,
            cluster_id,
            matrix,
            center,
            fallback_terms=top_terms,
        )

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
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--comments-column", default="Comments")
    parser.add_argument("--rating-column", default="Rating")
    parser.add_argument("--high-rating-threshold", type=float, default=4.0)
    parser.add_argument("--high-rating-clusters", type=int, default=None)
    parser.add_argument("--low-rating-clusters", type=int, default=None)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--top-terms", type=int, default=5)
    return parser.parse_args()


def validate_cluster_count(cluster_count: int, comments_count: int, label: str) -> None:
    if cluster_count < 2:
        raise ValueError(f"{label} must be at least 2.")
    if cluster_count >= comments_count:
        raise ValueError(
            f"{label} ({cluster_count}) must be less than number of comments in that segment ({comments_count})."
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

    if args.high_rating_clusters is not None:
        validate_cluster_count(args.high_rating_clusters, len(filtered_high_comments), "--high-rating-clusters")
    if args.low_rating_clusters is not None:
        validate_cluster_count(args.low_rating_clusters, len(low_comments), "--low-rating-clusters")

    removed_short_positive = len(high_comments) - len(filtered_high_comments)
    print(
        f"Loaded high-rating comments (>= {args.high_rating_threshold}): {len(high_comments)} "
        f"(filtered out {removed_short_positive} very short/extremely positive comments)"
    )
    high_labels, high_model, high_vectorizer, high_k, high_diag = cluster_comments(
        filtered_high_comments,
        args.high_rating_clusters,
        k_min=args.k_min,
        k_max=args.k_max,
    )
    print(f"Selected high-rating k: {high_k}")
    if high_diag:
        print(f"k sweep (high): {high_diag}")

    print_cluster_summary(
        f"Cluster Summary: Rating >= {args.high_rating_threshold}",
        GREEN,
        filtered_high_comments,
        high_labels,
        high_vectorizer,
        high_model,
        args.top_terms,
    )

    print(f"\nLoaded low-rating comments (< {args.high_rating_threshold}): {len(low_comments)}")
    low_labels, low_model, low_vectorizer, low_k, low_diag = cluster_comments(
        low_comments,
        args.low_rating_clusters,
        k_min=args.k_min,
        k_max=args.k_max,
    )
    print(f"Selected low-rating k: {low_k}")
    if low_diag:
        print(f"k sweep (low): {low_diag}")

    print_cluster_summary(
        f"Cluster Summary: Rating < {args.high_rating_threshold}",
        BLUE,
        low_comments,
        low_labels,
        low_vectorizer,
        low_model,
        args.top_terms,
    )


if __name__ == "__main__":
    main()
