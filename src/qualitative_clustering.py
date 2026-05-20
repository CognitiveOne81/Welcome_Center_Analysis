"""Cluster qualitative visitor feedback comments.

This script reads the Welcome Center feedback CSV, filters useful text comments,
selects a KMeans cluster count using silhouette score, and writes analysis
artifacts for review.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score


DEFAULT_INPUT = Path("7-1-2024-5-19-2026 General Feedback.csv")
DEFAULT_OUTPUT_DIR = Path("outputs")
TEXT_COLUMN = "Comments"
RATING_COLUMN = "Rating"
EXCLUDED_OUTPUT_COLUMNS = {"Phone", "Email"}
EMPTY_RESPONSES = {
    "",
    "n/a",
    "na",
    "none",
    "no response",
    "no comment",
    "no comments",
    "nothing",
    "nil",
    "null",
}


@dataclass(frozen=True)
class ClusterSelection:
    cluster_count: int
    silhouette: float | None
    scores: dict[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run qualitative clustering analysis on Welcome Center feedback."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated analysis artifacts.",
    )
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def clean_comment(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_useful_comment(text: str) -> bool:
    normalized = text.lower().strip(" .!?-")
    return len(normalized) >= 8 and normalized not in EMPTY_RESPONSES


def load_feedback(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Expected a '{TEXT_COLUMN}' column in {path}")

    df["clean_comment"] = df[TEXT_COLUMN].map(clean_comment)
    df = df[df["clean_comment"].map(is_useful_comment)].copy()
    if df.empty:
        raise ValueError("No useful comments were found after cleaning.")

    if RATING_COLUMN in df.columns:
        df[RATING_COLUMN] = pd.to_numeric(df[RATING_COLUMN], errors="coerce")

    return df


def make_vectorizer(document_count: int) -> TfidfVectorizer:
    min_df = 2 if document_count >= 25 else 1
    custom_stop_words = set(ENGLISH_STOP_WORDS).union(
        {
            "tour",
            "campus",
            "unf",
            "osprey",
            "ospreys",
            "student",
            "students",
            "really",
            "just",
        }
    )
    return TfidfVectorizer(
        stop_words=list(custom_stop_words),
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.9,
        strip_accents="unicode",
    )


def choose_cluster_count(
    features,
    min_clusters: int,
    max_clusters: int,
    random_state: int,
) -> ClusterSelection:
    row_count = features.shape[0]
    if row_count < 3:
        return ClusterSelection(cluster_count=1, silhouette=None, scores={})

    min_k = max(2, min_clusters)
    max_k = min(max_clusters, row_count - 1)
    if min_k > max_k:
        min_k = max_k

    scores: dict[int, float] = {}
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(features)
        if len(set(labels)) > 1:
            scores[k] = float(silhouette_score(features, labels))

    if not scores:
        return ClusterSelection(cluster_count=min_k, silhouette=None, scores={})

    best_k = max(scores, key=scores.get)
    return ClusterSelection(cluster_count=best_k, silhouette=scores[best_k], scores=scores)


def top_terms_by_cluster(model: KMeans, vectorizer: TfidfVectorizer, top_n: int = 12) -> dict[int, list[str]]:
    terms = np.asarray(vectorizer.get_feature_names_out())
    top_terms: dict[int, list[str]] = {}
    for cluster_id, center in enumerate(model.cluster_centers_):
        indices = center.argsort()[::-1][:top_n]
        top_terms[cluster_id] = terms[indices].tolist()
    return top_terms


def representative_comments(
    df: pd.DataFrame,
    features,
    model: KMeans,
    cluster_id: int,
    limit: int = 3,
) -> list[str]:
    cluster_indices = np.flatnonzero(df["cluster"].to_numpy() == cluster_id)
    if len(cluster_indices) == 0:
        return []

    center = model.cluster_centers_[cluster_id]
    distances = np.linalg.norm(features[cluster_indices].toarray() - center, axis=1)
    nearest = cluster_indices[np.argsort(distances)[:limit]]
    return df.iloc[nearest]["clean_comment"].tolist()


def build_outputs(
    df: pd.DataFrame,
    features,
    model: KMeans,
    vectorizer: TfidfVectorizer,
    selection: ClusterSelection,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    top_terms = top_terms_by_cluster(model, vectorizer)
    output_columns = [c for c in df.columns if c not in EXCLUDED_OUTPUT_COLUMNS]
    clustered = df[output_columns].copy()
    clustered.to_csv(output_dir / "clustered_feedback.csv", index=False)

    summaries = []
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster_id]
        summaries.append(
            {
                "cluster": int(cluster_id),
                "count": int(len(cluster_df)),
                "share": round(float(len(cluster_df) / len(df)), 4),
                "average_rating": round(float(cluster_df[RATING_COLUMN].mean()), 3)
                if RATING_COLUMN in cluster_df
                else None,
                "top_terms": ", ".join(top_terms[int(cluster_id)]),
                "representative_comments": " | ".join(
                    representative_comments(df, features, model, int(cluster_id))
                ),
            }
        )
    pd.DataFrame(summaries).to_csv(output_dir / "cluster_summary.csv", index=False)

    term_rows = [
        {"cluster": cluster_id, "rank": rank + 1, "term": term}
        for cluster_id, terms in top_terms.items()
        for rank, term in enumerate(terms)
    ]
    pd.DataFrame(term_rows).to_csv(output_dir / "cluster_terms.csv", index=False)

    write_plot(df, features, output_dir)

    metrics = {
        "input_rows_after_cleaning": int(len(df)),
        "cluster_count": int(selection.cluster_count),
        "silhouette_score": selection.silhouette,
        "silhouette_scores_by_k": {str(k): v for k, v in selection.scores.items()},
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def write_plot(df: pd.DataFrame, features, output_dir: Path) -> None:
    if features.shape[0] < 2 or features.shape[1] < 2:
        return

    reducer = TruncatedSVD(n_components=2, random_state=42)
    points = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=df["cluster"], cmap="tab10", alpha=0.78)
    ax.set_title("Qualitative Feedback Clusters")
    ax.set_xlabel("Text component 1")
    ax.set_ylabel("Text component 2")
    ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_plot.png", dpi=160)
    plt.close(fig)


def run_analysis(
    input_path: Path,
    output_dir: Path,
    min_clusters: int,
    max_clusters: int,
    random_state: int,
) -> ClusterSelection:
    df = load_feedback(input_path)
    vectorizer = make_vectorizer(len(df))
    features = vectorizer.fit_transform(df["clean_comment"])

    selection = choose_cluster_count(features, min_clusters, max_clusters, random_state)
    model = KMeans(n_clusters=selection.cluster_count, n_init=20, random_state=random_state)
    df["cluster"] = model.fit_predict(features)

    build_outputs(df, features, model, vectorizer, selection, output_dir)
    return selection


def main() -> None:
    args = parse_args()
    selection = run_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        random_state=args.random_state,
    )
    score = "n/a" if selection.silhouette is None else f"{selection.silhouette:.4f}"
    print(f"Selected {selection.cluster_count} clusters with silhouette score {score}.")
    print(f"Artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
