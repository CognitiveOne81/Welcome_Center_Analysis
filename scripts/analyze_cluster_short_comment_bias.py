#!/usr/bin/env python3
"""Analyze whether short/simple comments disproportionately drive cluster phrases."""
import csv
import datetime as dt
import json
import re
from collections import Counter
from pathlib import Path

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
SHORT_WORD_LIMIT = 4
TOP_K = 12


def find_text_column(columns: list[str]) -> str:
    for col in TEXT_COLUMN_CANDIDATES:
        if col in columns:
            return col
    raise ValueError("No supported text column found.")


def parse_date(value: str):
    token = (value or "").strip().split()[0] if value else ""
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(token, fmt).date()
        except ValueError:
            pass
    return None


def clean_text(value: str) -> str:
    value = (value or "").strip()
    return "" if value.lower() in {"", "n/a", "na", "none", "-"} else value


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def ngrams(tokens: list[str], n: int):
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def extract_top_phrases(comments: list[str], top_k: int = TOP_K) -> list[str]:
    counts = Counter()
    for text in comments:
        tokens = tokenize(text)
        if len(tokens) < 2:
            continue
        seen = set()
        for n in range(2, 6):
            for phrase in ngrams(tokens, n):
                seen.add(phrase)
        counts.update(seen)
    return [phrase for phrase, _ in counts.most_common(top_k)]


def phrase_short_bias(phrases: list[str], comments: list[str]) -> list[dict]:
    rows = []
    for phrase in phrases:
        total = 0
        short_hits = 0
        for text in comments:
            tokens = tokenize(text)
            if phrase in " ".join(tokens):
                total += 1
                if len(tokens) <= SHORT_WORD_LIMIT:
                    short_hits += 1
        rows.append({
            "phrase": phrase,
            "comment_hits": total,
            "short_comment_hits": short_hits,
            "short_hit_share": round(short_hits / total, 3) if total else 0.0,
        })
    return rows


def main() -> None:
    with DATA_FILE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        text_col = find_text_column(reader.fieldnames or [])
        data = []
        for row in reader:
            d = parse_date(row.get(DATE_COLUMN, ""))
            if d is None:
                continue
            try:
                rating = int(float((row.get(RATING_COLUMN, "") or "").strip()))
            except ValueError:
                continue
            text = clean_text(row.get(text_col, ""))
            if text:
                data.append((d, rating, text))

    clusters = []
    for start_s, end_s, phase_name in PHASE_WINDOWS:
        start = dt.datetime.strptime(start_s, "%Y-%m-%d").date()
        end = dt.datetime.strptime(end_s, "%Y-%m-%d").date()
        phase = [r for r in data if start <= r[0] <= end]
        for ratings, name in RATING_GROUPS:
            comments = [t for _, rt, t in phase if rt in ratings]
            short_count = sum(1 for t in comments if len(tokenize(t)) <= SHORT_WORD_LIMIT)
            top = extract_top_phrases(comments, TOP_K)
            clusters.append({
                "phase": phase_name,
                "cluster": name,
                "ratings_included": ratings,
                "non_empty_comments": len(comments),
                "short_comment_count": short_count,
                "short_comment_share": round(short_count / len(comments), 3) if comments else 0.0,
                "top_phrases": top,
                "top_phrase_short_comment_bias": phrase_short_bias(top, comments),
            })

    print(json.dumps({
        "short_comment_word_limit": SHORT_WORD_LIMIT,
        "rows_with_non_empty_comments": len(data),
        "text_column": text_col,
        "clusters": clusters,
    }, indent=2))


if __name__ == "__main__":
    main()
