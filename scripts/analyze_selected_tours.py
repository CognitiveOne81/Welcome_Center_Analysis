#!/usr/bin/env python3
"""Extract top phrases for a combined set of selected tours."""
import csv
import datetime as dt
import json
import re
from collections import Counter
from pathlib import Path

DATA_FILE = Path("7-1-2024-5-19-2026 General Feedback.csv")
DATE_COLUMN = "Date"
RATING_COLUMN = "Rating"
TOUR_TYPE_COLUMN = "What type of tour did you take?"
TEXT_COLUMN_CANDIDATES = ["Comments", "Feedback", "comment", "text"]

TARGET_TOURS = {
    "College of Computing, Engineering, and Construction Tour",
    "Brooks College of Health Tour",
    "College of Arts and Sciences Tour",
    "Silverfield College of Education and Human Services Tour",
    "Coggin College of Business Tour",
}
RATING_GROUPS = [
    ([4], "rating_4"),
    ([0, 1, 2, 3], "rating_0_1_2_3"),
]


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


def extract_top_phrases(comments: list[str], top_k: int = 12) -> list[str]:
    counts = Counter()
    for text in comments:
        tokens = tokenize(text)
        if len(tokens) < 2:
            continue
        seen = set()
        for n in range(2, 6):
            for i in range(len(tokens) - n + 1):
                seen.add(" ".join(tokens[i : i + n]))
        counts.update(seen)
    return [phrase for phrase, _ in counts.most_common(top_k)]


def main() -> None:
    with DATA_FILE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        text_col = find_text_column(reader.fieldnames or [])

        filtered = []
        for row in reader:
            if (row.get(TOUR_TYPE_COLUMN, "") or "").strip() not in TARGET_TOURS:
                continue
            d = parse_date(row.get(DATE_COLUMN, ""))
            if d is None:
                continue
            try:
                rating = int(float((row.get(RATING_COLUMN, "") or "").strip()))
            except ValueError:
                continue
            text = clean_text(row.get(text_col, ""))
            filtered.append((rating, text))

    clusters = []
    for ratings, name in RATING_GROUPS:
        bucket = [text for rt, text in filtered if rt in ratings]
        comments = [text for text in bucket if text]
        top_k = 3 if name == "rating_4" else 2
        clusters.append(
            {
                "cluster": name,
                "ratings_included": ratings,
                "rows_in_bucket": len(bucket),
                "non_empty_comments": len(comments),
                "top_phrases": extract_top_phrases(comments, top_k=top_k),
            }
        )

    print(
        json.dumps(
            {
                "tour_set": sorted(TARGET_TOURS),
                "rows_used_after_cleaning": len(filtered),
                "text_column": text_col,
                "clusters": clusters,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
