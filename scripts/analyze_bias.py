#!/usr/bin/env python3
"""Check rating/text-length bias in comment coverage."""
import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from statistics import median

DATA_FILE = Path("7-1-2024-5-19-2026 General Feedback.csv")
DATE_COLUMN = "Date"
RATING_COLUMN = "Rating"
TEXT_COLUMN_CANDIDATES = ["Comments", "Feedback", "comment", "text"]


def find_text_column(columns: list[str]) -> str:
    for col in TEXT_COLUMN_CANDIDATES:
        if col in columns:
            return col
    raise ValueError("No supported text column found.")


def parse_date(value: str):
    if not value:
        return None
    token = value.strip().split()[0]
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(token, fmt).date()
        except ValueError:
            pass
    return None


def clean_text(value: str) -> str:
    value = (value or "").strip()
    return "" if value.lower() in {"", "n/a", "na", "none", "-"} else value


def summary(rows: list[str]) -> dict:
    if not rows:
        return {"non_empty_comments": 0}
    word_counts = [len(text.split()) for text in rows]
    short_count = sum(1 for c in word_counts if c <= 4)
    unique = len(set(rows))
    return {
        "non_empty_comments": len(rows),
        "mean_words": round(sum(word_counts) / len(word_counts), 2),
        "median_words": median(word_counts),
        "short_simple_share": round(short_count / len(rows), 3),
        "unique_comments": unique,
        "repeat_share": round(1 - (unique / len(rows)), 3),
    }


def main() -> None:
    with DATA_FILE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        text_col = find_text_column(reader.fieldnames or [])

        cleaned = []
        for row in reader:
            date = parse_date(row.get(DATE_COLUMN, ""))
            if date is None:
                continue
            try:
                rating = int(float((row.get(RATING_COLUMN, "") or "").strip()))
            except ValueError:
                continue
            cleaned.append((rating, clean_text(row.get(text_col, ""))))

    by_rating = {}
    for rating in [0, 1, 2, 3, 4]:
        bucket = [text for rt, text in cleaned if rt == rating]
        non_empty = [text for text in bucket if text]
        stats = summary(non_empty)
        stats["rows_in_rating"] = len(bucket)
        stats["comment_rate"] = round(len(non_empty) / len(bucket), 3) if bucket else 0.0
        by_rating[str(rating)] = stats

    low = [text for rt, text in cleaned if rt in {0, 1, 2} and text]
    high = [text for rt, text in cleaned if rt in {3, 4} and text]

    out = {
        "rows_used_after_cleaning": len(cleaned),
        "text_column": text_col,
        "rating_breakdown": by_rating,
        "group_comparison": {
            "ratings_0_1_2": summary(low),
            "ratings_3_4": summary(high),
        },
        "top_repeated_low_comments": [
            {"comment": comment, "count": count}
            for comment, count in Counter(low).most_common(10)
            if count > 1
        ],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
