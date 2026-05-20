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
POSITIVE_RATING = 4
SHORT_POSITIVE_WORD_LIMIT = 5


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


def is_college_tour(tour_type: str) -> bool:
    tour_label = (tour_type or "").strip()
    return tour_label in TARGET_TOURS or "college" in tour_label.lower()


def main() -> None:
    with DATA_FILE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        text_col = find_text_column(reader.fieldnames or [])

        filtered = []
        for row in reader:
            tour_type = row.get(TOUR_TYPE_COLUMN, "") or ""
            if not is_college_tour(tour_type):
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

    positive_comments = [text for rt, text in filtered if rt == POSITIVE_RATING and text]
    short_positive_comments = [
        text for text in positive_comments if len(tokenize(text)) <= SHORT_POSITIVE_WORD_LIMIT
    ]
    notable_long_positive_comments = [
        text for text in positive_comments if len(tokenize(text)) > SHORT_POSITIVE_WORD_LIMIT
    ]

    print(
        json.dumps(
            {
                "tour_set": sorted(TARGET_TOURS),
                "college_filter": "exact target tours OR any tour type containing 'college'",
                "rows_used_after_cleaning": len(filtered),
                "text_column": text_col,
                "positive_rating": POSITIVE_RATING,
                "short_positive_word_limit": SHORT_POSITIVE_WORD_LIMIT,
                "positive_non_empty_comments": len(positive_comments),
                "short_positive_comments": {
                    "count": len(short_positive_comments),
                    "top_phrases": extract_top_phrases(short_positive_comments, top_k=3),
                },
                "notable_long_positive_comments": {
                    "count": len(notable_long_positive_comments),
                    "examples": notable_long_positive_comments[:10],
                    "top_phrases": extract_top_phrases(notable_long_positive_comments, top_k=5),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
