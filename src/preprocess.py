import argparse
import ast
import gzip
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import (
    INTERACTIONS_FILE,
    ITEMS_FILE,
    METADATA_FILE,
    REVIEWS_FILE,
    ensure_directories,
)


REVIEW_COLUMNS = [
    "reviewerID",
    "asin",
    "overall",
    "reviewText",
    "summary",
    "unixReviewTime",
]

METADATA_COLUMNS = [
    "asin",
    "title",
    "brand",
    "category",
    "price",
    "description",
    "imageURL",
    "imageURLHighRes",
]


def _read_json_gz_lines(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield ast.literal_eval(line)


def _clean_list_value(value) -> str:
    if isinstance(value, list):
        flattened = []
        for item in value:
            if isinstance(item, list):
                flattened.extend(str(child) for child in item)
            else:
                flattened.append(str(item))
        return " | ".join(flattened)
    if pd.isna(value):
        return ""
    return str(value)


def load_reviews(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    rows = []
    for idx, record in enumerate(_read_json_gz_lines(path)):
        if max_rows is not None and idx >= max_rows:
            break
        rows.append({column: record.get(column) for column in REVIEW_COLUMNS})

    reviews = pd.DataFrame(rows)
    if reviews.empty:
        raise ValueError(f"No review rows were loaded from {path}")

    reviews = reviews.rename(
        columns={
            "reviewerID": "user_id",
            "asin": "item_id",
            "overall": "rating",
            "reviewText": "review_text",
            "summary": "review_title",
            "unixReviewTime": "timestamp",
        }
    )
    reviews = reviews.dropna(subset=["user_id", "item_id", "rating"])
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    reviews = reviews.dropna(subset=["rating"])
    reviews["rating"] = reviews["rating"].astype(float)
    reviews["review_text"] = reviews["review_text"].fillna("")
    reviews["review_title"] = reviews["review_title"].fillna("")
    reviews["timestamp"] = pd.to_numeric(reviews["timestamp"], errors="coerce").fillna(0).astype(int)
    return reviews.drop_duplicates(subset=["user_id", "item_id"], keep="last")


def load_metadata(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    rows = []
    for idx, record in enumerate(_read_json_gz_lines(path)):
        if max_rows is not None and idx >= max_rows:
            break
        rows.append({column: record.get(column) for column in METADATA_COLUMNS})

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise ValueError(f"No metadata rows were loaded from {path}")

    metadata = metadata.rename(columns={"asin": "item_id"})
    metadata = metadata.dropna(subset=["item_id"]).drop_duplicates(subset=["item_id"], keep="last")
    for column in ["title", "brand", "category", "price", "description", "imageURL", "imageURLHighRes"]:
        if column in metadata.columns:
            metadata[column] = metadata[column].apply(_clean_list_value)
    return metadata


def preprocess(
    reviews_file: Path = REVIEWS_FILE,
    metadata_file: Path = METADATA_FILE,
    max_review_rows: int | None = None,
    max_metadata_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_directories()

    if not reviews_file.exists():
        raise FileNotFoundError(
            f"Missing review file: {reviews_file}. Put Video_Games_5.json.gz in the raw data folder."
        )

    interactions = load_reviews(reviews_file, max_rows=max_review_rows)
    interactions.to_csv(INTERACTIONS_FILE, index=False)

    if metadata_file.exists():
        items = load_metadata(metadata_file, max_rows=max_metadata_rows)
        items = items[items["item_id"].isin(interactions["item_id"].unique())]
    else:
        items = pd.DataFrame({"item_id": sorted(interactions["item_id"].unique())})
    items.to_csv(ITEMS_FILE, index=False)

    return interactions, items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-review-rows", type=int, default=None)
    parser.add_argument("--max-metadata-rows", type=int, default=None)
    args = parser.parse_args()
    interactions, items = preprocess(
        max_review_rows=args.max_review_rows,
        max_metadata_rows=args.max_metadata_rows,
    )
    print(f"Saved {len(interactions):,} interactions to {INTERACTIONS_FILE}")
    print(f"Saved {len(items):,} items to {ITEMS_FILE}")


if __name__ == "__main__":
    main()
