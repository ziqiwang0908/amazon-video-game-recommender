import html
import json
import re
import shutil
from pathlib import Path

import pandas as pd

from config import DEMO_DATA_DIR, ITEMS_FILE, METRICS_FILE, RESULTS_DIR, TOP10_RECOMMENDATIONS_FILE, TRAIN_FILE


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]*>?", " ", text)
    text = text.replace("by\n", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_price(value) -> str:
    text = clean_text(value)
    if not text or text.startswith("<"):
        return ""
    if re.search(r"\$\s*\d", text) or re.fullmatch(r"\d+(\.\d+)?", text):
        return text[:24]
    return ""


def first_image_url(row: pd.Series) -> str:
    for column in ["imageURLHighRes", "imageURL"]:
        value = row.get(column, "")
        if pd.isna(value):
            continue
        for part in str(value).split(" | "):
            part = part.strip()
            if part.startswith("http"):
                return part
    return ""


def build_item_lookup(items: pd.DataFrame) -> dict:
    if items.empty:
        return {}
    items = items.copy()
    items["image_url"] = items.apply(first_image_url, axis=1)
    cleaned = []
    for _, row in items.iterrows():
        item_id = row["item_id"]
        cleaned.append(
            {
                "item_id": item_id,
                "title": clean_text(row.get("title", "")) or item_id,
                "brand": clean_text(row.get("brand", "")),
                "category": clean_text(row.get("category", "")),
                "price": clean_price(row.get("price", "")),
                "image_url": clean_text(row.get("image_url", "")),
            }
        )
    return {row["item_id"]: row for row in cleaned}


def item_payload(item_id: str, item_lookup: dict) -> dict:
    return item_lookup.get(
        item_id,
        {
            "item_id": item_id,
            "title": item_id,
            "brand": "",
            "category": "",
            "price": "",
            "image_url": "",
        },
    ).copy()


def write_json(filename: str, payload) -> None:
    path = RESULTS_DIR / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    shutil.copy2(path, DEMO_DATA_DIR / filename)


def export_demo_data(
    train_file: Path = TRAIN_FILE,
    items_file: Path = ITEMS_FILE,
    metrics_file: Path = METRICS_FILE,
    recommendations_file: Path = TOP10_RECOMMENDATIONS_FILE,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_file)
    items = pd.read_csv(items_file) if items_file.exists() else pd.DataFrame(columns=["item_id"])
    recommendations = pd.read_csv(recommendations_file)
    item_lookup = build_item_lookup(items)

    demo_users = recommendations["user_id"].drop_duplicates().tolist()
    demo_user_set = set(demo_users)

    recommendation_rows = []
    used_item_ids = set()
    for user_id, group in recommendations.groupby("user_id", sort=False):
        rows = []
        for _, row in group.sort_values("rank").iterrows():
            item_id = row["item_id"]
            used_item_ids.add(item_id)
            payload = item_payload(item_id, item_lookup)
            payload.update(
                {
                    "rank": int(row["rank"]),
                    "predicted_rating": float(row["predicted_rating"]),
                    "explanation": clean_text(row.get("explanation", "")),
                }
            )
            rows.append(payload)
        recommendation_rows.append({"user_id": user_id, "recommendations": rows})

    history_rows = []
    history_source = train[(train["user_id"].isin(demo_user_set)) & (train["rating"] >= 4.0)].copy()
    history_source = history_source.sort_values(["user_id", "rating"], ascending=[True, False])
    for user_id in demo_users:
        group = history_source[history_source["user_id"] == user_id].head(8)
        rows = []
        for _, row in group.iterrows():
            item_id = row["item_id"]
            used_item_ids.add(item_id)
            payload = item_payload(item_id, item_lookup)
            payload["rating"] = float(row["rating"])
            rows.append(payload)
        history_rows.append({"user_id": user_id, "history": rows})

    item_rows = [item_payload(item_id, item_lookup) for item_id in sorted(used_item_ids)]

    write_json("user_history.json", history_rows)
    write_json("recommendations.json", recommendation_rows)
    write_json("item_metadata.json", item_rows)

    if metrics_file.exists():
        shutil.copy2(metrics_file, DEMO_DATA_DIR / "metrics.json")


if __name__ == "__main__":
    export_demo_data()
