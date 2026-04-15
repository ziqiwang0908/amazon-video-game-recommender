import argparse
import json

import pandas as pd

from baseline import MeanRatingBaseline
from config import (
    METRICS_FILE,
    NEIGHBORHOOD_SIZE,
    POPULARITY_RECOMMENDATIONS_FILE,
    RATING_PREDICTIONS_FILE,
    TEST_FILE,
    TOP10_RECOMMENDATIONS_FILE,
    TOP_N,
    TRAIN_FILE,
    ensure_directories,
)
from evaluate import rating_metrics, top_n_metrics
from export_demo_data import export_demo_data
from item_cf import ItemItemCF
from preprocess import preprocess
from recommend import generate_popularity_recommendations, generate_recommendations
from split_data import create_split


def run_pipeline(
    max_review_rows: int | None = None,
    max_metadata_rows: int | None = None,
    max_recommendation_users: int | None = 300,
    max_candidate_items: int | None = 1000,
    skip_preprocess: bool = False,
) -> dict:
    ensure_directories()
    if max_recommendation_users is not None and max_recommendation_users <= 0:
        max_recommendation_users = None
    if max_candidate_items is not None and max_candidate_items <= 0:
        max_candidate_items = None

    if not skip_preprocess:
        preprocess(
            max_review_rows=max_review_rows,
            max_metadata_rows=max_metadata_rows,
        )
        create_split()

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    baseline = MeanRatingBaseline().fit(train)
    baseline_predictions = baseline.predict_frame(test)
    baseline_metrics = rating_metrics(baseline_predictions)

    item_cf = ItemItemCF(n_neighbors=NEIGHBORHOOD_SIZE).fit(train)
    item_cf_predictions = item_cf.predict_frame(test)
    item_cf_predictions.to_csv(RATING_PREDICTIONS_FILE, index=False)
    item_cf_rating_metrics = rating_metrics(item_cf_predictions)

    recommendations = generate_recommendations(
        item_cf,
        train,
        top_n=TOP_N,
        max_users=max_recommendation_users,
        max_candidates=max_candidate_items,
    )
    recommendations.to_csv(TOP10_RECOMMENDATIONS_FILE, index=False)
    item_cf_top_n_metrics = top_n_metrics(recommendations, test, top_n=TOP_N)

    popularity_recommendations = generate_popularity_recommendations(
        train,
        top_n=TOP_N,
        max_users=max_recommendation_users,
        max_candidates=max_candidate_items,
    )
    popularity_recommendations.to_csv(POPULARITY_RECOMMENDATIONS_FILE, index=False)
    popularity_top_n_metrics = top_n_metrics(popularity_recommendations, test, top_n=TOP_N)

    metrics = {
        "dataset": {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "num_users": int(train["user_id"].nunique()),
            "num_items": int(train["item_id"].nunique()),
        },
        "baseline": baseline_metrics,
        "item_item_cf": {
            **item_cf_rating_metrics,
            **item_cf_top_n_metrics,
        },
        "popularity_top_n": popularity_top_n_metrics,
        "settings": {
            "top_n": TOP_N,
            "neighborhood_size": NEIGHBORHOOD_SIZE,
            "max_recommendation_users": max_recommendation_users,
            "max_candidate_items": max_candidate_items,
        },
    }

    with METRICS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    export_demo_data()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-review-rows", type=int, default=None)
    parser.add_argument("--max-metadata-rows", type=int, default=None)
    parser.add_argument("--max-recommendation-users", type=int, default=300)
    parser.add_argument("--max-candidate-items", type=int, default=1000)
    parser.add_argument("--skip-preprocess", action="store_true")
    args = parser.parse_args()

    metrics = run_pipeline(
        max_review_rows=args.max_review_rows,
        max_metadata_rows=args.max_metadata_rows,
        max_recommendation_users=args.max_recommendation_users,
        max_candidate_items=args.max_candidate_items,
        skip_preprocess=args.skip_preprocess,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
