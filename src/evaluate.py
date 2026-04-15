import math

import numpy as np
import pandas as pd


def rating_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    errors = predictions["rating"].astype(float) - predictions["prediction"].astype(float)
    mae = float(np.abs(errors).mean())
    rmse = float(np.sqrt(np.square(errors).mean()))
    return {"mae": mae, "rmse": rmse}


def _dcg(hits: list[int]) -> float:
    return float(sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hits)))


def top_n_metrics(
    recommendations: pd.DataFrame,
    test: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, float]:
    if recommendations.empty or test.empty:
        return {
            "precision_at_10": 0.0,
            "recall_at_10": 0.0,
            "f1_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "evaluated_users": 0,
        }

    test_items = {
        user_id: set(group["item_id"])
        for user_id, group in test.groupby("user_id", sort=False)
    }

    precisions = []
    recalls = []
    ndcgs = []

    for user_id, group in recommendations.groupby("user_id", sort=False):
        relevant = test_items.get(user_id)
        if not relevant:
            continue
        ranked_items = group.sort_values("rank")["item_id"].head(top_n).tolist()
        hits = [1 if item_id in relevant else 0 for item_id in ranked_items]
        hit_count = sum(hits)
        precisions.append(hit_count / top_n)
        recalls.append(hit_count / len(relevant))

        ideal_hits = [1] * min(len(relevant), top_n)
        ideal_dcg = _dcg(ideal_hits)
        ndcgs.append(_dcg(hits) / ideal_dcg if ideal_dcg > 0 else 0.0)

    if not precisions:
        return {
            "precision_at_10": 0.0,
            "recall_at_10": 0.0,
            "f1_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "evaluated_users": 0,
        }

    precision = float(np.mean(precisions))
    recall = float(np.mean(recalls))
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "precision_at_10": precision,
        "recall_at_10": recall,
        "f1_at_10": f1,
        "ndcg_at_10": float(np.mean(ndcgs)),
        "evaluated_users": len(precisions),
    }
