import pandas as pd

from config import TOP_N


def popular_candidate_items(train: pd.DataFrame, max_candidates: int | None = None) -> list[str]:
    item_stats = (
        train.groupby("item_id")
        .agg(mean_rating=("rating", "mean"), rating_count=("rating", "size"))
        .reset_index()
        .sort_values(["rating_count", "mean_rating", "item_id"], ascending=[False, False, True])
    )
    if max_candidates is not None:
        item_stats = item_stats.head(max_candidates)
    return item_stats["item_id"].tolist()


def generate_recommendations(
    model,
    train: pd.DataFrame,
    top_n: int = TOP_N,
    max_users: int | None = None,
    max_candidates: int | None = 1000,
) -> pd.DataFrame:
    users = train["user_id"].drop_duplicates().tolist()
    if max_users is not None:
        users = users[:max_users]

    candidates = popular_candidate_items(train, max_candidates=max_candidates)
    all_recommendations = []
    for user_id in users:
        user_recs = model.recommend_for_user(
            user_id=user_id,
            top_n=top_n,
            candidate_items=candidates,
        )
        if not user_recs.empty:
            all_recommendations.append(user_recs)

    if not all_recommendations:
        return pd.DataFrame(
            columns=["user_id", "rank", "item_id", "predicted_rating", "explanation"]
        )
    return pd.concat(all_recommendations, ignore_index=True)


def generate_popularity_recommendations(
    train: pd.DataFrame,
    top_n: int = TOP_N,
    max_users: int | None = None,
    max_candidates: int | None = 1000,
) -> pd.DataFrame:
    users = train["user_id"].drop_duplicates().tolist()
    if max_users is not None:
        users = users[:max_users]

    item_stats = (
        train.groupby("item_id")
        .agg(predicted_rating=("rating", "mean"), rating_count=("rating", "size"))
        .reset_index()
        .sort_values(["rating_count", "predicted_rating", "item_id"], ascending=[False, False, True])
    )
    if max_candidates is not None:
        item_stats = item_stats.head(max_candidates)

    user_seen = {
        user_id: set(group["item_id"])
        for user_id, group in train.groupby("user_id", sort=False)
    }

    rows = []
    for user_id in users:
        seen = user_seen.get(user_id, set())
        user_rows = item_stats[~item_stats["item_id"].isin(seen)].head(top_n)
        for rank, (_, row) in enumerate(user_rows.iterrows(), start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "rank": rank,
                    "item_id": row["item_id"],
                    "predicted_rating": float(row["predicted_rating"]),
                    "explanation": "Popular among many users in the training data.",
                }
            )

    return pd.DataFrame(
        rows,
        columns=["user_id", "rank", "item_id", "predicted_rating", "explanation"],
    )
