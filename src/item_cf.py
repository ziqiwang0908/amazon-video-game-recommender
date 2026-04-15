from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


@dataclass
class PredictionContext:
    user_id: str
    item_id: str
    prediction: float
    reason: str


class ItemItemCF:
    def __init__(self, n_neighbors: int = 50) -> None:
        self.n_neighbors = n_neighbors
        self.global_mean = 0.0
        self.user_means: dict[str, float] = {}
        self.item_means: dict[str, float] = {}
        self.user_ratings: dict[str, dict[str, float]] = {}
        self.item_neighbors: dict[str, list[tuple[str, float]]] = {}
        self.items: list[str] = []

    def fit(self, train: pd.DataFrame) -> "ItemItemCF":
        train = train[["user_id", "item_id", "rating"]].copy()
        self.global_mean = float(train["rating"].mean())
        self.user_means = train.groupby("user_id")["rating"].mean().to_dict()
        self.item_means = train.groupby("item_id")["rating"].mean().to_dict()
        self.items = sorted(train["item_id"].unique())

        users = sorted(train["user_id"].unique())
        user_index = {user_id: idx for idx, user_id in enumerate(users)}
        item_index = {item_id: idx for idx, item_id in enumerate(self.items)}

        rows = train["item_id"].map(item_index).to_numpy()
        cols = train["user_id"].map(user_index).to_numpy()
        values = train["rating"].astype(float).to_numpy()
        item_user_matrix = csr_matrix((values, (rows, cols)), shape=(len(self.items), len(users)))

        neighbor_count = min(self.n_neighbors + 1, len(self.items))
        model = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine", algorithm="brute")
        model.fit(item_user_matrix)
        distances, indices = model.kneighbors(item_user_matrix, return_distance=True)

        for item_pos, item_id in enumerate(self.items):
            neighbors = []
            for neighbor_pos, distance in zip(indices[item_pos], distances[item_pos]):
                neighbor_id = self.items[int(neighbor_pos)]
                if neighbor_id == item_id:
                    continue
                similarity = max(0.0, 1.0 - float(distance))
                if similarity > 0:
                    neighbors.append((neighbor_id, similarity))
            self.item_neighbors[item_id] = neighbors[: self.n_neighbors]

        self.user_ratings = {
            user_id: dict(zip(group["item_id"], group["rating"]))
            for user_id, group in train.groupby("user_id", sort=False)
        }
        return self

    def _fallback(self, user_id: str, item_id: str) -> PredictionContext:
        if item_id in self.item_means and user_id in self.user_means:
            prediction = (self.item_means[item_id] + self.user_means[user_id]) / 2
            reason = "Fallback average of user and item rating behavior."
        elif item_id in self.item_means:
            prediction = self.item_means[item_id]
            reason = "Fallback item average rating."
        elif user_id in self.user_means:
            prediction = self.user_means[user_id]
            reason = "Fallback user average rating."
        else:
            prediction = self.global_mean
            reason = "Fallback global average rating."
        return PredictionContext(user_id, item_id, float(prediction), reason)

    def predict_with_context(self, user_id: str, item_id: str) -> PredictionContext:
        rated_items = self.user_ratings.get(user_id, {})
        if not rated_items or item_id not in self.item_neighbors:
            return self._fallback(user_id, item_id)

        weighted_sum = 0.0
        weight_total = 0.0
        matched_neighbors = []
        for neighbor_id, similarity in self.item_neighbors[item_id]:
            if neighbor_id not in rated_items:
                continue
            weighted_sum += similarity * rated_items[neighbor_id]
            weight_total += abs(similarity)
            matched_neighbors.append(neighbor_id)

        if weight_total == 0:
            return self._fallback(user_id, item_id)

        prediction = float(np.clip(weighted_sum / weight_total, 1.0, 5.0))
        reason = "Similar to games this user rated before."
        if matched_neighbors:
            reason = f"Similar to {len(matched_neighbors)} game(s) this user rated before."
        return PredictionContext(user_id, item_id, prediction, reason)

    def predict(self, user_id: str, item_id: str) -> float:
        return self.predict_with_context(user_id, item_id).prediction

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        predictions = frame[["user_id", "item_id", "rating"]].copy()
        predictions["prediction"] = [
            self.predict(user_id, item_id)
            for user_id, item_id in zip(predictions["user_id"], predictions["item_id"])
        ]
        return predictions

    def recommend_for_user(
        self,
        user_id: str,
        top_n: int = 10,
        candidate_items: list[str] | None = None,
    ) -> pd.DataFrame:
        rated_items = set(self.user_ratings.get(user_id, {}))
        candidates = candidate_items if candidate_items is not None else self.items
        rows = []
        for item_id in candidates:
            if item_id in rated_items:
                continue
            context = self.predict_with_context(user_id, item_id)
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "predicted_rating": context.prediction,
                    "explanation": context.reason,
                }
            )

        recommendations = pd.DataFrame(rows)
        if recommendations.empty:
            return recommendations
        recommendations = recommendations.sort_values(
            ["predicted_rating", "item_id"], ascending=[False, True]
        ).head(top_n)
        recommendations.insert(1, "rank", range(1, len(recommendations) + 1))
        return recommendations
