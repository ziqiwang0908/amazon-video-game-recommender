import pandas as pd


class MeanRatingBaseline:
    def __init__(self) -> None:
        self.global_mean = 0.0
        self.user_means: dict[str, float] = {}
        self.item_means: dict[str, float] = {}

    def fit(self, train: pd.DataFrame) -> "MeanRatingBaseline":
        self.global_mean = float(train["rating"].mean())
        self.user_means = train.groupby("user_id")["rating"].mean().to_dict()
        self.item_means = train.groupby("item_id")["rating"].mean().to_dict()
        return self

    def predict(self, user_id: str, item_id: str) -> float:
        if item_id in self.item_means and user_id in self.user_means:
            return float((self.item_means[item_id] + self.user_means[user_id]) / 2)
        if item_id in self.item_means:
            return float(self.item_means[item_id])
        if user_id in self.user_means:
            return float(self.user_means[user_id])
        return float(self.global_mean)

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        predictions = frame[["user_id", "item_id", "rating"]].copy()
        predictions["prediction"] = [
            self.predict(user_id, item_id)
            for user_id, item_id in zip(predictions["user_id"], predictions["item_id"])
        ]
        return predictions
