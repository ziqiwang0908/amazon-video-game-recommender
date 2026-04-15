import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("MDM_DATA_ROOT", PROJECT_ROOT / "data")).expanduser()

RAW_DATA_DIR = DATA_ROOT / "raw"
INTERIM_DATA_DIR = DATA_ROOT / "interim"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"

RESULTS_DIR = DATA_ROOT / "results"
DEMO_DATA_DIR = PROJECT_ROOT / "demo" / "video-game-recommender" / "public" / "data"

REVIEWS_FILE = RAW_DATA_DIR / "Video_Games_5.json.gz"
METADATA_FILE = RAW_DATA_DIR / "meta_Video_Games.json.gz"

INTERACTIONS_FILE = PROCESSED_DATA_DIR / "interactions.csv"
ITEMS_FILE = PROCESSED_DATA_DIR / "items.csv"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

METRICS_FILE = RESULTS_DIR / "metrics.json"
RATING_PREDICTIONS_FILE = RESULTS_DIR / "rating_predictions.csv"
TOP10_RECOMMENDATIONS_FILE = RESULTS_DIR / "top10_recommendations.csv"
POPULARITY_RECOMMENDATIONS_FILE = RESULTS_DIR / "popularity_top10_recommendations.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.2
TOP_N = 10
NEIGHBORHOOD_SIZE = 50


def ensure_directories() -> None:
    for path in [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        RESULTS_DIR,
        DEMO_DATA_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
