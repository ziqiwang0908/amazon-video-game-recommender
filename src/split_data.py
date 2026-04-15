import argparse

import numpy as np
import pandas as pd

from config import INTERACTIONS_FILE, RANDOM_SEED, TEST_FILE, TEST_SIZE, TRAIN_FILE, ensure_directories


def split_per_user(
    interactions: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    train_parts = []
    test_parts = []

    for _, group in interactions.groupby("user_id", sort=False):
        group = group.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
        if len(group) < 2:
            train_parts.append(group)
            continue
        test_count = max(1, int(round(len(group) * test_size)))
        test_count = min(test_count, len(group) - 1)
        test_parts.append(group.iloc[:test_count])
        train_parts.append(group.iloc[test_count:])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=interactions.columns)
    return train, test


def create_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_directories()
    interactions = pd.read_csv(INTERACTIONS_FILE)
    train, test = split_per_user(interactions)
    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    train, test = create_split()
    print(f"Saved {len(train):,} training rows to {TRAIN_FILE}")
    print(f"Saved {len(test):,} testing rows to {TEST_FILE}")


if __name__ == "__main__":
    main()
