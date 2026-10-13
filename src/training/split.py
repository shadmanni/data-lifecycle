"""
Splits data by dataset VERSION (older versions = train, newest version = test).
This avoids the data leakage that a random split would cause.
Memory-optimized with downcasting.
"""
import os
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.memory_opt import optimize_dtypes, MemoryTracker


def time_based_split(df: pd.DataFrame):
    """
    Chronological time-based split:
    Earlier versions = train set; held-out latest versions = test set.
    """
    test_versions = {"linux-6.4", "gcc-12.4.0", "synth_v3"}
    present_versions = set(df["version"].unique())
    matching_test_versions = present_versions.intersection(test_versions)

    if matching_test_versions and len(present_versions) > len(matching_test_versions):
        test_df = df[df["version"].isin(matching_test_versions)].copy()
        train_df = df[~df["version"].isin(matching_test_versions)].copy()
    else:
        unique_versions = sorted(present_versions)
        if len(unique_versions) > 1:
            split_idx = max(1, int(len(unique_versions) * 0.75))
            train_vers = set(unique_versions[:split_idx])
            test_vers = set(unique_versions[split_idx:])
            train_df = df[df["version"].isin(train_vers)].copy()
            test_df = df[df["version"].isin(test_vers)].copy()
        else:
            train_df = df.sample(frac=0.8, random_state=42).copy()
            test_df = df.drop(train_df.index).copy()

    return train_df, test_df


def run_split(
    in_path: str = "data/labels/objects_dataset.parquet",
    train_out: str = "data/labels/train.parquet",
    test_out: str = "data/labels/test.parquet",
):
    with MemoryTracker("Time-Based Train/Test Split"):
        df = pd.read_parquet(in_path)
        train_df, test_df = time_based_split(df)

        train_df = optimize_dtypes(train_df)
        test_df = optimize_dtypes(test_df)

        print(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
        os.makedirs(os.path.dirname(train_out), exist_ok=True)
        os.makedirs(os.path.dirname(test_out), exist_ok=True)
        train_df.to_parquet(train_out, compression="snappy", index=False)
        test_df.to_parquet(test_out, compression="snappy", index=False)
        print(f"Saved train split to {train_out}")
        print(f"Saved test split to {test_out}")
    return train_df, test_df


if __name__ == "__main__":
    run_split()
