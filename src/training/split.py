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
    # synth_v1, synth_v2 = train ; synth_v3 = test (held-out "future" version)
    train_df = df[df["version"].isin(["synth_v1", "synth_v2"])].copy()
    test_df = df[df["version"] == "synth_v3"].copy()

    # If dataset has custom versions or single version, fallback to 80/20 time/index split
    if len(train_df) == 0 or len(test_df) == 0:
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

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
        train_df.to_parquet(train_out, compression="snappy", index=False)
        test_df.to_parquet(test_out, compression="snappy", index=False)
        print(f"Saved train split to {train_out}")
        print(f"Saved test split to {test_out}")
    return train_df, test_df


if __name__ == "__main__":
    run_split()