"""
Splits data by dataset VERSION (older versions = train, newest version = test).
This avoids the data leakage that a random split would cause.
"""
import os
import pandas as pd


def time_based_split(df: pd.DataFrame):
    # linux_v6.1, v6.4 = train ; linux_v6.7 = test (held-out "future" version)
    train_df = df[df["version"].isin(["synth_v1", "synth_v2"])].copy()
    test_df = df[df["version"] == "synth_v3"].copy()
    return train_df, test_df


if __name__ == "__main__":
    in_path = os.path.join("data", "labels", "objects_dataset.parquet")
    df = pd.read_parquet(in_path)
    train_df, test_df = time_based_split(df)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    train_df.to_parquet(os.path.join("data", "labels", "train.parquet"))
    test_df.to_parquet(os.path.join("data", "labels", "test.parquet"))