"""
Splits data by dataset VERSION (older versions = train, newest version = test).
This avoids the data leakage that a random split would cause.
"""
import os
import pandas as pd


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


if __name__ == "__main__":
    in_path = os.path.join("data", "labels", "objects_dataset.parquet")
    df = pd.read_parquet(in_path)
    train_df, test_df = time_based_split(df)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    
    out_dir = os.path.join("data", "labels")
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(out_dir, "train.parquet"))
    test_df.to_parquet(os.path.join(out_dir, "test.parquet"))