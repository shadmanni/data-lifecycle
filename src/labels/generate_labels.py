"""
Generates lifecycle action labels using documented, explainable rules.
Ground truth rule-based policy engine.
Memory-optimized with categorical downcasting and batch assertion checks.
"""
import os
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.memory_opt import optimize_dtypes, MemoryTracker, validate_batch_integrity


def assign_label(row) -> str:
    if row["access_freq_7d"] > 3:
        return "retain"

    if (row["redundancy_score"] > 0.85
            and row["num_dependents"] == 0
            and row["object_age_days"] > 365
            and row["access_freq_90d"] == 0):
        return "delete"

    if row["redundancy_score"] > 0.6 and row["num_dependents"] < 3:
        return "dedup"

    if row["object_age_days"] > 180 and row["days_since_last_access"] > 90:
        return "archive"

    if row["redundancy_score"] < 0.3 and row["access_freq_30d"] > 0 and row["compression_ratio"] > 1.5:
        return "compress"

    return "retain"


def generate_labels(
    in_path: str = "data/processed/objects_with_cost.parquet",
    out_path: str = "data/labels/objects_dataset.parquet",
):
    with MemoryTracker("Generate Lifecycle Labels"):
        import numpy as np
        df = pd.read_parquet(in_path)

        cond_retain = df["access_freq_7d"] > 3
        cond_delete = (df["redundancy_score"] > 0.85) & (df["num_dependents"] == 0) & (df["object_age_days"] > 365) & (df["access_freq_90d"] == 0)
        cond_dedup = (df["redundancy_score"] > 0.6) & (df["num_dependents"] < 3)
        cond_archive = (df["object_age_days"] > 180) & (df["days_since_last_access"] > 90)
        cond_compress = (df["redundancy_score"] < 0.3) & (df["access_freq_30d"] > 0) & (df["compression_ratio"] > 1.5)

        conditions = [cond_retain, cond_delete, cond_dedup, cond_archive, cond_compress]
        choices = ["retain", "delete", "dedup", "archive", "compress"]
        df["lifecycle_label"] = np.select(conditions, choices, default="retain")

        feature_cols = [
            "redundancy_score", "object_size", "object_age_days",
            "access_freq_7d", "access_freq_30d", "access_freq_90d",
            "days_since_last_access", "num_dependents",
            "compression_ratio", "compute_time_sec", "monthly_storage_cost_usd",
        ]

        # Validate data integrity before saving
        validate_batch_integrity(df, feature_cols, target_col="lifecycle_label")

        # Downcast dtypes
        df = optimize_dtypes(df, verbose=True)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, compression="snappy", index=False)

        print("Lifecycle label distribution:")
        print(df["lifecycle_label"].value_counts())
        print(f"Final labeled dataset saved to {out_path}")
    return df


if __name__ == "__main__":
    generate_labels()