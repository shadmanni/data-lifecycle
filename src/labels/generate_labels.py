"""
Generates lifecycle action labels using documented, explainable rules.
Ground truth rule-based policy engine.
Memory-optimized with categorical downcasting and batch assertion checks.
"""
import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.memory_opt import optimize_dtypes, MemoryTracker, validate_batch_integrity


def assign_label(row) -> str:
<<<<<<< HEAD
    # 1. Deduplication: High chunk sharing across engine telemetry runs
    if row.get("redundancy_score", 0.0) > 0.5 and row.get("num_dependents", 0) < 3:
=======
    """Row-level label assignment function."""
    if row["access_freq_7d"] > 3:
        return "retain"

    if (row["redundancy_score"] > 0.85
            and row["num_dependents"] == 0
            and row["object_age_days"] > 365
            and row["access_freq_90d"] == 0):
        return "delete"

    if row["redundancy_score"] > 0.6 and row["num_dependents"] < 3:
>>>>>>> 804f91863ad72e711450ea597561d1d202cc30dd
        return "dedup"

    # 2. Deletion: Inactive, retired engine telemetry with no dependents and aged history
    if (row.get("num_dependents", 0) == 0
            and row.get("days_since_last_access", 0) > 80
            and row.get("object_age_days", 0) > 250
            and row.get("access_freq_90d", 0) < 5):
        return "delete"

    # 3. Archival: Infrequently accessed historical telemetry
    if row.get("object_age_days", 0) > 120 and row.get("days_since_last_access", 0) > 40 and row.get("access_freq_7d", 0) <= 5:
        return "archive"

    # 4. Compression: Telemetry with strong compression ratio and moderate access
    if row.get("compression_ratio", 1.0) > 2.0 and row.get("redundancy_score", 0.0) < 0.4:
        return "compress"

    # 5. Retention: Active, critical telemetry with recent accesses or dependent downstream models
    if row.get("access_freq_7d", 0) > 10 or row.get("num_dependents", 0) >= 2:
        return "retain"

    if row.get("compression_ratio", 1.0) > 1.5:
        return "compress"

    return "retain"


def generate_labels(
    in_path: str = "data/processed/objects_with_cost.parquet",
    out_path: str = "data/labels/objects_dataset.parquet",
):
    with MemoryTracker("Generate Lifecycle Labels"):
        df = pd.read_parquet(in_path)

        # High performance vectorized labeling
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
<<<<<<< HEAD
    in_path = os.path.join("data", "processed", "objects_with_cost.parquet")
    df = pd.read_parquet(in_path)

    df["lifecycle_label"] = df.apply(assign_label, axis=1)

    print(df["lifecycle_label"].value_counts())

    out_path = os.path.join("data", "labels", "objects_dataset.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)
    print(f"Final labeled dataset saved to {out_path}")
=======
    generate_labels()
>>>>>>> 804f91863ad72e711450ea597561d1d202cc30dd
