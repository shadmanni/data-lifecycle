"""
Generates lifecycle action labels using documented, explainable rules.
This is your "ground truth" since no public dataset has real lifecycle labels.
Document every threshold in your paper's methodology section.
"""
import os
import pandas as pd


def assign_label(row) -> str:
    # 1. Deduplication: High chunk sharing across engine telemetry runs
    if row.get("redundancy_score", 0.0) > 0.5 and row.get("num_dependents", 0) < 3:
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


if __name__ == "__main__":
    in_path = os.path.join("data", "processed", "objects_with_cost.parquet")
    df = pd.read_parquet(in_path)

    df["lifecycle_label"] = df.apply(assign_label, axis=1)

    print(df["lifecycle_label"].value_counts())

    out_path = os.path.join("data", "labels", "objects_dataset.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)
    print(f"Final labeled dataset saved to {out_path}")