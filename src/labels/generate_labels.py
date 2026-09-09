"""
Generates lifecycle action labels using documented, explainable rules.
This is your "ground truth" since no public dataset has real lifecycle labels.
Document every threshold in your paper's methodology section.
"""
import os
import pandas as pd


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


if __name__ == "__main__":
    in_path = os.path.join("data", "processed", "objects_with_cost.parquet")
    df = pd.read_parquet(in_path)

    df["lifecycle_label"] = df.apply(assign_label, axis=1)

    print(df["lifecycle_label"].value_counts())

    out_path = os.path.join("data", "labels", "objects_dataset.parquet")
    df.to_parquet(out_path)
    print(f"Final labeled dataset saved to {out_path}")
    