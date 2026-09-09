"""
Simulates applying the trained policy across the test set and reports
storage savings vs. a "retain everything" baseline.
"""
import os
import joblib
import pandas as pd

FEATURE_COLUMNS = [
    "redundancy_score", "object_size", "object_age_days",
    "access_freq_7d", "access_freq_30d", "access_freq_90d",
    "days_since_last_access", "num_dependents",
    "compression_ratio", "compute_time_sec", "monthly_storage_cost_usd",
]

ACTION_SAVINGS_FACTOR = {
    "retain": 0.0,
    "compress": 0.4,
    "dedup": 0.7,
    "archive": 0.85,
    "delete": 1.0,
}


def evaluate():
    clf = joblib.load(os.path.join("models", "lifecycle_policy_model.pkl"))
    le = joblib.load(os.path.join("models", "label_encoder.pkl"))
    test_df = pd.read_parquet(os.path.join("data", "labels", "test.parquet"))

    X_test = test_df[FEATURE_COLUMNS]
    preds = clf.predict(X_test)
    predicted_labels = le.inverse_transform(preds)
    test_df = test_df.copy()
    test_df["predicted_action"] = predicted_labels

    baseline_cost = test_df["monthly_storage_cost_usd"].sum()

    test_df["savings_factor"] = test_df["predicted_action"].map(ACTION_SAVINGS_FACTOR)
    test_df["cost_after_action"] = test_df["monthly_storage_cost_usd"] * (1 - test_df["savings_factor"])
    optimized_cost = test_df["cost_after_action"].sum()

    print(f"Baseline (retain everything) monthly cost: ${baseline_cost:.4f}")
    print(f"Optimized (your policy) monthly cost:       ${optimized_cost:.4f}")
    print(f"Estimated savings:                            {(1 - optimized_cost/baseline_cost)*100:.1f}%")

    print("\nAction distribution recommended by the model:")
    print(test_df["predicted_action"].value_counts())


if __name__ == "__main__":
    evaluate()