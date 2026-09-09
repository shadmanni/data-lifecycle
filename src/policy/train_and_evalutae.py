"""
src/policy/train_and_evaluate.py
-----------------------------------
Takes data/processed/final_features.csv (real redundancy + real access
overlay + synthetic dependents/cost/criticality), assigns lifecycle
labels using a documented rule function, trains a Random Forest
classifier, and evaluates it against baselines -- same methodology as
the earlier all-synthetic prototype, now running on real file data.

Run:
    python src/policy/train_and_evaluate.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib

FEATURES_PATH = Path("data/processed/final_features.csv")
LABELS_OUT_PATH = Path("data/labels/labeled_objects.csv")
RESULTS_PATH = Path("models/results.json")
MODEL_PATH = Path("models/rf_model.joblib")

ACTIONS = ["RETAIN", "DEDUPLICATE", "COMPRESS", "ARCHIVE", "DELETE"]
FEATURE_COLS = [
    "age_days", "size_mb", "access_count_30d", "days_since_last_access",
    "redundancy_score", "exact_duplicate", "num_dependents",
    "owner_criticality", "predicted_future_access",
    "storage_cost_per_month", "compute_cost_to_process", "version_count",
]


def assign_label(row, rng):
    """Documented rule-based ground-truth generator (Step 2.6 of the roadmap)."""
    protected = (row["owner_criticality"] > 0.75) or (row["num_dependents"] >= 3)
    noise = rng.normal(0, 0.05)

    if row["exact_duplicate"] == 1 and not protected:
        return "DEDUPLICATE"
    if row["redundancy_score"] > 0.7 and not protected:
        return "DEDUPLICATE"

    cold = (row["days_since_last_access"] > 15 and
            row["predicted_future_access"] < 0.22 + noise)
    if cold and not protected:
        if row["owner_criticality"] < 0.25 and row["age_days"] > 200 and row["num_dependents"] == 0:
            return "DELETE"
        return "ARCHIVE"

    warm_large = (row["access_count_30d"] <= 3 and row["size_mb"] > 0.05 and
                  row["redundancy_score"] < 0.6 and row["days_since_last_access"] > 5)
    if warm_large:
        return "COMPRESS"

    return "RETAIN"


def false_deletion_rate(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_delete = y_true != "DELETE"
    if non_delete.sum() == 0:
        return 0.0
    return ((y_pred == "DELETE") & non_delete).sum() / non_delete.sum()


def false_retention_rate(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    delete_mask = y_true == "DELETE"
    if delete_mask.sum() == 0:
        return 0.0
    return ((y_pred != "DELETE") & delete_mask).sum() / delete_mask.sum()


def baseline_age_threshold(df):
    preds = []
    for _, row in df.iterrows():
        if row["days_since_last_access"] > 45:
            preds.append("DELETE")
        elif row["days_since_last_access"] > 20:
            preds.append("ARCHIVE")
        else:
            preds.append("RETAIN")
    return np.array(preds)


def main():
    if not FEATURES_PATH.exists():
        print(f"[ERROR] {FEATURES_PATH} not found -- run build_features.py first.")
        return

    df = pd.read_csv(FEATURES_PATH)
    n = len(df)
    print(f"Loaded {n} objects from {FEATURES_PATH}")

    rng = np.random.default_rng(42)
    df["action"] = df.apply(lambda row: assign_label(row, rng), axis=1)

    LABELS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LABELS_OUT_PATH, index=False)

    print("\nClass distribution:")
    print(df["action"].value_counts())

    if n < 30 or df["action"].nunique() < 2:
        print("\n[NOTE] Very small / low-diversity dataset -- this is expected if you "
              "ran this on a small sample. Increase MAX_FILES_PER_VERSION in "
              "build_object_table.py and re-run the pipeline for a real train/test split.")

    X = df[FEATURE_COLS]
    y = df["action"]

    test_size = 0.25 if n >= 20 else max(1, n // 4) / n
    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=14, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    results = {
        "RandomForest": {
            "accuracy": accuracy_score(y_test, rf_pred),
            "macro_f1": f1_score(y_test, rf_pred, average="macro", zero_division=0),
            "false_deletion_rate": false_deletion_rate(y_test, rf_pred),
            "false_retention_rate": false_retention_rate(y_test, rf_pred),
        }
    }

    age_pred = baseline_age_threshold(df.loc[X_test.index])
    results["Baseline_AgeThreshold"] = {
        "accuracy": accuracy_score(y_test, age_pred),
        "macro_f1": f1_score(y_test, age_pred, average="macro", zero_division=0),
        "false_deletion_rate": false_deletion_rate(y_test, age_pred),
        "false_retention_rate": false_retention_rate(y_test, age_pred),
    }

    print("\n" + "=" * 90)
    print(f"{'Model':<25}{'Accuracy':>10}{'MacroF1':>10}{'FalseDelRate':>15}{'FalseRetRate':>15}")
    print("=" * 90)
    for name, r in results.items():
        print(f"{name:<25}{r['accuracy']:>10.4f}{r['macro_f1']:>10.4f}"
              f"{r['false_deletion_rate']:>15.4f}{r['false_retention_rate']:>15.4f}")

    print("\nClassification report (RandomForest):")
    print(classification_report(y_test, rf_pred, labels=ACTIONS, zero_division=0))

    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_test, rf_pred, labels=ACTIONS)
    print(pd.DataFrame(cm, index=[f"true_{a}" for a in ACTIONS], columns=[f"pred_{a}" for a in ACTIONS]))

    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances.to_string())

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "results": results,
            "feature_importances": importances.to_dict(),
            "n_total": n, "n_train": len(X_train), "n_test": len(X_test),
            "class_distribution": df["action"].value_counts().to_dict(),
        }, f, indent=2, default=float)

    print(f"\nSaved model to {MODEL_PATH} and results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()