"""
Evaluation suite for policy optimization models.
Computes multi-class classification metrics (Accuracy, Macro F1, Precision,
Recall, ROC-AUC), business storage savings, confusion matrices, and feature importances.
"""
from typing import Any, Dict, List, Optional
import os
import sys
# Ensure project root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.inspection import permutation_importance

FEATURE_COLUMNS = [
    "redundancy_score",
    "object_size",
    "object_age_days",
    "access_freq_7d",
    "access_freq_30d",
    "access_freq_90d",
    "days_since_last_access",
    "num_dependents",
    "compression_ratio",
    "compute_time_sec",
    "monthly_storage_cost_usd",
]

ACTION_SAVINGS_FACTOR = {
    "retain": 0.0,
    "compress": 0.4,
    "dedup": 0.7,
    "archive": 0.85,
    "delete": 1.0,
}


USD_TO_INR_RATE = 95.0


def calculate_storage_savings(test_df: pd.DataFrame, predicted_labels: np.ndarray, usd_to_inr: float = USD_TO_INR_RATE) -> Dict[str, float]:
    """
    Calculate cloud storage cost reduction resulting from predicted actions.

    Args:
        test_df: Test dataframe containing 'monthly_storage_cost_usd'.
        predicted_labels: String array or series of predicted policy actions.
        usd_to_inr: Conversion multiplier from USD to Indian Rupees (INR).

    Returns:
        Dictionary with baseline cost, optimized cost, savings percentage, and INR equivalents.
    """
    baseline_cost = float(test_df["monthly_storage_cost_usd"].sum())
    savings_factor = pd.Series(predicted_labels, index=test_df.index).map(ACTION_SAVINGS_FACTOR).fillna(0.0)
    cost_after_action = test_df["monthly_storage_cost_usd"] * (1.0 - savings_factor)
    optimized_cost = float(cost_after_action.sum())
    
    savings_pct = ((baseline_cost - optimized_cost) / baseline_cost * 100.0) if baseline_cost > 0 else 0.0
    baseline_cost_inr = baseline_cost * usd_to_inr
    optimized_cost_inr = optimized_cost * usd_to_inr
    savings_inr = (baseline_cost - optimized_cost) * usd_to_inr

    return {
        "baseline_cost": baseline_cost,
        "optimized_cost": optimized_cost,
        "savings_pct": savings_pct,
        "baseline_cost_inr": baseline_cost_inr,
        "optimized_cost_inr": optimized_cost_inr,
        "savings_inr": savings_inr,
    }


def compute_metrics(
    clf: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    le: LabelEncoder,
    training_time_sec: float = 0.0,
    test_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics and simulated savings for an estimator.

    Args:
        clf: Trained estimator or Pipeline.
        X_test: Features test matrix.
        y_test: Encoded ground-truth labels.
        le: Fitted LabelEncoder instance.
        training_time_sec: Duration in seconds taken to train the estimator.
        test_df: Optional full test dataframe to compute domain cost savings.

    Returns:
        Dictionary containing metric values, reports, predictions, and confusion matrix.
    """
    y_pred = clf.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    # ROC-AUC calculation (handling multi-class and potential missing classes in y_test)
    roc_auc: Optional[float] = None
    y_proba: Optional[np.ndarray] = None
    if hasattr(clf, "predict_proba"):
        try:
            y_proba = clf.predict_proba(X_test)
            present_classes = np.unique(y_test)
            if len(present_classes) > 1:
                # Binarize over all classes in LabelEncoder
                y_bin = label_binarize(y_test, classes=range(len(le.classes_)))
                # Only score classes present in ground truth to avoid sklearn ValueError
                if y_bin.shape[1] == y_proba.shape[1]:
                    roc_auc = float(
                        roc_auc_score(
                            y_bin[:, present_classes],
                            y_proba[:, present_classes],
                            average="macro",
                            multi_class="ovr",
                        )
                    )
        except Exception:
            roc_auc = None

    # Storage savings computation
    savings_data: Dict[str, float] = {}
    if test_df is not None and "monthly_storage_cost_usd" in test_df.columns:
        savings_data = calculate_storage_savings(test_df, y_pred_labels)

    # Detailed classification report dataframe & dictionary
    class_report_df = generate_classification_report_df(
        y_true=y_test,
        y_pred=y_pred,
        target_names=list(le.classes_),
        labels=list(range(len(le.classes_))),
    )
    class_report_dict = classification_report(
        y_test,
        y_pred,
        labels=range(len(le.classes_)),
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "roc_auc": roc_auc,
        "training_time_sec": training_time_sec,
        "savings_pct": savings_data.get("savings_pct", 0.0),
        "optimized_cost": savings_data.get("optimized_cost", 0.0),
        "baseline_cost": savings_data.get("baseline_cost", 0.0),
        "optimized_cost_inr": savings_data.get("optimized_cost_inr", 0.0),
        "baseline_cost_inr": savings_data.get("baseline_cost_inr", 0.0),
        "savings_inr": savings_data.get("savings_inr", 0.0),
        "y_pred": y_pred,
        "y_pred_labels": y_pred_labels,
        "y_proba": y_proba,
        "class_report": class_report_dict,
        "class_report_df": class_report_df,
        "confusion_matrix": cm,
    }


def generate_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    labels: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Generate a formatted classification report DataFrame containing per-class metrics
    as well as summary rows ('accuracy', 'macro avg', 'weighted avg') with real class names.

    Args:
        y_true: Ground truth target values.
        y_pred: Estimated target values from estimator.
        target_names: Optional list of display class names matching labels.
        labels: Optional list of label indices.

    Returns:
        DataFrame containing formatted columns ['Class / Metric', 'Precision', 'Recall', 'F1-Score', 'Support'].
    """
    if labels is None and target_names is not None:
        labels = list(range(len(target_names)))

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    rows: List[Dict[str, Any]] = []

    # 1. Per-class rows
    if target_names is not None:
        for name in target_names:
            if name in report_dict:
                item = report_dict[name]
                rows.append({
                    "Class / Metric": name,
                    "Precision": float(item["precision"]),
                    "Recall": float(item["recall"]),
                    "F1-Score": float(item["f1-score"]),
                    "Support": int(item["support"]),
                })
    else:
        for k, v in report_dict.items():
            if isinstance(v, dict) and k not in ("macro avg", "weighted avg"):
                rows.append({
                    "Class / Metric": str(k),
                    "Precision": float(v["precision"]),
                    "Recall": float(v["recall"]),
                    "F1-Score": float(v["f1-score"]),
                    "Support": int(v["support"]),
                })

    total_support = sum(r["Support"] for r in rows)

    # 2. Accuracy row
    if "accuracy" in report_dict:
        acc_val = float(report_dict["accuracy"])
        rows.append({
            "Class / Metric": "accuracy",
            "Precision": acc_val,
            "Recall": acc_val,
            "F1-Score": acc_val,
            "Support": total_support,
        })

    # 3. Macro avg & Weighted avg rows
    for avg_key in ["macro avg", "weighted avg"]:
        if avg_key in report_dict:
            item = report_dict[avg_key]
            rows.append({
                "Class / Metric": avg_key,
                "Precision": float(item["precision"]),
                "Recall": float(item["recall"]),
                "F1-Score": float(item["f1-score"]),
                "Support": int(item["support"]),
            })

    return pd.DataFrame(rows)


def get_feature_importances(
    clf: Any,
    feature_names: List[str],
    X_sample: Optional[pd.DataFrame] = None,
    y_sample: Optional[np.ndarray] = None,
) -> Optional[pd.Series]:
    """
    Extract or approximate feature importance scores across different model families.
    Uses tree feature_importances_, linear model absolute coefficients, or permutation importance.

    Args:
        clf: Trained estimator or Pipeline.
        feature_names: List of feature names.
        X_sample: Optional feature samples for permutation importance fallback.
        y_sample: Optional label samples for permutation importance fallback.

    Returns:
        Pandas Series indexed by feature names, or None if unavailable.
    """
    estimator = clf
    if hasattr(clf, "named_steps") and "model" in clf.named_steps:
        estimator = clf.named_steps["model"]

    # 1. Direct feature_importances_ (Trees, Forests, Gradient Boosting, XGBoost, LightGBM)
    if hasattr(estimator, "feature_importances_"):
        raw_importances = np.array(estimator.feature_importances_, dtype=float)
        total = np.sum(raw_importances)
        normalized = raw_importances / total if total > 0 else raw_importances
        return pd.Series(normalized, index=feature_names).sort_values(ascending=False)

    # 2. Linear model coefficients (Logistic Regression)
    if hasattr(estimator, "coef_"):
        coef = np.array(estimator.coef_, dtype=float)
        # Average absolute magnitude across classes for multi-class
        if coef.ndim > 1:
            raw_importances = np.mean(np.abs(coef), axis=0)
        else:
            raw_importances = np.abs(coef)
        total = np.sum(raw_importances)
        normalized = raw_importances / total if total > 0 else raw_importances
        return pd.Series(normalized, index=feature_names).sort_values(ascending=False)

    # 3. Permutation importance for SVM, KNN, MLP
    if X_sample is not None and y_sample is not None and len(X_sample) > 0:
        try:
            # Use a sample of up to 200 rows for swift responsiveness
            sample_size = min(200, len(X_sample))
            X_sub = X_sample.iloc[:sample_size]
            y_sub = y_sample[:sample_size]
            result = permutation_importance(
                clf,
                X_sub,
                y_sub,
                n_repeats=3,
                random_state=42,
                scoring="f1_macro",
            )
            raw = np.maximum(0, result.importances_mean)
            total = np.sum(raw)
            normalized = raw / total if total > 0 else raw
            return pd.Series(normalized, index=feature_names).sort_values(ascending=False)
        except Exception:
            return None

    return None


def evaluate() -> None:
    """
    Simulates applying the trained default policy across the test set and reports
    storage savings vs. a 'retain everything' baseline. Retains backward compatibility.
    """
    model_path = os.path.join("models", "lifecycle_policy_model.pkl")
    encoder_path = os.path.join("models", "label_encoder.pkl")
    test_path = os.path.join("data", "labels", "test.parquet")

    if not os.path.exists(model_path) or not os.path.exists(encoder_path) or not os.path.exists(test_path):
        print("Required artifacts not found for evaluation.")
        return

    clf = joblib.load(model_path)
    le = joblib.load(encoder_path)
    test_df = pd.read_parquet(test_path)

    X_test = test_df[FEATURE_COLUMNS]
    preds = clf.predict(X_test)
    predicted_labels = le.inverse_transform(preds)
    test_df = test_df.copy()
    test_df["predicted_action"] = predicted_labels

    savings = calculate_storage_savings(test_df, predicted_labels)

    print(f"Baseline (retain everything) monthly cost: INR {savings['baseline_cost_inr']:,.2f} (${savings['baseline_cost']:.4f} USD)")
    print(f"Optimized (your policy) monthly cost:       INR {savings['optimized_cost_inr']:,.2f} (${savings['optimized_cost']:.4f} USD)")
    print(f"Estimated savings:                            {savings['savings_pct']:.1f}% (INR {savings['savings_inr']:,.2f}/month saved)")

    print("\nAction distribution recommended by the model:")
    print(test_df["predicted_action"].value_counts())


if __name__ == "__main__":
    evaluate()