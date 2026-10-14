"""
Trains machine learning models to predict data lifecycle actions.
Supports training the default XGBoost baseline as well as any of the 10 supported models.
CPU-only, no GPU needed.
"""
from typing import Any, Dict, List, Optional, Tuple
import os
import sys
import time

# Ensure project root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.training.models import get_model, DEFAULT_HYPERPARAMS
from src.evaluation.evaluate import compute_metrics, get_feature_importances, FEATURE_COLUMNS


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-split train and test datasets."""
    train_path = os.path.join("data", "labels", "train.parquet")
    test_path = os.path.join("data", "labels", "test.parquet")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Parquet files not found in data/labels/. Please run main.py first."
        )
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    return train_df, test_df


def train_single_model(
    model_name: str,
    hyperparams: Optional[Dict[str, Any]] = None,
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    le: Optional[LabelEncoder] = None,
    task_type: str = "classification",
) -> Tuple[Any, Dict[str, Any], LabelEncoder]:
    """
    Train and evaluate a single model from the registry.

    Args:
        model_name: Name of model to train.
        hyperparams: Dictionary of hyperparameter overrides.
        train_df: Optional train dataframe.
        test_df: Optional test dataframe.
        le: Optional fitted LabelEncoder.
        task_type: 'classification' or 'regression'.

    Returns:
        Tuple of (trained_estimator, metrics_dictionary, label_encoder).
    """
    if train_df is None or test_df is None:
        train_df, test_df = load_dataset()

    if le is None:
        le = LabelEncoder()
        # Fit on all unique classes across train and test to avoid missing class indexing
        all_labels = pd.concat([train_df["lifecycle_label"], test_df["lifecycle_label"]]).unique()
        le.fit(all_labels)

    y_train = le.transform(train_df["lifecycle_label"])
    y_test = le.transform(test_df["lifecycle_label"])

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    params = hyperparams or {}
    estimator = get_model(model_name, task_type=task_type, **params)

    # Time the training duration
    start_time = time.time()
    estimator.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Compute comprehensive evaluation metrics
    metrics = compute_metrics(
        clf=estimator,
        X_test=X_test,
        y_test=y_test,
        le=le,
        training_time_sec=training_time,
        test_df=test_df,
    )

    return estimator, metrics, le


def train() -> None:
    """
    Original baseline training entry point. Fits XGBoost classifier and exports
    artifacts to models/ directory, preserving full backward compatibility.
    """
    train_df, test_df = load_dataset()

    le = LabelEncoder()
    # Fit encoder preserving exact original behavior
    y_train = le.fit_transform(train_df["lifecycle_label"])
    y_test = le.transform(test_df["lifecycle_label"])

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    # Use XGBoost factory with exact existing hyperparameters
    clf = get_model("XGBoost", task_type="classification", n_estimators=200, max_depth=6, learning_rate=0.1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("=== Classification report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    print("Macro F1:", f1_score(y_test, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=45)
    ax.set_yticklabels(le.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.title("Lifecycle Policy Confusion Matrix")
    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig(os.path.join("models", "confusion_matrix.png"))
    plt.close(fig)
    print("Saved confusion matrix plot to models/confusion_matrix.png")

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    import xgboost as xgb
    xgb.plot_importance(clf, ax=ax2)
    plt.tight_layout()
    plt.savefig(os.path.join("models", "feature_importance.png"))
    plt.close(fig2)
    print("Saved feature importance plot to models/feature_importance.png")

    joblib.dump(clf, os.path.join("models", "lifecycle_policy_model.pkl"))
    joblib.dump(le, os.path.join("models", "label_encoder.pkl"))
    print("Model saved to models/lifecycle_policy_model.pkl")


if __name__ == "__main__":
    train()
