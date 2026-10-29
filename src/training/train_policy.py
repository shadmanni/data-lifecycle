"""
Trains machine learning models to predict data lifecycle actions.
Supports memory-efficient training (LightGBM hist CPU, XGBoost hist, Incremental SGDClassifier)
as well as the full 10-model registry benchmarking suite for the Streamlit dashboard.
CPU-only, no GPU needed.
"""
from typing import Any, Dict, List, Optional, Tuple
import os
import sys
import time
import json
import logging
import argparse

# Ensure project root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import SGDClassifier

from src.utils.memory_opt import optimize_dtypes, MemoryTracker, validate_batch_integrity, get_memory_info
from src.training.incremental_model import IncrementalPipelineWrapper
from src.training.models import get_model, DEFAULT_HYPERPARAMS
from src.evaluation.evaluate import compute_metrics, get_feature_importances, FEATURE_COLUMNS
from src.multimodal.fusion import MultimodalPipeline
from src.multimodal.graph_encoder import load_or_create_dependency_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_policy")


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-split train and test datasets."""
    train_path = os.path.join("data", "labels", "train.parquet")
    test_path = os.path.join("data", "labels", "test.parquet")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Parquet files not found in data/labels/. Please run main.py or run_pipeline.py first."
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


def validate_training_data(df: pd.DataFrame, feature_cols: list, target_col: str = "lifecycle_label"):
    """
    Automated assertion checks for training inputs:
    - Verifies no NaNs or Infs
    - Checks boundary constraints
    - Checks target label presence
    """
    logger.info("Executing automated assertion checks on dataset...")
    validate_batch_integrity(df, feature_cols, target_col=target_col)

    # Assert class diversity
    classes = df[target_col].unique()
    assert len(classes) >= 2, f"Training requires at least 2 distinct target classes, found: {classes}"
    logger.info(f"Integrity checks passed. Found {len(classes)} distinct classes: {list(classes)}")


def train_lightgbm(X_train: pd.DataFrame, y_train: np.ndarray, num_classes: int):
    """Memory-efficient LightGBM histogram classifier."""
    import lightgbm as lgb
    logger.info("Initializing LightGBM (hist-based CPU gradient boosting)...")

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        boosting_type="gbdt",
        device="cpu",
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        max_bin=255,          # 8-bit histogram bins for lowest memory usage
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def train_xgboost(X_train: pd.DataFrame, y_train: np.ndarray, num_classes: int):
    """Memory-efficient XGBoost classifier using CPU histogram tree method."""
    import xgboost as xgb
    logger.info("Initializing XGBoost with tree_method='hist'...")

    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        tree_method="hist",   # Fast histogram construction on CPU
        max_bin=256,
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def train_incremental(X_train: pd.DataFrame, y_train: np.ndarray, num_classes: int, batch_size: int = 5000):
    """
    Incremental batch-wise fitting using partial_fit.
    Streams through training data in bounded batches to ensure zero-OOM scaling.
    """
    logger.info(f"Initializing Incremental SGDClassifier (batch size: {batch_size})...")
    classes = np.arange(num_classes)
    model = IncrementalPipelineWrapper(classes=classes)

    n_samples = len(X_train)
    n_batches = int(np.ceil(n_samples / batch_size))

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_samples)
        X_batch = X_train.iloc[start:end]
        y_batch = y_train[start:end]

        model.partial_fit(X_batch, y_batch)
        if (b + 1) % max(1, n_batches // 5) == 0 or b == n_batches - 1:
            mem = get_memory_info()
            logger.info(
                f"  [Batch {b + 1}/{n_batches}] Processed samples {start:,} to {end:,} | "
                f"RSS: {mem['process_rss_mb']:.1f} MB"
            )

    return model


def train(
    model_type: str = "lightgbm",
    train_path: str = "data/labels/train.parquet",
    test_path: str = "data/labels/test.parquet",
    models_dir: str = "models",
    batch_size: int = 5000,
):
    os.makedirs(models_dir, exist_ok=True)

    with MemoryTracker(f"Model Training ({model_type.upper()})"):
        logger.info(f"Loading training data from {train_path}...")
        train_df = optimize_dtypes(pd.read_parquet(train_path))
        test_df = optimize_dtypes(pd.read_parquet(test_path))

        # 1. Assertion and Integrity Checks
        validate_training_data(train_df, FEATURE_COLUMNS, "lifecycle_label")
        validate_training_data(test_df, FEATURE_COLUMNS, "lifecycle_label")

        # 2. Encode Labels
        le = LabelEncoder()
        y_train = le.fit_transform(train_df["lifecycle_label"])
        y_test = le.transform(test_df["lifecycle_label"])
        num_classes = len(le.classes_)

        X_train = train_df[FEATURE_COLUMNS]
        X_test = test_df[FEATURE_COLUMNS]

        logger.info(f"Training dataset size: {len(X_train):,} samples, {len(FEATURE_COLUMNS)} features")
        logger.info(f"Test dataset size:     {len(X_test):,} samples")

        # 3. Model Fitting
        if model_type == "lightgbm":
            clf = train_lightgbm(X_train, y_train, num_classes)
        elif model_type == "xgboost":
            clf = train_xgboost(X_train, y_train, num_classes)
        elif model_type == "incremental":
            clf = train_incremental(X_train, y_train, num_classes, batch_size=batch_size)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose 'lightgbm', 'xgboost', or 'incremental'.")

        # 4. Inference & NaN Propagation Guard
        logger.info("Evaluating on held-out test split...")
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        assert not np.isnan(y_proba).any(), "ERROR: Model predicted NaN probabilities!"
        assert not np.isneginf(y_proba).any() and not np.isposinf(y_proba).any(), "ERROR: Infinite probability values!"

        # 5. Metrics & Reporting
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)

        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
        print(f"Macro F1 Score: {macro_f1:.4f}\n")

        # 6. Save Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(le.classes_)))
        ax.set_yticks(range(len(le.classes_)))
        ax.set_xticklabels(le.classes_, rotation=45)
        ax.set_yticklabels(le.classes_)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(len(le.classes_)):
            for j in range(len(le.classes_)):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        plt.title(f"Lifecycle Policy Confusion Matrix ({model_type.upper()})")
        plt.tight_layout()
        cm_path = os.path.join(models_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close(fig)
        print(f"Saved confusion matrix plot to {cm_path}")

        # 7. Feature Importance Plot
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        else:
            importances = np.ones(len(FEATURE_COLUMNS))

        indices = np.argsort(importances)[::-1]
        sorted_feats = [FEATURE_COLUMNS[i] for i in indices]
        sorted_vals = importances[indices]

        ax2.barh(range(len(sorted_feats)), sorted_vals[::-1], align="center", color="#2b5c8f")
        ax2.set_yticks(range(len(sorted_feats)))
        ax2.set_yticklabels(sorted_feats[::-1])
        ax2.set_xlabel("Importance Score")
        ax2.set_title(f"Feature Importance ({model_type.upper()})")
        plt.tight_layout()
        fi_path = os.path.join(models_dir, "feature_importance.png")
        plt.savefig(fi_path)
        plt.close(fig2)
        print(f"Saved feature importance plot to {fi_path}")

        # 8. Save Artifacts
        model_file = os.path.join(models_dir, "lifecycle_policy_model.pkl")
        encoder_file = os.path.join(models_dir, "label_encoder.pkl")
        joblib.dump(clf, model_file)
        joblib.dump(le, encoder_file)
        print(f"Model saved to {model_file}")
        print(f"Encoder saved to {encoder_file}")

        # 9. Save Metadata JSON
        metrics_summary = {
            "model_type": model_type,
            "macro_f1": macro_f1,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "classes": list(le.classes_),
            "classification_report": report_dict,
        }
        with open(os.path.join(models_dir, "training_metrics.json"), "w") as f:
            json.dump(metrics_summary, f, indent=2)

    return clf, le, macro_f1


def train_multimodal(
    fusion_type: str = "early",
    base_model_name: str = "LightGBM",
    train_path: str = "data/labels/train.parquet",
    test_path: str = "data/labels/test.parquet",
    graph_path: str = "data/processed/dependency_graph.gpickle",
    models_dir: str = "models",
    text_dim: int = 16,
    graph_spectral_dim: int = 4,
    modality_weights: Tuple[float, float, float] = (0.50, 0.30, 0.20),
) -> Tuple[MultimodalPipeline, Dict[str, Any], float]:
    """
    Trains an end-to-end multimodal lifecycle classification pipeline fusing:
    1. Tabular Metadata (11 features)
    2. Textual Semantics (LSA embedding of code comments/headers)
    3. Structural Graph Topology (DAG metrics and spectral embeddings)
    """
    os.makedirs(models_dir, exist_ok=True)

    with MemoryTracker(f"Multimodal Model Training ({fusion_type.upper()} FUSION)"):
        logger.info(f"Loading datasets for multimodal training from {train_path}...")
        train_df = optimize_dtypes(pd.read_parquet(train_path))
        test_df = optimize_dtypes(pd.read_parquet(test_path))

        validate_training_data(train_df, FEATURE_COLUMNS, "lifecycle_label")
        validate_training_data(test_df, FEATURE_COLUMNS, "lifecycle_label")

        # Load dependency DAG
        G = load_or_create_dependency_graph(graph_path=graph_path, df=train_df)

        logger.info(f"Initializing MultimodalPipeline (Fusion: {fusion_type.upper()}, Base: {base_model_name})...")
        pipeline = MultimodalPipeline(
            fusion_type=fusion_type,
            base_model_name=base_model_name,
            text_dim=text_dim,
            graph_spectral_dim=graph_spectral_dim,
            modality_weights=modality_weights,
        )

        start_time = time.time()
        pipeline.fit(train_df=train_df, G=G, label_col="lifecycle_label")
        train_duration = time.time() - start_time
        logger.info(f"Multimodal pipeline training completed in {train_duration:.2f} seconds.")

        # Evaluate on test set
        logger.info("Evaluating multimodal pipeline on held-out test split...")
        y_pred = pipeline.predict(test_df)
        y_test = test_df["lifecycle_label"].to_numpy()

        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        print("\n=== Multimodal Classification Report ===")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}\n")

        # Save artifacts
        pipe_path = os.path.join(models_dir, "multimodal_pipeline.pkl")
        pipeline.save(pipe_path)
        print(f"Saved multimodal pipeline to {pipe_path}")

        metrics_summary = {
            "pipeline_type": "multimodal",
            "fusion_type": fusion_type,
            "base_model_name": base_model_name,
            "training_time_sec": train_duration,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "classes": list(pipeline.label_encoder.classes_),
            "classification_report": report_dict,
        }
        with open(os.path.join(models_dir, "multimodal_metrics.json"), "w") as f:
            json.dump(metrics_summary, f, indent=2)

    return pipeline, metrics_summary, macro_f1


def main():
    parser = argparse.ArgumentParser(description="Memory-Efficient Policy Model Training")
    parser.add_argument(
        "--model_type",
        choices=["lightgbm", "xgboost", "incremental"],
        default="lightgbm",
        help="Algorithm choice: 'lightgbm' (recommended for CPU), 'xgboost' (hist), or 'incremental' (SGDClassifier)",
    )
    parser.add_argument("--batch_size", type=int, default=5000, help="Batch size for incremental learning")
    parser.add_argument("--train_path", type=str, default="data/labels/train.parquet")
    parser.add_argument("--test_path", type=str, default="data/labels/test.parquet")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--multimodal", action="store_true", help="Enable multimodal feature integration")
    parser.add_argument(
        "--fusion_type",
        choices=["early", "late"],
        default="early",
        help="Multimodal fusion strategy: 'early' (feature concatenation) or 'late' (probability stacking)",
    )
    args = parser.parse_args()

    if args.multimodal:
        train_multimodal(
            fusion_type=args.fusion_type,
            base_model_name="LightGBM" if args.model_type == "lightgbm" else "XGBoost",
            train_path=args.train_path,
            test_path=args.test_path,
            models_dir=args.models_dir,
        )
    else:
        train(
            model_type=args.model_type,
            train_path=args.train_path,
            test_path=args.test_path,
            models_dir=args.models_dir,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
