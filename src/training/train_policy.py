"""
Memory-Efficient Model Training for Data Lifecycle Optimization.
Supports:
1. LightGBM (fast histogram-based CPU gradient boosting, low memory footprint)
2. XGBoost (histogram tree method)
3. Incremental Batch-wise Learning (SGDClassifier with partial_fit for zero-OOM scaling)

Includes automated assertion checks, memory monitoring, and NaN propagation guards.
"""
import os
import sys
import json
import logging
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.linear_model import SGDClassifier

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.memory_opt import optimize_dtypes, MemoryTracker, validate_batch_integrity, get_memory_info
from src.training.incremental_model import IncrementalPipelineWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_policy")

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
    args = parser.parse_args()

    train(
        model_type=args.model_type,
        train_path=args.train_path,
        test_path=args.test_path,
        models_dir=args.models_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
