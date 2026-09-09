"""
Trains an XGBoost multiclass classifier to predict the lifecycle action.
CPU-only (tree_method='hist'), no GPU needed.
"""
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb

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


def train():
    train_df = pd.read_parquet(os.path.join("data", "labels", "train.parquet"))
    test_df = pd.read_parquet(os.path.join("data", "labels", "test.parquet"))

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["lifecycle_label"])
    y_test = le.transform(test_df["lifecycle_label"])

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        tree_method="hist",   # CPU-friendly, fast
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("=== Classification report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Macro F1:", f1_score(y_test, y_pred, average="macro"))

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
    print("Saved confusion matrix plot to models/confusion_matrix.png")

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    xgb.plot_importance(clf, ax=ax2)
    plt.tight_layout()
    plt.savefig(os.path.join("models", "feature_importance.png"))
    print("Saved feature importance plot to models/feature_importance.png")

    joblib.dump(clf, os.path.join("models", "lifecycle_policy_model.pkl"))
    joblib.dump(le, os.path.join("models", "label_encoder.pkl"))
    print("Model saved to models/lifecycle_policy_model.pkl")


if __name__ == "__main__":
    train()
