"""
Cloud Data Lifecycle Policy Optimization & Model Benchmarking Dashboard
Interactive Streamlit application for training, evaluating, and comparing
10 machine learning models for data lifecycle policy prediction.
"""
from typing import Any, Dict, List, Optional
import os
import sys
import time

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

from src.training.models import (
    MODEL_REGISTRY,
    DEFAULT_HYPERPARAMS,
    SCALED_MODELS,
)
from src.training.train_policy import train_single_model, load_dataset
from src.evaluation.evaluate import (
    FEATURE_COLUMNS,
    ACTION_SAVINGS_FACTOR,
    compute_metrics,
    get_feature_importances,
    generate_classification_report_df,
)

# -----------------------------------------------------------------------------
# App Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cloud Data Lifecycle Advisor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "lifecycle_policy_model.pkl")
DEFAULT_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
CM_PLOT_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
FI_PLOT_PATH = os.path.join(MODEL_DIR, "feature_importance.png")

# Currency Conversion: 1 USD = 95 INR
USD_TO_INR_RATE = 95.0

ALL_MODEL_NAMES = list(MODEL_REGISTRY.keys())


# -----------------------------------------------------------------------------
# Data & Cache Initialization
# -----------------------------------------------------------------------------
def get_cached_dataset():
    """Load train and test parquet data with file-modification cache invalidation."""
    train_path = os.path.join("data", "labels", "train.parquet")
    test_path = os.path.join("data", "labels", "test.parquet")
    train_mtime = os.path.getmtime(train_path) if os.path.exists(train_path) else 0.0
    test_mtime = os.path.getmtime(test_path) if os.path.exists(test_path) else 0.0
    return _load_dataset_cached(train_mtime, test_mtime)


@st.cache_data
def _load_dataset_cached(train_mtime: float, test_mtime: float):
    return load_dataset()


def initialize_session_state():
    """Initialize session state storage for trained models, metrics, and classification reports."""
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}
    if "model_metrics" not in st.session_state:
        st.session_state.model_metrics = {}
    if "model_errors" not in st.session_state:
        st.session_state.model_errors = {}
    if "label_encoder" not in st.session_state:
        st.session_state.label_encoder = None
    if "classification_reports" not in st.session_state:
        st.session_state.classification_reports = {}

    # Load existing default XGBoost model from disk if available
    if not st.session_state.trained_models and os.path.exists(DEFAULT_MODEL_PATH) and os.path.exists(DEFAULT_ENCODER_PATH):
        try:
            default_clf = joblib.load(DEFAULT_MODEL_PATH)
            default_le = joblib.load(DEFAULT_ENCODER_PATH)
            st.session_state.label_encoder = default_le
            st.session_state.trained_models["XGBoost"] = default_clf

            # Compute evaluation metrics for existing XGBoost model
            _, test_df = get_cached_dataset()
            metrics = compute_metrics(
                clf=default_clf,
                X_test=test_df[FEATURE_COLUMNS],
                y_test=default_le.transform(test_df["lifecycle_label"]),
                le=default_le,
                training_time_sec=0.760,
                test_df=test_df,
            )
            st.session_state.model_metrics["XGBoost"] = metrics
        except Exception:
            pass


initialize_session_state()


# -----------------------------------------------------------------------------
# Interactive Chart Generators (Plotly Native Widgets with Hover Tooltips)
# -----------------------------------------------------------------------------
def create_f1_vs_accuracy_chart(comp_df: pd.DataFrame) -> go.Figure:
    """Render an interactive grouped bar chart comparing Macro F1 and Accuracy on hover."""
    fig = go.Figure()
    models = comp_df["Model"].tolist()

    fig.add_trace(go.Bar(
        y=models,
        x=comp_df["Macro F1"],
        name="Macro F1",
        orientation="h",
        marker_color="#3b82f6",
        text=[f"{v:.4f}" for v in comp_df["Macro F1"]],
        textposition="outside",
        textfont=dict(color="#F8FAFC", size=10.5),
        hovertemplate="<b>%{y}</b><br>Macro F1: <b>%{x:.4f}</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=models,
        x=comp_df["Accuracy"],
        name="Accuracy",
        orientation="h",
        marker_color="#10b981",
        text=[f"{v:.2%}" for v in comp_df["Accuracy"]],
        textposition="outside",
        textfont=dict(color="#F8FAFC", size=10.5),
        hovertemplate="<b>%{y}</b><br>Accuracy: <b>%{x:.2%}</b><extra></extra>",
    ))

    fig.update_layout(
        font=dict(color="#EEEEEE", family="sans-serif"),
        title=dict(text="Model Performance: Macro F1 vs. Accuracy", font=dict(size=14, color="#F8FAFC")),
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(360, 200 + (len(models) * 30)),
        margin=dict(l=10, r=45, t=55, b=40),
        xaxis=dict(
            title=dict(text="Score", font=dict(size=12, color="#E2E8F0")),
            range=[0, 1.18],
            tickfont=dict(size=11, color="#E2E8F0"),
            gridcolor="rgba(255, 255, 255, 0.12)",
            zerolinecolor="rgba(255, 255, 255, 0.25)",
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=11, color="#E2E8F0"),
            gridcolor="rgba(255, 255, 255, 0.05)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#E2E8F0"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_color="#F8FAFC", bordercolor="#475569"),
    )
    return fig


def create_training_time_chart(comp_df: pd.DataFrame) -> go.Figure:
    """Render an interactive horizontal bar chart displaying model training durations in seconds."""
    sorted_df = comp_df.sort_values("Training Time (s)", ascending=True).reset_index(drop=True)
    times = sorted_df["Training Time (s)"].values
    max_time = max(times) if len(times) > 0 and max(times) > 0 else 1.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_df["Model"],
        x=times,
        orientation="h",
        marker_color="#6366f1",
        text=[f"{v:.3f} s" for v in times],
        textposition="outside",
        textfont=dict(color="#F8FAFC", size=10.5),
        hovertemplate="<b>%{y}</b><br>Training Time: <b>%{x:.3f} seconds</b><extra></extra>",
    ))

    fig.update_layout(
        font=dict(color="#EEEEEE", family="sans-serif"),
        title=dict(text="Training Execution Duration by Model", font=dict(size=14, color="#F8FAFC")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(360, 200 + (len(sorted_df) * 30)),
        margin=dict(l=10, r=45, t=55, b=40),
        xaxis=dict(
            title=dict(text="Duration (seconds)", font=dict(size=12, color="#E2E8F0")),
            range=[0, max_time * 1.28],
            tickfont=dict(size=11, color="#E2E8F0"),
            gridcolor="rgba(255, 255, 255, 0.12)",
            zerolinecolor="rgba(255, 255, 255, 0.25)",
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=11, color="#E2E8F0"),
            gridcolor="rgba(255, 255, 255, 0.05)",
        ),
        hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_color="#F8FAFC", bordercolor="#475569"),
    )
    return fig


def create_feature_importance_chart(fi_series: pd.Series, model_name: str) -> go.Figure:
    """Render an interactive horizontal bar chart for feature importances with hover details."""
    sorted_fi = fi_series.sort_values(ascending=True)
    max_val = sorted_fi.max() if len(sorted_fi) > 0 and sorted_fi.max() > 0 else 1.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_fi.index,
        x=sorted_fi.values,
        orientation="h",
        marker_color="#3b82f6",
        text=[f"{v:.4f}" for v in sorted_fi.values],
        textposition="outside",
        textfont=dict(color="#F8FAFC", size=10.5),
        hovertemplate="Feature: <b>%{y}</b><br>Importance: <b>%{x:.4f}</b> (%{x:.1%})<extra></extra>",
    ))

    fig.update_layout(
        font=dict(color="#EEEEEE", family="sans-serif"),
        title=dict(text=f"Feature Importances ({model_name})", font=dict(size=14, color="#F8FAFC")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=10, r=45, t=55, b=40),
        xaxis=dict(
            title=dict(text="Relative Importance Score", font=dict(size=12, color="#E2E8F0")),
            range=[0, max_val * 1.28],
            tickfont=dict(size=11, color="#E2E8F0"),
            gridcolor="rgba(255, 255, 255, 0.12)",
            zerolinecolor="rgba(255, 255, 255, 0.25)",
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#E2E8F0"),
            gridcolor="rgba(255, 255, 255, 0.05)",
        ),
        hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_color="#F8FAFC", bordercolor="#475569"),
    )
    return fig


def create_confusion_matrix_chart(cm: np.ndarray, classes: List[str], model_name: str) -> go.Figure:
    """Render an interactive confusion matrix heatmap with cell counts and hover inspections."""
    z_text = [[f"{cm[i, j]:,}" for j in range(len(classes))] for i in range(len(classes))]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        text=z_text,
        texttemplate="%{text}",
        textfont=dict(size=11, family="sans-serif"),
        colorscale="Blues",
        colorbar=dict(
            title=dict(text="Count", font=dict(size=11, color="#F8FAFC")),
            tickfont=dict(size=10, color="#E2E8F0"),
            thickness=14,
            len=0.9,
        ),
        hovertemplate="Actual Action: <b>%{y}</b><br>Predicted Action: <b>%{x}</b><br>Count: <b>%{z:,}</b><extra></extra>",
    ))

    fig.update_layout(
        font=dict(color="#EEEEEE", family="sans-serif"),
        title=dict(text=f"Confusion Matrix ({model_name})", font=dict(size=14, color="#F8FAFC")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=30, r=30, t=55, b=40),
        xaxis=dict(
            title=dict(text="Predicted Action", font=dict(size=12, color="#E2E8F0")),
            tickfont=dict(size=11, color="#E2E8F0"),
            side="bottom",
        ),
        yaxis=dict(
            title=dict(text="Actual Action", font=dict(size=12, color="#E2E8F0")),
            autorange="reversed",
            tickfont=dict(size=11, color="#E2E8F0"),
        ),
        hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_color="#F8FAFC", bordercolor="#475569"),
    )
    return fig


def create_probability_distribution_chart(prob_df: pd.DataFrame) -> go.Figure:
    """Render an interactive probability confidence chart with top prediction highlighting."""
    sorted_df = prob_df.sort_values("Probability", ascending=True).reset_index(drop=True)
    probs = sorted_df["Probability"].values
    actions = [str(a).upper() for a in sorted_df["Action"]]
    colors = ["#10b981" if idx == len(sorted_df) - 1 else "#64748b" for idx in range(len(sorted_df))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=actions,
        x=probs,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in probs],
        textposition="outside",
        textfont=dict(color="#F8FAFC", size=11),
        hovertemplate="Action: <b>%{y}</b><br>Confidence: <b>%{x:.2%}</b><extra></extra>",
    ))

    fig.update_layout(
        font=dict(color="#EEEEEE", family="sans-serif"),
        title=dict(text="Action Prediction Probabilities", font=dict(size=13, color="#F8FAFC")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=10, r=45, t=45, b=30),
        xaxis=dict(
            title=dict(text="Probability", font=dict(size=11.5, color="#E2E8F0")),
            range=[0, 1.18],
            tickformat=".0%",
            tickfont=dict(size=11, color="#E2E8F0"),
            gridcolor="rgba(255, 255, 255, 0.12)",
            zerolinecolor="rgba(255, 255, 255, 0.25)",
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#E2E8F0"),
        ),
        hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_color="#F8FAFC", bordercolor="#475569"),
    )
    return fig


# -----------------------------------------------------------------------------
# Sidebar: Model Selection, Hyperparameters, and Execution Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Model Configuration")
    st.markdown("Configure and execute multi-model comparative benchmarking.")

    # Model selector multiselect (Default: XGBoost)
    selected_models = st.multiselect(
        "Select Models to Train & Compare",
        options=ALL_MODEL_NAMES,
        default=["XGBoost"] if "XGBoost" in ALL_MODEL_NAMES else [ALL_MODEL_NAMES[0]],
        help="Select one or more models from the 10 supported estimators.",
    )

    # Quick action selection helpers
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        if st.button("Select All (10)", width="stretch"):
            st.session_state["model_selector"] = ALL_MODEL_NAMES
            st.rerun()
    with col_sel2:
        if st.button("Reset (XGBoost)", width="stretch"):
            st.session_state["model_selector"] = ["XGBoost"]
            st.rerun()

    # Informational note regarding distance/kernel algorithms on larger datasets
    if any(m in selected_models for m in ["Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)"]):
        st.info("**Performance Note**: SVM and KNN perform extensive pairwise distance calculations and may take longer to evaluate.")

    st.markdown("---")
    st.subheader("Hyperparameters")
    st.caption("Expand any selected model to customize key tuning parameters.")

    model_hyperparams: Dict[str, Dict[str, Any]] = {}

    for model_name in selected_models:
        with st.expander(f"{model_name}", expanded=False):
            defaults = DEFAULT_HYPERPARAMS.get(model_name, {})
            hp: Dict[str, Any] = {}

            if model_name == "Logistic Regression":
                hp["C"] = st.slider("Regularization strength (C)", 0.01, 10.0, float(defaults.get("C", 1.0)), step=0.1, key="lr_C")
                hp["max_iter"] = st.slider("Max iterations", 100, 2000, int(defaults.get("max_iter", 1000)), step=100, key="lr_iter")
                hp["solver"] = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"], index=0, key="lr_solver")

            elif model_name == "Decision Tree":
                hp["max_depth"] = st.slider("Max depth", 1, 20, int(defaults.get("max_depth", 6)), key="dt_depth")
                hp["min_samples_split"] = st.slider("Min samples split", 2, 20, int(defaults.get("min_samples_split", 2)), key="dt_split")
                hp["criterion"] = st.selectbox("Criterion", ["gini", "entropy", "log_loss"], index=0, key="dt_crit")

            elif model_name == "Random Forest":
                hp["n_estimators"] = st.slider("Number of estimators", 10, 300, int(defaults.get("n_estimators", 100)), step=10, key="rf_trees")
                hp["max_depth"] = st.slider("Max depth", 1, 20, int(defaults.get("max_depth", 6)), key="rf_depth")
                hp["min_samples_split"] = st.slider("Min samples split", 2, 10, int(defaults.get("min_samples_split", 2)), key="rf_split")

            elif model_name == "Extra Trees":
                hp["n_estimators"] = st.slider("Number of estimators", 10, 300, int(defaults.get("n_estimators", 100)), step=10, key="et_trees")
                hp["max_depth"] = st.slider("Max depth", 1, 20, int(defaults.get("max_depth", 6)), key="et_depth")
                hp["min_samples_split"] = st.slider("Min samples split", 2, 10, int(defaults.get("min_samples_split", 2)), key="et_split")

            elif model_name == "Gradient Boosting":
                hp["n_estimators"] = st.slider("Number of estimators", 10, 300, int(defaults.get("n_estimators", 100)), step=10, key="gb_trees")
                hp["max_depth"] = st.slider("Max depth", 1, 10, int(defaults.get("max_depth", 3)), key="gb_depth")
                hp["learning_rate"] = st.slider("Learning rate", 0.01, 0.5, float(defaults.get("learning_rate", 0.1)), step=0.01, key="gb_lr")

            elif model_name == "XGBoost":
                hp["n_estimators"] = st.slider("Number of estimators", 50, 400, int(defaults.get("n_estimators", 200)), step=25, key="xgb_trees")
                hp["max_depth"] = st.slider("Max depth", 1, 15, int(defaults.get("max_depth", 6)), key="xgb_depth")
                hp["learning_rate"] = st.slider("Learning rate", 0.01, 0.5, float(defaults.get("learning_rate", 0.1)), step=0.01, key="xgb_lr")

            elif model_name == "LightGBM":
                hp["n_estimators"] = st.slider("Number of estimators", 20, 300, int(defaults.get("n_estimators", 100)), step=10, key="lgb_trees")
                hp["max_depth"] = st.slider("Max depth", 1, 15, int(defaults.get("max_depth", 6)), key="lgb_depth")
                hp["learning_rate"] = st.slider("Learning rate", 0.01, 0.5, float(defaults.get("learning_rate", 0.1)), step=0.01, key="lgb_lr")
                hp["num_leaves"] = st.slider("Number of leaves", 10, 100, int(defaults.get("num_leaves", 31)), step=5, key="lgb_leaves")

            elif model_name == "K-Nearest Neighbors (KNN)":
                hp["n_neighbors"] = st.slider("Number of neighbors (k)", 1, 30, int(defaults.get("n_neighbors", 5)), key="knn_k")
                hp["weights"] = st.selectbox("Weight function", ["uniform", "distance"], index=0, key="knn_w")
                hp["metric"] = st.selectbox("Distance metric", ["minkowski", "euclidean", "manhattan"], index=0, key="knn_m")

            elif model_name == "Support Vector Machine (SVM)":
                hp["C"] = st.slider("Regularization (C)", 0.1, 10.0, float(defaults.get("C", 1.0)), step=0.1, key="svm_C")
                hp["kernel"] = st.selectbox("Kernel type", ["rbf", "linear", "poly"], index=0, key="svm_k")

            elif model_name == "MLP Neural Network":
                hidden_choice = st.selectbox(
                    "Hidden layer topology",
                    ["(64, 32)", "(100,)", "(128, 64, 32)", "(32, 16)"],
                    index=0,
                    key="mlp_hidden",
                )
                hidden_map = {
                    "(64, 32)": (64, 32),
                    "(100,)": (100,),
                    "(128, 64, 32)": (128, 64, 32),
                    "(32, 16)": (32, 16),
                }
                hp["hidden_layer_sizes"] = hidden_map[hidden_choice]
                hp["learning_rate_init"] = st.slider("Initial learning rate", 0.0005, 0.01, 0.001, step=0.0005, format="%.4f", key="mlp_lr")
                hp["max_iter"] = st.slider("Max training epochs", 100, 1000, 500, step=50, key="mlp_iter")

            model_hyperparams[model_name] = hp

    st.markdown("---")
    run_benchmark = st.button("Run Selected Models", type="primary", width="stretch")


# -----------------------------------------------------------------------------
# Main Content Area Header
# -----------------------------------------------------------------------------
st.title("Cloud Data Lifecycle Policy Optimization & Model Benchmarking")
st.caption(
    "Machine-learning guided data lifecycle actions across 10 benchmark algorithms "
    "trained on production Linux kernel, GCC, and Azure trace datasets."
)

# Load datasets
try:
    train_df, test_df = get_cached_dataset()
    has_dataset = True
except Exception as e:
    st.error(f"Error loading training datasets: {e}")
    st.info("Run `python main.py` or `python run_pipeline.py` in the terminal to execute the initial data processing pipeline.")
    st.stop()


# -----------------------------------------------------------------------------
# Training Execution Routine
# -----------------------------------------------------------------------------
if run_benchmark:
    if not selected_models:
        st.warning("Please select at least one model to train.")
    else:
        st.session_state.model_errors.clear()
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, model_name in enumerate(selected_models):
            status_text.markdown(f"**Training [{idx+1}/{len(selected_models)}]:** `{model_name}`...")
            try:
                clf, metrics, le = train_single_model(
                    model_name=model_name,
                    hyperparams=model_hyperparams.get(model_name, {}),
                    train_df=train_df,
                    test_df=test_df,
                    le=st.session_state.label_encoder,
                    task_type="classification",
                )
                st.session_state.trained_models[model_name] = clf
                st.session_state.model_metrics[model_name] = metrics
                st.session_state.label_encoder = le

                # If XGBoost is trained, persist to disk to maintain system artifacts
                if model_name == "XGBoost":
                    os.makedirs(MODEL_DIR, exist_ok=True)
                    joblib.dump(clf, DEFAULT_MODEL_PATH)
                    joblib.dump(le, DEFAULT_ENCODER_PATH)

            except Exception as exc:
                st.session_state.model_errors[model_name] = str(exc)

            progress_bar.progress((idx + 1) / len(selected_models))

        status_text.markdown("**Training completed successfully.**")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        st.rerun()


# Display training error banners if any
if st.session_state.model_errors:
    for err_model, err_msg in st.session_state.model_errors.items():
        st.error(f"Failed to train **{err_model}**: {err_msg}")


# Check if at least one model is available
if not st.session_state.trained_models:
    st.warning("No trained models available. Select models from the sidebar and click **Run Selected Models**.")
    st.stop()

le = st.session_state.label_encoder


# -----------------------------------------------------------------------------
# Tabs Layout: Comparison, Diagnostics, Live Simulator
# -----------------------------------------------------------------------------
tab_compare, tab_diagnostics, tab_simulator = st.tabs([
    "Model Comparison & Leaderboard",
    "Per-Model Diagnostics & Explainability",
    "Interactive Policy Inference",
])


# -----------------------------------------------------------------------------
# TAB 1: Model Comparison & Leaderboard
# -----------------------------------------------------------------------------
with tab_compare:
    st.subheader("Model Benchmark Leaderboard")
    st.markdown("Comparative performance metrics across all models evaluated on the held-out chronological test set.")

    # Prepare comparison dataframe
    table_rows = []
    for m_name, m_metrics in st.session_state.model_metrics.items():
        roc_str = f"{m_metrics['roc_auc']:.4f}" if m_metrics.get("roc_auc") is not None else "N/A"

        # Calculate INR currency amounts (1 USD = 95 INR)
        base_usd = m_metrics.get("baseline_cost", 0.0)
        opt_usd = m_metrics.get("optimized_cost", 0.0)
        base_inr = base_usd * USD_TO_INR_RATE
        opt_inr = opt_usd * USD_TO_INR_RATE
        savings_inr = (base_usd - opt_usd) * USD_TO_INR_RATE

        table_rows.append({
            "Model": m_name,
            "Macro F1": m_metrics["macro_f1"],
            "Accuracy": m_metrics["accuracy"],
            "Macro Precision": m_metrics["macro_precision"],
            "Macro Recall": m_metrics["macro_recall"],
            "ROC-AUC": roc_str,
            "Storage Savings (%)": m_metrics["savings_pct"],
            "Monthly Cost (INR)": opt_inr,
            "Monthly Savings (INR)": savings_inr,
            "Training Time (s)": m_metrics["training_time_sec"],
            "Scaling Pipeline": "StandardScaler" if m_name in SCALED_MODELS else "Direct",
        })

    comp_df = pd.DataFrame(table_rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)

    # Top KPI Metrics Cards
    if not comp_df.empty:
        best_f1_row = comp_df.iloc[0]
        fastest_row = comp_df.sort_values("Training Time (s)").iloc[0]
        best_savings_row = comp_df.sort_values("Storage Savings (%)", ascending=False).iloc[0]
        best_acc_row = comp_df.sort_values("Accuracy", ascending=False).iloc[0]

        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
        with kpi_col1:
            st.metric("Top Model (Macro F1)", best_f1_row["Model"], f"{best_f1_row['Macro F1']:.4f}")
        with kpi_col2:
            st.metric("Highest Accuracy", best_acc_row["Model"], f"{best_acc_row['Accuracy']:.2%}")
        with kpi_col3:
            st.metric("Fastest Trainer", fastest_row["Model"], f"{fastest_row['Training Time (s)']:.3f} s")
        with kpi_col4:
            st.metric("Peak Cloud Savings", best_savings_row["Model"], f"{best_savings_row['Storage Savings (%)']:.1f}%")
        with kpi_col5:
            st.metric(
                "Optimized Monthly Cost",
                f"INR {best_savings_row['Monthly Cost (INR)']:,.2f}",
                delta=f"-INR {best_savings_row['Monthly Savings (INR)']:,.2f}",
                delta_color="normal",
            )

    st.markdown("### Metrics Table")
    st.dataframe(
        comp_df.style.format({
            "Macro F1": "{:.4f}",
            "Accuracy": "{:.2%}",
            "Macro Precision": "{:.4f}",
            "Macro Recall": "{:.4f}",
            "Storage Savings (%)": "{:.1f}%",
            "Monthly Cost (INR)": "INR {:,.2f}",
            "Monthly Savings (INR)": "INR {:,.2f}",
            "Training Time (s)": "{:.3f}",
        }).background_gradient(subset=["Macro F1", "Accuracy", "Storage Savings (%)"], cmap="Blues"),
        width="stretch",
    )

    st.markdown("### Visual Performance Comparison")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.plotly_chart(create_f1_vs_accuracy_chart(comp_df), use_container_width=True)

    with chart_col2:
        st.plotly_chart(create_training_time_chart(comp_df), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: Per-Model Diagnostics & Explainability
# -----------------------------------------------------------------------------
with tab_diagnostics:
    st.subheader("Model Diagnostics & Explainability")

    available_models = list(st.session_state.trained_models.keys())
    untrained_models = [m for m in ALL_MODEL_NAMES if m not in available_models]
    if untrained_models:
        with st.expander("Train another model to inspect directly", expanded=False):
            col_add1, col_add2 = st.columns([3, 1])
            with col_add1:
                model_to_quick_train = st.selectbox("Select model to train", options=untrained_models, key="quick_train_sel")
            with col_add2:
                st.write("")
                if st.button(f"Train {model_to_quick_train}", width="stretch", key="quick_train_btn"):
                    with st.spinner(f"Training {model_to_quick_train}..."):
                        clf_qt, metrics_qt, le_qt = train_single_model(
                            model_name=model_to_quick_train,
                            hyperparams=model_hyperparams.get(model_to_quick_train, {}),
                            train_df=train_df,
                            test_df=test_df,
                            le=st.session_state.label_encoder,
                            task_type="classification",
                        )
                        st.session_state.trained_models[model_to_quick_train] = clf_qt
                        st.session_state.model_metrics[model_to_quick_train] = metrics_qt
                        st.session_state.label_encoder = le_qt
                        st.rerun()

    inspect_model = st.selectbox(
        "Select Model to Inspect",
        options=list(st.session_state.trained_models.keys()),
        index=0,
    )

    clf_selected = st.session_state.trained_models[inspect_model]
    metrics_selected = st.session_state.model_metrics[inspect_model]

    # Recompute predictions dynamically on the current test set
    X_test_current = test_df[FEATURE_COLUMNS]
    y_test_current = le.transform(test_df["lifecycle_label"])
    y_pred_current = clf_selected.predict(X_test_current)

    diag_col1, diag_col2 = st.columns(2)

    # 1. Feature Importance
    with diag_col1:
        st.markdown(f"#### Feature Importances ({inspect_model})")
        fi_series = get_feature_importances(
            clf=clf_selected,
            feature_names=FEATURE_COLUMNS,
            X_sample=test_df[FEATURE_COLUMNS],
            y_sample=y_test_current,
        )
        if fi_series is not None and not fi_series.empty:
            st.plotly_chart(create_feature_importance_chart(fi_series, inspect_model), use_container_width=True)
        else:
            st.info("Feature importance is not directly supported or available for this estimator architecture.")

    # 2. Confusion Matrix
    with diag_col2:
        st.markdown(f"#### Confusion Matrix ({inspect_model})")
        cm = confusion_matrix(y_test_current, y_pred_current, labels=range(len(le.classes_)))
        classes = list(le.classes_)
        st.plotly_chart(create_confusion_matrix_chart(cm, classes, inspect_model), use_container_width=True)

    st.markdown("---")
    st.markdown(f"#### Detailed Classification Report ({inspect_model})")

    # Generate classification report from current model's predictions on current test set
    report_df = generate_classification_report_df(
        y_true=y_test_current,
        y_pred=y_pred_current,
        target_names=list(le.classes_),
        labels=list(range(len(le.classes_))),
    )

    st.session_state.classification_reports[inspect_model] = report_df

    # Sanity check: report accuracy matches dashboard metric for the same model
    acc_row = report_df[report_df["Class / Metric"] == "accuracy"]
    report_accuracy = float(acc_row["F1-Score"].iloc[0]) if not acc_row.empty else float(np.mean(y_test_current == y_pred_current))
    cached_accuracy = float(metrics_selected["accuracy"])

    san_col1, san_col2 = st.columns([3, 1])
    with san_col1:
        st.caption(f"Dynamically evaluated on current held-out test set ({len(test_df):,} objects).")
    with san_col2:
        if abs(report_accuracy - cached_accuracy) < 1e-4:
            st.success(f"Sanity Verified: {report_accuracy:.4f}")
        else:
            st.info(f"Report: {report_accuracy:.4f} | Leaderboard: {cached_accuracy:.4f}")

    st.dataframe(
        report_df.style.format({
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
            "Support": "{:,.0f}",
        }),
        width="stretch",
    )


# -----------------------------------------------------------------------------
# TAB 3: Interactive Policy Simulator (Live Inference)
# -----------------------------------------------------------------------------
with tab_simulator:
    st.subheader("Interactive Lifecycle Action Simulator")
    st.caption("Configure cloud object attributes to test real-time policy predictions and model consensus.")

    sim_model_choice = st.selectbox(
        "Model for Prediction",
        options=list(st.session_state.trained_models.keys()),
        index=0,
        key="sim_model_select",
    )

    sim_col1, sim_col2, sim_col3 = st.columns(3)

    with sim_col1:
        st.markdown("#### Redundancy & Size")
        redundancy_score = st.slider("Redundancy score", 0.0, 1.0, 0.35, step=0.01, help="Fraction of chunks shared with other objects")
        object_size = st.number_input("Object size (bytes)", min_value=1, value=65536, step=4096, format="%d", help="Size in bytes (e.g. 65,536 bytes = 64 KB)")
        object_age_days = st.slider("Object age (days)", 0, 1000, 45, step=1)

    with sim_col2:
        st.markdown("#### Access Frequencies")
        access_freq_7d = st.slider("Accesses in last 7 days", 0, 50, 2, step=1)
        access_freq_30d = st.slider("Accesses in last 30 days", 0, 200, 8, step=1)
        access_freq_90d = st.slider("Accesses in last 90 days", 0, 500, 18, step=1)
        days_since_last_access = st.slider("Days since last access", 0, 365, 4, step=1)

    with sim_col3:
        st.markdown("#### Dependencies & Cost")
        num_dependents = st.slider("Number of dependent objects", 0, 50, 0, step=1, help="In-degree in dependency graph")
        compression_ratio = st.slider("Compression ratio", 1.0, 10.0, 2.4, step=0.1)
        compute_time_sec = st.number_input("Estimated compute time (seconds)", min_value=0.0, value=0.0150, format="%.4f", step=0.0010)
        # Currency input in Indian Rupees (INR, ₹), converted to USD internally for ML model
        monthly_storage_cost_inr = st.number_input(
            "Monthly storage cost (INR, ₹)",
            min_value=0.0,
            value=0.14,
            step=0.01,
            format="%.2f",
            help=f"Baseline monthly object storage cost in Indian Rupees (₹). Converted to USD at 1 USD = {USD_TO_INR_RATE:.1f} INR for model inference.",
        )

    st.divider()

    if st.button("Predict Optimal Lifecycle Action", type="primary", width="stretch"):
        # Convert INR input back to USD for model feature input compatibility
        monthly_storage_cost_usd = monthly_storage_cost_inr / USD_TO_INR_RATE

        row = pd.DataFrame([{
            "redundancy_score": redundancy_score,
            "object_size": object_size,
            "object_age_days": object_age_days,
            "access_freq_7d": access_freq_7d,
            "access_freq_30d": access_freq_30d,
            "access_freq_90d": access_freq_90d,
            "days_since_last_access": days_since_last_access,
            "num_dependents": num_dependents,
            "compression_ratio": compression_ratio,
            "compute_time_sec": compute_time_sec,
            "monthly_storage_cost_usd": monthly_storage_cost_usd,
        }])

        active_clf = st.session_state.trained_models[sim_model_choice]
        pred_idx = active_clf.predict(row)[0]
        recommended_action = le.inverse_transform([pred_idx])[0]

        # Savings factor corresponding to recommended action
        savings_factor = ACTION_SAVINGS_FACTOR.get(recommended_action, 0.0)
        projected_monthly_savings_inr = monthly_storage_cost_inr * savings_factor

        st.success(f"### Recommended Action ({sim_model_choice}): **{recommended_action.upper()}**")
        st.info(
            f"Projected Cost Reduction: **{savings_factor * 100:.1f}%** "
            f"(₹{projected_monthly_savings_inr:,.2f}/month saved on this object, baseline ₹{monthly_storage_cost_inr:,.2f})"
        )

        # Probabilities chart and breakdown
        if hasattr(active_clf, "predict_proba"):
            probs = active_clf.predict_proba(row)[0]
            prob_df = pd.DataFrame({
                "Action": le.classes_,
                "Probability": probs,
            }).sort_values("Probability", ascending=False)

            pcol1, pcol2 = st.columns([3, 2])
            with pcol1:
                st.plotly_chart(create_probability_distribution_chart(prob_df), use_container_width=True)
            with pcol2:
                st.markdown("#### Confidence Breakdown")
                st.dataframe(
                    prob_df.style.format({"Probability": "{:.2%}"}),
                    width="stretch",
                )

        # Cross-Model Consensus Comparison
        if len(st.session_state.trained_models) > 1:
            st.markdown("---")
            st.markdown("#### Cross-Model Consensus Comparison")
            consensus_rows = []
            for m_name, m_inst in st.session_state.trained_models.items():
                m_pred = m_inst.predict(row)[0]
                m_label = le.inverse_transform([m_pred])[0]
                m_factor = ACTION_SAVINGS_FACTOR.get(m_label, 0.0)
                m_savings_inr = monthly_storage_cost_inr * m_factor
                consensus_rows.append({
                    "Model": m_name,
                    "Predicted Action": m_label.upper(),
                    "Action Savings Factor": f"{m_factor * 100:.1f}%",
                    "Est. Monthly Savings (INR)": f"₹{m_savings_inr:,.2f}",
                })
            consensus_df = pd.DataFrame(consensus_rows)
            st.dataframe(consensus_df, width="stretch")
