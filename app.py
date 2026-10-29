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
from src.training.train_policy import train_single_model, load_dataset, train_multimodal
from src.evaluation.evaluate import (
    FEATURE_COLUMNS,
    ACTION_SAVINGS_FACTOR,
    compute_metrics,
    get_feature_importances,
    generate_classification_report_df,
)
from src.multimodal.fusion import MultimodalPipeline
from src.multimodal.diagnostics import (
    project_embeddings,
    create_embedding_scatter_plot,
    analyze_modality_importance,
    create_modality_contribution_chart,
)
from src.multimodal.text_encoder import extract_text_semantics

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
MULTIMODAL_MODEL_PATH = os.path.join(MODEL_DIR, "multimodal_pipeline.pkl")
MULTIMODAL_METRICS_PATH = os.path.join(MODEL_DIR, "multimodal_metrics.json")
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
    if "multimodal_pipeline" not in st.session_state:
        st.session_state.multimodal_pipeline = None

    # Load existing multimodal pipeline if available
    if st.session_state.multimodal_pipeline is None and os.path.exists(MULTIMODAL_MODEL_PATH):
        try:
            st.session_state.multimodal_pipeline = joblib.load(MULTIMODAL_MODEL_PATH)
        except Exception:
            pass

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
tab_compare, tab_diagnostics, tab_simulator, tab_multimodal = st.tabs([
    "Model Comparison & Leaderboard",
    "Per-Model Diagnostics & Explainability",
    "Interactive Policy Inference",
    "Multimodal Diagnostics & Fusion",
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


# -----------------------------------------------------------------------------
# TAB 4: Multimodal Diagnostics, Embedding Projection & Live Fusion Simulator
# -----------------------------------------------------------------------------
with tab_multimodal:
    st.subheader("Multimodal Feature Integration & Diagnostics")
    st.caption(
        "Fuses Tabular Metadata (11 features), Textual Semantics (16-D LSA comments/docstrings/headers), "
        "and Structural Graph Topology (DAG PageRank, degree ratios & spectral embeddings)."
    )

    # Architectural Overview Banner
    arch_col1, arch_col2, arch_col3 = st.columns(3)
    with arch_col1:
        st.markdown(
            """
            <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 14px;">
                <h4 style="margin: 0 0 6px 0; color: #60a5fa;">📊 Modality 1: Tabular</h4>
                <p style="font-size: 0.85rem; margin: 0; color: #cbd5e1;">
                    11 engineered features: size, redundancy score, access frequencies (7d/30d/90d),
                    recency, age, dependents count, and baseline hot storage cost.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with arch_col2:
        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 14px;">
                <h4 style="margin: 0 0 6px 0; color: #34d399;">📝 Modality 2: Text Semantics</h4>
                <p style="font-size: 0.85rem; margin: 0; color: #cbd5e1;">
                    Parses code comments, docstrings, include headers, and README text via TF-IDF
                    and Latent Semantic Analysis (TruncatedSVD, 16 dense dimensions).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with arch_col3:
        st.markdown(
            """
            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 14px;">
                <h4 style="margin: 0 0 6px 0; color: #fbbf24;">🕸️ Modality 3: Graph Topology</h4>
                <p style="font-size: 0.85rem; margin: 0; color: #cbd5e1;">
                    Dependency DAG metrics: PageRank, in/out degrees, clustering coefficient,
                    coreness number, and 4-D low-rank spectral network embeddings.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Multimodal Pipeline Training / Configuration Expander
    mm_pipe: Optional[MultimodalPipeline] = st.session_state.multimodal_pipeline

    with st.expander("Configure & Train Multimodal Pipeline", expanded=(mm_pipe is None)):
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
        with cfg_col1:
            fusion_strategy_choice = st.selectbox(
                "Fusion Strategy",
                ["Early Fusion (Feature Concatenation)", "Late Fusion (Probability Stacking & Blending)"],
                index=0,
                help="Early: Concatenates all modalities into a unified vector. Late: Trains modality experts and stacks probabilities.",
            )
            fusion_type_param = "early" if "Early" in fusion_strategy_choice else "late"

        with cfg_col2:
            base_mm_model = st.selectbox(
                "Base Learner / Classifier",
                ["LightGBM", "XGBoost", "Random Forest"],
                index=0,
                help="Classifier architecture used for fused feature training or tabular expert.",
            )

        with cfg_col3:
            text_dim_slider = st.slider("Text LSA Embedding Dimensions", 8, 32, 16, step=4)
            graph_spec_slider = st.slider("Graph Spectral Dimensions", 2, 8, 4, step=2)

        if st.button("Train Multimodal Pipeline", type="primary", width="stretch"):
            with st.spinner("Extracting text semantics, graph topology, and training multimodal pipeline..."):
                try:
                    new_mm_pipe, mm_metrics, mm_f1 = train_multimodal(
                        fusion_type=fusion_type_param,
                        base_model_name=base_mm_model,
                        train_path="data/labels/train.parquet",
                        test_path="data/labels/test.parquet",
                        models_dir=MODEL_DIR,
                        text_dim=text_dim_slider,
                        graph_spectral_dim=graph_spec_slider,
                    )
                    st.session_state.multimodal_pipeline = new_mm_pipe
                    st.success(f"Multimodal Pipeline trained successfully! Macro F1: {mm_f1:.4f}")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Training failed: {ex}")

    if mm_pipe is None:
        st.info("No multimodal pipeline loaded. Click **Train Multimodal Pipeline** above to initialize.")
        st.stop()

    # Active Pipeline KPI Metrics
    st.markdown("### Active Multimodal Pipeline Status")
    kpi_m1, kpi_m2, kpi_m3, kpi_m4, kpi_m5 = st.columns(5)
    with kpi_m1:
        st.metric("Fusion Mode", f"{mm_pipe.fusion_type.upper()} FUSION", mm_pipe.base_model_name)
    with kpi_m2:
        st.metric("Total Fused Dimensions", f"{len(mm_pipe.feature_names_fused)} Dims", f"11 Tab + {mm_pipe.text_dim} Text + {len(mm_pipe.graph_encoder.feature_names)} Graph")
    with kpi_m3:
        st.metric("Vocabulary Features", f"{mm_pipe.text_encoder.max_features}", f"{mm_pipe.text_dim} LSA Components")
    with kpi_m4:
        st.metric("Graph Spectral Dim", f"{mm_pipe.graph_spectral_dim} Dims", "DAG Projected")
    with kpi_m5:
        st.metric("Lifecycle Actions", f"{len(mm_pipe.label_encoder.classes_)} Classes", "Full Coverage")

    st.markdown("---")

    # Modality Contribution Analysis
    st.markdown("### Modality Contribution & Explainability")
    mod_col1, mod_col2 = st.columns([3, 2])

    with mod_col1:
        modality_importance = analyze_modality_importance(mm_pipe)
        st.plotly_chart(create_modality_contribution_chart(modality_importance), use_container_width=True)

    with mod_col2:
        st.markdown("#### Text Semantics Keywords")
        st.caption("Top influential vocabulary terms learned by the LSA text encoder.")
        top_terms_dict = mm_pipe.text_encoder.get_top_terms_per_component(n_top_terms=4)
        if top_terms_dict:
            term_rows = []
            for comp_name, terms_list in list(top_terms_dict.items())[:6]:
                terms_str = ", ".join([f"`{t[0]}`" for t in terms_list])
                term_rows.append({"Component": comp_name, "Top Semantic Keywords": terms_str})
            st.dataframe(pd.DataFrame(term_rows), width="stretch", hide_index=True)
        else:
            st.info("Keyword mapping available once text encoder is fitted.")

    st.markdown("---")

    # Embedding Projection Section (PCA & t-SNE)
    st.markdown("### 🌐 Multimodal Embedding Space Visualizer")
    st.caption("Project 35+ dimensional multimodal embeddings into 2D or 3D interactive feature space colored by lifecycle policy action.")

    proj_ctrl1, proj_ctrl2, proj_ctrl3 = st.columns(3)
    with proj_ctrl1:
        proj_method = st.selectbox("Projection Method", ["PCA", "t-SNE"], index=0, key="proj_method_select")
    with proj_ctrl2:
        proj_dims = st.radio("Dimensions", [2, 3], index=0, horizontal=True, key="proj_dims_radio")
    with proj_ctrl3:
        sample_pts = st.slider("Subsample Count", 200, 1500, 800, step=100, help="Number of test points to project for sub-second responsiveness.")

    with st.spinner(f"Computing {proj_method} {proj_dims}D projection across multimodal feature space..."):
        try:
            # Transform test dataset into multimodal embeddings
            X_tab_t, X_text_t, X_graph_t = mm_pipe.transform_modalities(test_df)
            X_fused_test = np.hstack([X_tab_t, X_text_t, X_graph_t]).astype(np.float32)

            coords, var_ratio, sample_idxs = project_embeddings(
                X_fused_test,
                method=proj_method,
                n_components=proj_dims,
                sample_size=sample_pts,
            )

            sampled_labels = test_df["lifecycle_label"].iloc[sample_idxs].tolist()
            sampled_df = test_df.iloc[sample_idxs].reset_index(drop=True)

            scatter_fig = create_embedding_scatter_plot(
                coords=coords,
                labels=sampled_labels,
                hover_df=sampled_df,
                method_name=proj_method,
                n_components=proj_dims,
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

            if var_ratio is not None:
                st.caption(
                    f"**PCA Explained Variance**: Component 1: {var_ratio[0]:.1%} | Component 2: {var_ratio[1]:.1%}"
                    + (f" | Component 3: {var_ratio[2]:.1%}" if len(var_ratio) > 2 else "")
                    + f" (Total Explained: {sum(var_ratio):.1%})"
                )
        except Exception as e_proj:
            st.error(f"Error computing projection: {e_proj}")

    st.markdown("---")

    # Live Multimodal Simulator Section
    st.markdown("### 🧪 Live Multimodal Policy Inference & Decision Simulator")
    st.caption("Simulate how changing code comments, graph topology, or tabular metrics impacts the lifecycle recommendation.")

    # Preset Scenarios
    preset_scenario = st.selectbox(
        "Choose a Realistic Scenario Preset (or configure manually)",
        [
            "Custom (Manual Input)",
            "1. Deprecated utility module slated for removal (Candidate: DELETE)",
            "2. Highly redundant shared library with few dependencies (Candidate: DEDUP)",
            "3. Critical kernel scheduler lock module in hot execution path (Candidate: RETAIN)",
            "4. Cold backup logs & configuration older than 6 months (Candidate: ARCHIVE)",
            "5. Highly compressible data table with steady monthly reads (Candidate: COMPRESS)",
        ],
        index=0,
        key="mm_scenario_preset",
    )

    # Preset values map
    if "1. Deprecated" in preset_scenario:
        p_red = 0.90
        p_deps = 0
        p_age = 420
        p_f7 = 0
        p_f30 = 0
        p_f90 = 0
        p_days = 120
        p_text = "// Deprecated legacy cache utility module slated for removal in v2.4\n/* Obsolete routine - no active callers */\n#include <stdio.h>"
        p_in_deg = 0
        p_out_deg = 1
        p_ratio = 3.2
    elif "2. Highly redundant" in preset_scenario:
        p_red = 0.78
        p_deps = 1
        p_age = 150
        p_f7 = 1
        p_f30 = 5
        p_f90 = 12
        p_days = 15
        p_text = "// Shared duplicate helper routines for data serialization\n#include <stdlib.h>\n#include <string.h>"
        p_in_deg = 1
        p_out_deg = 2
        p_ratio = 2.1
    elif "3. Critical kernel" in preset_scenario:
        p_red = 0.05
        p_deps = 28
        p_age = 300
        p_f7 = 18
        p_f30 = 95
        p_f90 = 340
        p_days = 0
        p_text = "/* Core kernel scheduler spinlock synchronization primitives */\n#include <linux/spinlock.h>\n#include <linux/sched.h>\nvoid acquire_kernel_mutex();"
        p_in_deg = 28
        p_out_deg = 4
        p_ratio = 1.6
    elif "4. Cold backup" in preset_scenario:
        p_red = 0.20
        p_deps = 0
        p_age = 280
        p_f7 = 0
        p_f30 = 0
        p_f90 = 1
        p_days = 110
        p_text = "# System execution audit logs and historical archive snapshot\n# Generated automatically on system shutdown\n# DO NOT MODIFY MANUALLY"
        p_in_deg = 0
        p_out_deg = 0
        p_ratio = 2.8
    elif "5. Highly compressible" in preset_scenario:
        p_red = 0.15
        p_deps = 2
        p_age = 90
        p_f7 = 1
        p_f30 = 12
        p_f90 = 45
        p_days = 4
        p_text = "// Analytical reporting dataset payload buffer with repetitive textual records\n#include <zlib.h>\nstruct report_payload_t buffer;"
        p_in_deg = 2
        p_out_deg = 1
        p_ratio = 4.2
    else:
        p_red = 0.35
        p_deps = 1
        p_age = 60
        p_f7 = 2
        p_f30 = 8
        p_f90 = 24
        p_days = 5
        p_text = "// Application helper routines\n#include <stdio.h>\nvoid process_payload();"
        p_in_deg = 1
        p_out_deg = 2
        p_ratio = 2.4

    sim_m_col1, sim_m_col2, sim_m_col3 = st.columns(3)

    with sim_m_col1:
        st.markdown("#### 📊 Modality 1: Tabular Metadata")
        sim_red = st.slider("Redundancy score", 0.0, 1.0, float(p_red), step=0.01, key="mm_sim_red")
        sim_size = st.number_input("Object size (bytes)", min_value=1, value=65536, step=4096, key="mm_sim_size")
        sim_age = st.slider("Object age (days)", 0, 1000, int(p_age), step=1, key="mm_sim_age")
        sim_f7 = st.slider("Accesses in last 7 days", 0, 50, int(p_f7), step=1, key="mm_sim_f7")
        sim_f30 = st.slider("Accesses in last 30 days", 0, 200, int(p_f30), step=1, key="mm_sim_f30")
        sim_f90 = st.slider("Accesses in last 90 days", 0, 500, int(p_f90), step=1, key="mm_sim_f90")
        sim_days_last = st.slider("Days since last access", 0, 365, int(p_days), step=1, key="mm_sim_days_last")
        sim_cost_inr = st.number_input("Monthly Storage Cost (INR, ₹)", min_value=0.0, value=0.15, step=0.02, format="%.2f", key="mm_sim_cost_inr")

    with sim_m_col2:
        st.markdown("#### 📝 Modality 2: Text Semantics")
        sim_text = st.text_area(
            "Source Code Comments, Docstrings, or Headers",
            value=p_text,
            height=210,
            key="mm_sim_text",
            help="Type or paste C/C++/Python comments, function signatures, or docstrings.",
        )
        sim_rel_path = st.text_input("Simulated Relative Path", value="kernel/sched/core.c", key="mm_sim_path")

    with sim_m_col3:
        st.markdown("#### 🕸️ Modality 3: Graph Topology")
        sim_in_deg = st.slider("In-Degree (Dependent modules)", 0, 50, int(p_in_deg), step=1, key="mm_sim_in_deg")
        sim_out_deg = st.slider("Out-Degree (Included headers)", 0, 30, int(p_out_deg), step=1, key="mm_sim_out_deg")
        sim_pr = st.slider("PageRank Centrality proxy", 0.0001, 0.1000, float(max(0.001, p_in_deg * 0.003)), step=0.0010, format="%.4f", key="mm_sim_pr")
        sim_clust = st.slider("Clustering coefficient", 0.0, 1.0, 0.15, step=0.05, key="mm_sim_clust")
        sim_ratio = st.slider("Compression ratio", 1.0, 10.0, float(p_ratio), step=0.1, key="mm_sim_ratio")

    st.markdown("---")

    if st.button("Predict Multimodal Lifecycle Action", type="primary", width="stretch", key="mm_predict_btn"):
        # Convert INR cost to USD for model feature scaling
        sim_cost_usd = sim_cost_inr / USD_TO_INR_RATE

        tab_input_dict = {
            "redundancy_score": sim_red,
            "object_size": sim_size,
            "object_age_days": sim_age,
            "access_freq_7d": sim_f7,
            "access_freq_30d": sim_f30,
            "access_freq_90d": sim_f90,
            "days_since_last_access": sim_days_last,
            "num_dependents": sim_in_deg,
            "compression_ratio": sim_ratio,
            "compute_time_sec": float(sim_size / (50.0 * 1024 * 1024)),
            "monthly_storage_cost_usd": sim_cost_usd,
        }

        graph_input_dict = {
            "in_degree": sim_in_deg,
            "out_degree": sim_out_deg,
            "pagerank": sim_pr,
            "clustering": sim_clust,
            "core_number": 1.0,
        }

        with st.spinner("Executing multimodal forward inference across all 3 encoders..."):
            sim_res = mm_pipe.predict_single(
                tabular_dict=tab_input_dict,
                source_code_or_comment=sim_text,
                graph_dict=graph_input_dict,
                relative_path=sim_rel_path,
            )

        recommended_action = sim_res["action"]
        savings_factor = sim_res["savings_factor"]
        projected_savings_inr = sim_cost_inr * savings_factor

        st.success(
            f"### Recommended Multimodal Action: **{recommended_action.upper()}** "
            f"(Confidence: {sim_res['confidence']:.2%})"
        )
        st.info(
            f"💰 Projected Cost Reduction: **{savings_factor * 100:.1f}%** | "
            f"Monthly Savings: **₹{projected_savings_inr:,.2f}** (Baseline ₹{sim_cost_inr:,.2f} / month)"
        )

        res_col1, res_col2 = st.columns([3, 2])
        with res_col1:
            prob_df = pd.DataFrame([
                {"Action": k.upper(), "Probability": v}
                for k, v in sim_res["probabilities"].items()
            ]).sort_values("Probability", ascending=False)
            st.plotly_chart(create_probability_distribution_chart(prob_df), use_container_width=True)

        with res_col2:
            st.markdown("#### Probability Confidence Breakdown")
            st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}), width="stretch")

        # Modality Decision Consensus Breakdown
        st.markdown("---")
        st.markdown("#### 🤝 Modality Decision Consensus & Voting Breakdown")
        st.caption("How each distinct modality expert individually voted on this object:")

        mod_breakdown = sim_res.get("modality_breakdown", {})
        consensus_items = []

        mod_titles = {
            "tabular": "📊 Tabular Metadata Expert",
            "text": "📝 Textual Semantics Expert",
            "graph": "🕸️ Graph Topology Expert",
        }

        for m_key, m_info in mod_breakdown.items():
            consensus_items.append({
                "Modality Expert": mod_titles.get(m_key, m_key.title()),
                "Voted Action": m_info["action"].upper(),
                "Confidence": f"{m_info['confidence']:.2%}",
                "Agreement with Final": "✅ Yes" if m_info["action"].lower() == recommended_action.lower() else "⚠️ Disagreed",
            })

        # Append Early vs Late comparison
        consensus_items.append({
            "Modality Expert": "⚡ Early Fusion (Concatenation)",
            "Voted Action": sim_res["early_action"].upper(),
            "Confidence": f"{sim_res['early_confidence']:.2%}",
            "Agreement with Final": "✅ Yes" if sim_res["early_action"].lower() == recommended_action.lower() else "⚠️ Disagreed",
        })
        consensus_items.append({
            "Modality Expert": "🥞 Late Fusion (Probability Stacking)",
            "Voted Action": sim_res["late_action"].upper(),
            "Confidence": f"{sim_res['late_confidence']:.2%}",
            "Agreement with Final": "✅ Yes" if sim_res["late_action"].lower() == recommended_action.lower() else "⚠️ Disagreed",
        })

        st.dataframe(pd.DataFrame(consensus_items), width="stretch", hide_index=True)

