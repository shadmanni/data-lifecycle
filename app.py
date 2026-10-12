"""
Cloud Data Lifecycle Policy Optimization & Model Benchmarking Dashboard
Interactive Streamlit application for training, evaluating, and comparing
10 machine learning models for data lifecycle policy prediction.
"""
from typing import Any, Dict, List, Optional
import os
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

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
)

# -----------------------------------------------------------------------------
# App Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cloud Data Lifecycle Advisor",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "lifecycle_policy_model.pkl")
DEFAULT_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
CM_PLOT_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
FI_PLOT_PATH = os.path.join(MODEL_DIR, "feature_importance.png")

ALL_MODEL_NAMES = list(MODEL_REGISTRY.keys())


# -----------------------------------------------------------------------------
# Data & Cache Initialization
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_dataset():
    """Load train and test parquet data with Streamlit data caching."""
    return load_dataset()


def initialize_session_state():
    """Initialize session state storage for trained models and metrics."""
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}
    if "model_metrics" not in st.session_state:
        st.session_state.model_metrics = {}
    if "model_errors" not in st.session_state:
        st.session_state.model_errors = {}
    if "label_encoder" not in st.session_state:
        st.session_state.label_encoder = None

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
                training_time_sec=0.76,
                test_df=test_df,
            )
            st.session_state.model_metrics["XGBoost"] = metrics
        except Exception:
            pass


initialize_session_state()

# -----------------------------------------------------------------------------
# Sidebar: Model Selection, Hyperparameters, and Execution Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Model Configuration")
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

    # Informational warning regarding distance/kernel algorithms on larger datasets
    if any(m in selected_models for m in ["Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)"]):
        st.info("ℹ️ **Performance Note**: SVM and KNN perform extensive pairwise distance calculations and may take slightly longer to evaluate.")

    st.markdown("---")
    st.subheader("🛠️ Hyperparameters")
    st.caption("Expand any selected model to customize 2-4 key tuning parameters.")

    model_hyperparams: Dict[str, Dict[str, Any]] = {}

    for model_name in selected_models:
        with st.expander(f"⚙️ {model_name}", expanded=False):
            defaults = DEFAULT_HYPERPARAMS.get(model_name, {})
            hp: Dict[str, Any] = {}

            if model_name == "Logistic Regression":
                hp["C"] = st.slider("Regularization strength (C)", 0.01, 10.0, float(defaults.get("C", 1.0)), step=0.1, key=f"lr_C")
                hp["max_iter"] = st.slider("Max iterations", 100, 2000, int(defaults.get("max_iter", 1000)), step=100, key=f"lr_iter")
                hp["solver"] = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"], index=0, key=f"lr_solver")

            elif model_name == "Decision Tree":
                hp["max_depth"] = st.slider("Max depth", 1, 20, int(defaults.get("max_depth", 6)), key=f"dt_depth")
                hp["min_samples_split"] = st.slider("Min samples split", 2, 20, int(defaults.get("min_samples_split", 2)), key=f"dt_split")
                hp["criterion"] = st.selectbox("Criterion", ["gini", "entropy", "log_loss"], index=0, key=f"dt_crit")

            elif model_name == "Random Forest":
                hp["n_estimators"] = st.slider("Number of estimators", 10, 300, int(defaults.get("n_estimators", 100)), step=10, key=f"rf_trees")
                hp["max_depth"] = st.slider("Max depth", 1, 20, int(defaults.get("max_depth", 6)), key=f"rf_depth")
                hp["min_samples_split"] = st.slider("Min samples split", 2, 10, int(defaults.get("min_samples_split", 2)), key=f"rf_split")

            elif model_name == "Extra Trees":
                hp["n_estimators"] = st.slider("Number of estimators", 10, 300, int(defaults.get("n_estimators", 100)), step=10, key=f"et_trees")
                hp["max_depth"] = st.slider("Max depth", 1, 20, int(defaults.get("max_depth", 6)), key=f"et_depth")
                hp["min_samples_split"] = st.slider("Min samples split", 2, 10, int(defaults.get("min_samples_split", 2)), key=f"et_split")

            elif model_name == "Gradient Boosting":
                hp["n_estimators"] = st.slider("Number of estimators", 10, 300, int(defaults.get("n_estimators", 100)), step=10, key=f"gb_trees")
                hp["max_depth"] = st.slider("Max depth", 1, 10, int(defaults.get("max_depth", 3)), key=f"gb_depth")
                hp["learning_rate"] = st.slider("Learning rate", 0.01, 0.5, float(defaults.get("learning_rate", 0.1)), step=0.01, key=f"gb_lr")

            elif model_name == "XGBoost":
                hp["n_estimators"] = st.slider("Number of estimators", 50, 400, int(defaults.get("n_estimators", 200)), step=25, key=f"xgb_trees")
                hp["max_depth"] = st.slider("Max depth", 1, 15, int(defaults.get("max_depth", 6)), key=f"xgb_depth")
                hp["learning_rate"] = st.slider("Learning rate", 0.01, 0.5, float(defaults.get("learning_rate", 0.1)), step=0.01, key=f"xgb_lr")

            elif model_name == "LightGBM":
                hp["n_estimators"] = st.slider("Number of estimators", 20, 300, int(defaults.get("n_estimators", 100)), step=10, key=f"lgb_trees")
                hp["max_depth"] = st.slider("Max depth", 1, 15, int(defaults.get("max_depth", 6)), key=f"lgb_depth")
                hp["learning_rate"] = st.slider("Learning rate", 0.01, 0.5, float(defaults.get("learning_rate", 0.1)), step=0.01, key=f"lgb_lr")
                hp["num_leaves"] = st.slider("Number of leaves", 10, 100, int(defaults.get("num_leaves", 31)), step=5, key=f"lgb_leaves")

            elif model_name == "K-Nearest Neighbors (KNN)":
                hp["n_neighbors"] = st.slider("Number of neighbors (k)", 1, 30, int(defaults.get("n_neighbors", 5)), key=f"knn_k")
                hp["weights"] = st.selectbox("Weight function", ["uniform", "distance"], index=0, key=f"knn_w")
                hp["metric"] = st.selectbox("Distance metric", ["minkowski", "euclidean", "manhattan"], index=0, key=f"knn_m")

            elif model_name == "Support Vector Machine (SVM)":
                hp["C"] = st.slider("Regularization (C)", 0.1, 10.0, float(defaults.get("C", 1.0)), step=0.1, key=f"svm_C")
                hp["kernel"] = st.selectbox("Kernel type", ["rbf", "linear", "poly"], index=0, key=f"svm_k")

            elif model_name == "MLP Neural Network":
                hidden_choice = st.selectbox(
                    "Hidden layer topology",
                    ["(64, 32)", "(100,)", "(128, 64, 32)", "(32, 16)"],
                    index=0,
                    key=f"mlp_hidden",
                )
                hidden_map = {
                    "(64, 32)": (64, 32),
                    "(100,)": (100,),
                    "(128, 64, 32)": (128, 64, 32),
                    "(32, 16)": (32, 16),
                }
                hp["hidden_layer_sizes"] = hidden_map[hidden_choice]
                hp["learning_rate_init"] = st.slider("Initial learning rate", 0.0005, 0.01, 0.001, step=0.0005, format="%.4f", key=f"mlp_lr")
                hp["max_iter"] = st.slider("Max training epochs", 100, 1000, 500, step=50, key=f"mlp_iter")

            model_hyperparams[model_name] = hp

    st.markdown("---")
    run_benchmark = st.button("🚀 Run Selected Models", type="primary", width="stretch")


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
    st.error(f"⚠️ Error loading training datasets: {e}")
    st.info("Run `python main.py` in the terminal to execute the initial data processing pipeline.")
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
            status_text.markdown(f"⏳ **Training [{idx+1}/{len(selected_models)}]:** `{model_name}`...")
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

        status_text.markdown("✅ **Training completed successfully!**")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        st.rerun()


# Display training error banners if any
if st.session_state.model_errors:
    for err_model, err_msg in st.session_state.model_errors.items():
        st.error(f"❌ Failed to train **{err_model}**: {err_msg}")


# Check if at least one model is available
if not st.session_state.trained_models:
    st.warning("⚠️ No trained models available. Select models from the sidebar and click **Run Selected Models**.")
    st.stop()

le = st.session_state.label_encoder


# -----------------------------------------------------------------------------
# Tabs Layout: Comparison, Diagnostics, Live Simulator
# -----------------------------------------------------------------------------
tab_compare, tab_diagnostics, tab_simulator = st.tabs([
    "📊 Model Comparison & Leaderboard",
    "🔬 Per-Model Diagnostics & Explainability",
    "🎯 Interactive Policy Inference",
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
        table_rows.append({
            "Model": m_name,
            "Macro F1": m_metrics["macro_f1"],
            "Accuracy": m_metrics["accuracy"],
            "Macro Precision": m_metrics["macro_precision"],
            "Macro Recall": m_metrics["macro_recall"],
            "ROC-AUC": roc_str,
            "Storage Savings (%)": m_metrics["savings_pct"],
            "Training Time (s)": m_metrics["training_time_sec"],
            "Needs Scaling": "Yes (Pipeline)" if m_name in SCALED_MODELS else "No",
        })

    comp_df = pd.DataFrame(table_rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)

    # Top KPI Metrics Cards
    if not comp_df.empty:
        best_f1_row = comp_df.iloc[0]
        fastest_row = comp_df.sort_values("Training Time (s)").iloc[0]
        best_savings_row = comp_df.sort_values("Storage Savings (%)", ascending=False).iloc[0]

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.metric("Top Model (Macro F1)", best_f1_row["Model"], f"{best_f1_row['Macro F1']:.4f}")
        with kpi_col2:
            st.metric("Highest Accuracy", comp_df.sort_values('Accuracy', ascending=False).iloc[0]["Model"], f"{comp_df['Accuracy'].max():.2%}")
        with kpi_col3:
            st.metric("Fastest Trainer", fastest_row["Model"], f"{fastest_row['Training Time (s)']:.3f}s")
        with kpi_col4:
            st.metric("Peak Cloud Savings", best_savings_row["Model"], f"{best_savings_row['Storage Savings (%)']:.1f}%")

    st.markdown("### Metrics Table")
    st.dataframe(
        comp_df.style.format({
            "Macro F1": "{:.4f}",
            "Accuracy": "{:.2%}",
            "Macro Precision": "{:.4f}",
            "Macro Recall": "{:.4f}",
            "Storage Savings (%)": "{:.2f}%",
            "Training Time (s)": "{:.4f}",
        }).background_gradient(subset=["Macro F1", "Accuracy", "Storage Savings (%)"], cmap="Blues"),
        width="stretch",
    )

    st.markdown("### Visual Performance Comparison")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Macro F1 Score vs. Accuracy")
        chart_data = comp_df.set_index("Model")[["Macro F1", "Accuracy"]]
        st.bar_chart(chart_data)

    with chart_col2:
        st.markdown("#### Training Duration (seconds)")
        time_chart_data = comp_df.set_index("Model")[["Training Time (s)"]]
        st.bar_chart(time_chart_data)


# -----------------------------------------------------------------------------
# TAB 2: Per-Model Diagnostics & Explainability
# -----------------------------------------------------------------------------
with tab_diagnostics:
    st.subheader("Model Diagnostics & Explainability")
    
    inspect_model = st.selectbox(
        "Select Model to Inspect",
        options=list(st.session_state.trained_models.keys()),
        index=0,
    )

    clf_selected = st.session_state.trained_models[inspect_model]
    metrics_selected = st.session_state.model_metrics[inspect_model]

    diag_col1, diag_col2 = st.columns(2)

    # 1. Feature Importance
    with diag_col1:
        st.markdown(f"#### Feature Importances ({inspect_model})")
        fi_series = get_feature_importances(
            clf=clf_selected,
            feature_names=FEATURE_COLUMNS,
            X_sample=test_df[FEATURE_COLUMNS],
            y_sample=le.transform(test_df["lifecycle_label"]),
        )
        if fi_series is not None and not fi_series.empty:
            fi_df = fi_series.to_frame(name="Importance")
            st.bar_chart(fi_df)
        else:
            st.info("Feature importance not directly supported or available for this estimator structure.")

    # 2. Confusion Matrix
    with diag_col2:
        st.markdown(f"#### Confusion Matrix ({inspect_model})")
        cm = metrics_selected["confusion_matrix"]
        classes = list(le.classes_)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted Action")
        ax.set_ylabel("Actual Action")

        for i in range(len(classes)):
            for j in range(len(classes)):
                text_color = "white" if cm[i, j] > (cm.max() / 2) else "black"
                ax.text(j, i, cm[i, j], ha="center", va="center", color=text_color, fontsize=10)

        plt.title(f"{inspect_model} Confusion Matrix", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.markdown(f"#### Detailed Classification Report ({inspect_model})")

    # Classification report breakdown table
    cr_dict = metrics_selected["class_report"]
    report_rows = []
    for cls_name in le.classes_:
        if cls_name in cr_dict:
            report_rows.append({
                "Action Class": cls_name,
                "Precision": cr_dict[cls_name]["precision"],
                "Recall": cr_dict[cls_name]["recall"],
                "F1-Score": cr_dict[cls_name]["f1-score"],
                "Support": int(cr_dict[cls_name]["support"]),
            })

    if report_rows:
        report_df = pd.DataFrame(report_rows)
        st.dataframe(
            report_df.style.format({
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score": "{:.4f}",
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
        redundancy_score = st.slider("Redundancy score", 0.0, 1.0, 0.35, help="Fraction of chunks shared with other objects")
        object_size = st.number_input("Object size (bytes)", min_value=1, value=65536, step=4096)
        object_age_days = st.slider("Object age (days)", 0, 1000, 45)

    with sim_col2:
        st.markdown("#### Access Frequencies")
        access_freq_7d = st.slider("Accesses in last 7 days", 0, 50, 2)
        access_freq_30d = st.slider("Accesses in last 30 days", 0, 200, 8)
        access_freq_90d = st.slider("Accesses in last 500 days", 0, 500, 18)
        days_since_last_access = st.slider("Days since last access", 0, 365, 4)

    with sim_col3:
        st.markdown("#### Dependencies & Cost")
        num_dependents = st.slider("Number of dependent objects", 0, 50, 0, help="In-degree in dependency graph")
        compression_ratio = st.slider("Compression ratio", 1.0, 10.0, 2.4, step=0.1)
        compute_time_sec = st.number_input("Estimated compute time (sec)", min_value=0.0, value=0.015, format="%.4f")
        monthly_storage_cost_usd = st.number_input("Monthly storage cost (USD)", min_value=0.0, value=0.0015, format="%.6f")

    st.divider()

    if st.button("🔮 Predict Optimal Lifecycle Action", type="primary", width="stretch"):
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
        projected_monthly_savings = monthly_storage_cost_usd * savings_factor

        st.success(f"### Recommended Action ({sim_model_choice}): **{recommended_action.upper()}**")
        st.info(f"💰 Projected Cost Reduction: **{savings_factor * 100:.1f}%** (${projected_monthly_savings:.6f}/month saved on this object)")

        # Probabilities
        if hasattr(active_clf, "predict_proba"):
            probs = active_clf.predict_proba(row)[0]
            prob_df = pd.DataFrame({
                "Action": le.classes_,
                "Probability": probs,
            }).sort_values("Probability", ascending=False)

            pcol1, pcol2 = st.columns([2, 1])
            with pcol1:
                st.markdown("#### Probability Distribution")
                st.bar_chart(prob_df.set_index("Action"))
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
                consensus_rows.append({
                    "Model": m_name,
                    "Predicted Action": m_label.upper(),
                    "Action Factor": f"{ACTION_SAVINGS_FACTOR.get(m_label, 0.0)*100:.0f}%",
                })
            consensus_df = pd.DataFrame(consensus_rows)
            st.dataframe(consensus_df, width="stretch")