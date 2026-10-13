import os
import sys
import joblib
import pandas as pd
import streamlit as st

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

st.set_page_config(page_title="Cloud Data Lifecycle Advisor", layout="wide")
st.title("Cloud Data Lifecycle Policy Optimization")
st.caption("Machine-learning guided data lifecycle actions trained on production Linux kernel, GCC, and Azure trace datasets.")

MODEL_PATH = os.path.join("models", "lifecycle_policy_model.pkl")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")
CM_PLOT_PATH = os.path.join("models", "confusion_matrix.png")
FI_PLOT_PATH = os.path.join("models", "feature_importance.png")


@st.cache_resource
def load_models():
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        clf = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        return clf, le
    return None, None


clf, le = load_models()

if clf is None or le is None:
    st.warning("⚠️ Trained model artifacts not found in `models/`.")
    st.info("Run `python main.py` or `python run_pipeline.py` in the terminal to execute the full pipeline, or click below to train immediately.")
    if st.button("Train Policy Model Now"):
        with st.spinner("Training model on dataset..."):
            from src.training.train_policy import train
            train()
            st.cache_resource.clear()
            st.success("Model trained successfully!")
            st.rerun()
    st.stop()

# Sidebar for Model Insights
with st.sidebar:
    st.header("Model Performance & Diagnostics")
    st.write(f"**Target Classes ({len(le.classes_)}):**")
    st.write(", ".join(f"`{c}`" for c in le.classes_))
    
    if os.path.exists(FI_PLOT_PATH):
        st.subheader("Feature Importance")
        st.image(FI_PLOT_PATH, use_container_width=True)
    if os.path.exists(CM_PLOT_PATH):
        st.subheader("Confusion Matrix")
        st.image(CM_PLOT_PATH, use_container_width=True)

# Main Form
st.subheader("Object Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Redundancy & Size")
    redundancy_score = st.slider("Redundancy score", 0.0, 1.0, 0.35, help="Fraction of chunks shared with other objects")
    object_size = st.number_input("Object size (bytes)", min_value=1, value=65536, step=4096)
    object_age_days = st.slider("Object age (days)", 0, 1000, 45)

with col2:
    st.markdown("#### Access Frequencies")
    access_freq_7d = st.slider("Accesses in last 7 days", 0, 50, 2)
    access_freq_30d = st.slider("Accesses in last 30 days", 0, 200, 8)
    access_freq_90d = st.slider("Accesses in last 90 days", 0, 500, 18)
    days_since_last_access = st.slider("Days since last access", 0, 365, 4)

with col3:
    st.markdown("#### Dependencies & Cost")
    num_dependents = st.slider("Number of dependent objects", 0, 50, 0, help="In-degree in dependency graph")
    compression_ratio = st.slider("Compression ratio", 1.0, 10.0, 2.4, step=0.1)
    compute_time_sec = st.number_input("Estimated compute time (sec)", min_value=0.0, value=0.015, format="%.4f")
    monthly_storage_cost_usd = st.number_input("Monthly storage cost (USD)", min_value=0.0, value=0.0015, format="%.6f")

st.divider()

if st.button("Predict Optimal Lifecycle Action", type="primary"):
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
    
    pred = clf.predict(row)
    probs = clf.predict_proba(row)[0]
    label = le.inverse_transform(pred)[0]

    st.success(f"### Recommended Action: **{label.upper()}**")
    
    prob_df = pd.DataFrame({
        "Action": le.classes_,
        "Probability": probs
    }).sort_values("Probability", ascending=False)
    
    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        st.bar_chart(prob_df.set_index("Action"))
    with col_table:
        st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}), use_container_width=True)
