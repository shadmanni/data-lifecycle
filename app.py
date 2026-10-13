import os
import sys
import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(__file__))

st.set_page_config(page_title="Cloud Data Lifecycle Advisor", layout="centered")
st.title("Intelligent Cloud Data Lifecycle Optimization")
st.write("Adjust the sliders to describe a data object and see the recommended lifecycle action.")

clf = joblib.load(os.path.join("models", "lifecycle_policy_model.pkl"))
le = joblib.load(os.path.join("models", "label_encoder.pkl"))

redundancy_score = st.slider("Redundancy score", 0.0, 1.0, 0.3)
object_size = st.number_input("Object size (bytes)", value=100000)
object_age_days = st.slider("Object age (days)", 0, 1000, 30)
access_freq_7d = st.slider("Accesses in last 7 days", 0, 20, 2)
access_freq_30d = st.slider("Accesses in last 30 days", 0, 50, 5)
access_freq_90d = st.slider("Accesses in last 90 days", 0, 100, 10)
days_since_last_access = st.slider("Days since last access", 0, 365, 5)
num_dependents = st.slider("Number of dependent objects", 0, 20, 0)
compression_ratio = st.slider("Compression ratio", 1.0, 10.0, 2.0)
compute_time_sec = st.number_input("Estimated compute time (sec)", value=0.01)
monthly_storage_cost_usd = st.number_input("Monthly storage cost (USD)", value=0.001)

if st.button("Get Recommendation"):
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

    st.success(f"Recommended action: **{label.upper()}**")
    prob_df = pd.DataFrame({"action": le.classes_, "probability": probs}).sort_values("probability", ascending=False)
    st.bar_chart(prob_df.set_index("action"))