import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib

from src.config import MODEL_CONFIG
from src.predict import model_exists
from src.visualizations import (
    roc_curve_chart, calibration_chart, confusion_matrix_chart,
    feature_importance_chart, lift_chart,
)

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("Model Performance")

if "model_metrics" not in st.session_state:
    import os
    if os.path.exists(MODEL_CONFIG["metrics_path"]):
        try:
            st.session_state["model_metrics"] = joblib.load(MODEL_CONFIG["metrics_path"])
        except Exception:
            pass

metrics = st.session_state.get("model_metrics")

if not model_exists() or metrics is None:
    st.info(
        "No trained model found. Go to **Executive Summary** and click "
        "**Train Model on Loaded Data** to generate metrics."
    )
    st.stop()

st.subheader("Model Information")
mc = st.columns(4)
mc[0].metric("Model",            metrics.get("model_name", "N/A"))
mc[1].metric("Trained at",       metrics.get("trained_at", "N/A"))
mc[2].metric("Test rows",        f"{metrics.get('n_test',0):,}")
mc[3].metric("Actual WR (test)", f"{metrics.get('actual_win_rate',0):.1%}")

st.divider()
st.subheader("Core Metrics")
m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("ROC AUC",    f"{metrics.get('roc_auc',0):.3f}")
m2.metric("Precision",  f"{metrics.get('precision',0):.3f}")
m3.metric("Recall",     f"{metrics.get('recall',0):.3f}")
m4.metric("F1 Score",   f"{metrics.get('f1',0):.3f}")
m5.metric("Brier Score",f"{metrics.get('brier_score',0):.4f}")

st.divider()
c1,c2 = st.columns(2)
with c1:
    if "roc_curve" in metrics:
        st.plotly_chart(roc_curve_chart(metrics["roc_curve"], metrics["roc_auc"]), use_container_width=True)
with c2:
    if "calibration" in metrics:
        st.plotly_chart(calibration_chart(metrics["calibration"]), use_container_width=True)

c3,c4 = st.columns(2)
with c3:
    if "confusion_matrix" in metrics:
        st.plotly_chart(confusion_matrix_chart(metrics["confusion_matrix"]), use_container_width=True)
with c4:
    if "lift_chart" in metrics and not metrics["lift_chart"].empty:
        st.plotly_chart(lift_chart(metrics["lift_chart"]), use_container_width=True)

if "feature_importance" in metrics:
    st.divider()
    st.subheader("Feature Importance")
    st.plotly_chart(feature_importance_chart(metrics["feature_importance"], top_n=25), use_container_width=True)
    with st.expander("Full feature importance table"):
        st.dataframe(metrics["feature_importance"].reset_index(drop=True), use_container_width=True)
