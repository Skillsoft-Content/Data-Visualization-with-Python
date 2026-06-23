import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
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


# ── Load metrics ──────────────────────────────────────────────────────────────

if "model_metrics" not in st.session_state:
    if os.path.exists(MODEL_CONFIG["metrics_path"]):
        try:
            st.session_state["model_metrics"] = joblib.load(MODEL_CONFIG["metrics_path"])
        except Exception:
            pass

metrics = st.session_state.get("model_metrics")

if not model_exists() or metrics is None:
    st.info(
        "No trained model found. Navigate to **Executive Summary** and click "
        "**Train Model on Loaded Data** to generate model metrics."
    )
    st.stop()

# ── Model metadata ────────────────────────────────────────────────────────────

st.subheader("Model Information")
meta_cols = st.columns(4)
meta_cols[0].metric("Model",        metrics.get("model_name", "N/A"))
meta_cols[1].metric("Trained at",   metrics.get("trained_at", "N/A"))
meta_cols[2].metric("Test rows",    f"{metrics.get('n_test', 0):,}")
meta_cols[3].metric("Actual WR (test)", f"{metrics.get('actual_win_rate', 0):.1%}")

st.divider()

# ── Core metrics ──────────────────────────────────────────────────────────────

st.subheader("Core Metrics")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("ROC AUC",    f"{metrics.get('roc_auc', 0):.3f}")
m2.metric("Precision",  f"{metrics.get('precision', 0):.3f}")
m3.metric("Recall",     f"{metrics.get('recall', 0):.3f}")
m4.metric("F1 Score",   f"{metrics.get('f1', 0):.3f}")
m5.metric("Brier Score",f"{metrics.get('brier_score', 0):.4f}",
          help="Lower is better. 0 = perfect, 0.25 = random.")

st.divider()

# ── Charts row 1 ─────────────────────────────────────────────────────────────

c1, c2 = st.columns(2)

with c1:
    if "roc_curve" in metrics:
        st.plotly_chart(roc_curve_chart(metrics["roc_curve"], metrics["roc_auc"]),
                        use_container_width=True)
with c2:
    if "calibration" in metrics:
        st.plotly_chart(calibration_chart(metrics["calibration"]),
                        use_container_width=True)

# ── Charts row 2 ─────────────────────────────────────────────────────────────

c3, c4 = st.columns(2)

with c3:
    if "confusion_matrix" in metrics:
        st.plotly_chart(confusion_matrix_chart(metrics["confusion_matrix"]),
                        use_container_width=True)
with c4:
    if "lift_chart" in metrics and not metrics["lift_chart"].empty:
        st.plotly_chart(lift_chart(metrics["lift_chart"]),
                        use_container_width=True)

# ── Feature importance ────────────────────────────────────────────────────────

if "feature_importance" in metrics:
    st.divider()
    st.subheader("Feature Importance")
    fi = metrics["feature_importance"]
    st.plotly_chart(feature_importance_chart(fi, top_n=25), use_container_width=True)

    with st.expander("View full feature importance table"):
        st.dataframe(fi.reset_index(drop=True), use_container_width=True)

# ── Actual vs predicted win rate by probability band ─────────────────────────

if "lift_chart" in metrics and not metrics["lift_chart"].empty:
    st.divider()
    st.subheader("Actual vs Predicted Win Rate by Probability Band")
    lc = metrics["lift_chart"].copy()
    lc["decile"] = lc["decile"].astype(str)
    lc = lc.rename(columns={
        "decile": "Probability Decile",
        "avg_prob": "Avg Predicted",
        "actual_win_rate": "Actual Win Rate",
        "count": "Quote Count",
    })
    st.dataframe(
        lc.style.format({
            "Avg Predicted":  "{:.1%}",
            "Actual Win Rate":"{:.1%}",
            "Quote Count":    "{:,}",
        }),
        use_container_width=True,
    )
