import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st
import joblib

from src.utils import get_sample_data, format_currency, format_pct, kpi_card
from src.data_cleaning import clean_data
from src.feature_engineering import run_feature_engineering
from src.predict import predict_win_probability, add_confidence_level, add_fallback_win_probability, model_exists
from src.recommendation_engine import compute_lane_summary, apply_recommendations
from src.visualizations import (
    win_rate_trend, grate_trend, quote_vs_shipment_volume,
    top_opportunity_lanes, win_rate_by_dimension, scatter_win_rate_vs_grate,
)
from src.train_model import train_and_select_best
from src.evaluate_model import evaluate
from src.config import MODEL_CONFIG

st.set_page_config(page_title="Executive Summary", layout="wide")
st.title("Executive Summary")


def _ensure_df():
    if "df" not in st.session_state:
        raw = get_sample_data()
        clean, _, excl = clean_data(raw)
        st.session_state["df"]          = clean
        st.session_state["excluded_df"] = excl
        st.toast("Demo data loaded. Visit the Home page to connect real data.", icon="ℹ")
    return st.session_state["df"]


def _ensure_predictions(df):
    if "df_pred" not in st.session_state:
        df_fe = run_feature_engineering(df.copy(), reference_df=df)
        if model_exists():
            try:
                df_fe = predict_win_probability(df_fe)
                df_fe = add_confidence_level(df_fe)
            except Exception:
                df_fe = add_fallback_win_probability(df_fe)
        else:
            df_fe = add_fallback_win_probability(df_fe)
        st.session_state["df_pred"] = df_fe
    return st.session_state["df_pred"]


def _ensure_lane_summary(df_pred):
    if "lane_summary" not in st.session_state:
        ls = compute_lane_summary(df_pred)
        ls = apply_recommendations(ls)
        st.session_state["lane_summary"] = ls
    return st.session_state["lane_summary"]


# ── Model training ─────────────────────────────────────────────────────────────

with st.expander("Model Training", expanded=not model_exists()):
    if model_exists():
        st.success("Trained model is available. Predictions are active.")
    else:
        st.info("No trained model found. Train now to enable win-probability predictions.")

    if st.button("Train Model on Loaded Data", type="primary"):
        df_raw = _ensure_df()
        bar = st.progress(0, text="Starting...")
        with st.spinner("Training..."):
            try:
                model, name, cv_scores, feature_cols, encoders, test_df, X_test, y_test = (
                    train_and_select_best(df_raw, progress_callback=lambda p, m: bar.progress(p, text=m))
                )
                metrics = evaluate(model, X_test, y_test, feature_cols=feature_cols, model_name=name)
                joblib.dump(metrics, MODEL_CONFIG["metrics_path"])
                st.session_state["model_metrics"] = metrics
                for k in ["df_pred", "lane_summary"]:
                    st.session_state.pop(k, None)
                st.success(
                    f"Best model: **{name}** | ROC AUC: {metrics['roc_auc']:.3f} | F1: {metrics['f1']:.3f}"
                )
                st.write("CV AUC by model:", cv_scores)
            except Exception as ex:
                st.error(f"Training failed: {ex}")

# ── Load data ────────────────────────────────────────────────────────────────

df      = _ensure_df()
df_pred = _ensure_predictions(df)
lane_df = _ensure_lane_summary(df_pred)

using_model = "WIN_PROBABILITY" in df_pred.columns and model_exists()

# ── KPIs ────────────────────────────────────────────────────────────────────

total_quotes   = df["QUOTE_GROUP_ID"].nunique()
total_ships    = int(df["WON"].sum())
actual_wr      = df["WON"].mean()
predicted_wr   = df_pred["WIN_PROBABILITY"].mean() if using_model else actual_wr
total_revenue  = df[df["WON"]==1]["REVENUE"].sum()
total_grate    = df[df["WON"]==1]["GRATE"].sum()
avg_grate_ship = (total_grate / total_ships) if total_ships else 0
won_kg         = df.loc[df["WON"]==1, "QUOTE_WGT_KG"].sum()
avg_grate_kg   = (total_grate / won_kg) if won_kg else 0

opp_mask      = lane_df["RECOMMENDATION"] == "Lower price selectively"
low_wr_lanes  = int((lane_df["ACTUAL_WIN_RATE"] < 0.30).sum())
opp_count     = int(opp_mask.sum())
est_grate_lift= lane_df.loc[opp_mask, "EXPECTED_GRATE_LIFT"].sum()
est_ship_lift = lane_df.loc[opp_mask, "EXPECTED_SHIPMENT_LIFT"].sum()

st.subheader("Key Performance Indicators")
r1 = st.columns(4)
r1[0].metric("Total Quotes",       f"{total_quotes:,}")
r1[1].metric("Total Shipments",    f"{total_ships:,}")
r1[2].metric("Actual Win Rate",    format_pct(actual_wr))
r1[3].metric("Predicted Win Rate", format_pct(predicted_wr),
             delta="Model active" if using_model else "Fallback mode")

r2 = st.columns(4)
r2[0].metric("Revenue",             format_currency(total_revenue))
r2[1].metric("GRATE",               format_currency(total_grate))
r2[2].metric("Avg GRATE/Shipment",  format_currency(avg_grate_ship))
r2[3].metric("Avg GRATE/KG",        format_currency(avg_grate_kg))

st.divider()
r3 = st.columns(4)
r3[0].metric("Low Win-Rate Lanes",       str(low_wr_lanes))
r3[1].metric("Opportunity Lanes",        str(opp_count))
r3[2].metric("Est. GRATE Lift (opps)",   format_currency(est_grate_lift))
r3[3].metric("Est. Shipment Lift (opps)",f"{est_ship_lift:.0f}")

st.divider()

# ── Charts ───────────────────────────────────────────────────────────────────

c1, c2 = st.columns(2)
with c1: st.plotly_chart(win_rate_trend(df_pred),            use_container_width=True)
with c2: st.plotly_chart(grate_trend(df_pred),               use_container_width=True)
st.plotly_chart(quote_vs_shipment_volume(df_pred),           use_container_width=True)
st.plotly_chart(top_opportunity_lanes(lane_df),              use_container_width=True)

c3, c4, c5 = st.columns(3)
with c3: st.plotly_chart(win_rate_by_dimension(df_pred, "PRODUCT",      "Win Rate by Product"),      use_container_width=True)
with c4: st.plotly_chart(win_rate_by_dimension(df_pred, "RATE_TYPE",    "Win Rate by Rate Type"),    use_container_width=True)
with c5:
    if "QUOTE_SOURCE" in df_pred.columns:
        st.plotly_chart(win_rate_by_dimension(df_pred, "QUOTE_SOURCE", "Win Rate by Quote Source"), use_container_width=True)

st.plotly_chart(scatter_win_rate_vs_grate(lane_df), use_container_width=True)
