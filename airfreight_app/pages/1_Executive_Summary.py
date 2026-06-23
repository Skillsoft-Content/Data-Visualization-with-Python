import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st

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
import joblib
from src.config import MODEL_CONFIG

st.set_page_config(page_title="Executive Summary", layout="wide")
st.title("Executive Summary")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_df() -> pd.DataFrame:
    if "df" not in st.session_state:
        raw = get_sample_data()
        clean, _, excluded = clean_data(raw)
        st.session_state["df"]          = clean
        st.session_state["excluded_df"] = excluded
        st.toast("Demo data loaded. Visit the Home page to load real data.", icon="ℹ")
    return st.session_state["df"]


def _ensure_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if "df_pred" not in st.session_state:
        df_fe = run_feature_engineering(df.copy(), reference_df=df)
        if model_exists():
            try:
                df_pred = predict_win_probability(df_fe)
                df_pred = add_confidence_level(df_pred)
            except Exception:
                df_pred = add_fallback_win_probability(df_fe)
        else:
            df_pred = add_fallback_win_probability(df_fe)
        st.session_state["df_pred"] = df_pred
    return st.session_state["df_pred"]


def _ensure_lane_summary(df_pred: pd.DataFrame) -> pd.DataFrame:
    if "lane_summary" not in st.session_state:
        ls = compute_lane_summary(df_pred)
        ls = apply_recommendations(ls)
        st.session_state["lane_summary"] = ls
    return st.session_state["lane_summary"]


# ── Model training panel ──────────────────────────────────────────────────────

with st.expander("Model Training", expanded=not model_exists()):
    if model_exists():
        st.success("A trained model is available. Predictions are active.")
    else:
        st.info("No trained model found. Train now to enable win-probability predictions.")

    if st.button("Train Model on Loaded Data", type="primary"):
        df_raw = _ensure_df()
        progress_bar = st.progress(0, text="Starting…")

        def _cb(pct, msg):
            progress_bar.progress(pct, text=msg)

        with st.spinner("Training in progress…"):
            try:
                model, name, cv_scores, feature_cols, encoders, test_df, X_test, y_test = (
                    train_and_select_best(df_raw, progress_callback=_cb)
                )
                metrics = evaluate(model, X_test, y_test,
                                   feature_cols=feature_cols, model_name=name)
                joblib.dump(metrics, MODEL_CONFIG["metrics_path"])
                st.session_state["model_metrics"] = metrics

                # Invalidate cached predictions
                for k in ["df_pred", "lane_summary"]:
                    st.session_state.pop(k, None)

                st.success(
                    f"Best model: **{name}** | "
                    f"ROC AUC: {metrics['roc_auc']:.3f} | "
                    f"F1: {metrics['f1']:.3f}"
                )
                st.write("CV AUC scores per model:", cv_scores)
            except Exception as ex:
                st.error(f"Training failed: {ex}")

# ── Load data ─────────────────────────────────────────────────────────────────

df       = _ensure_df()
df_pred  = _ensure_predictions(df)
lane_df  = _ensure_lane_summary(df_pred)

using_model = "WIN_PROBABILITY" in df_pred.columns and model_exists()

# ── KPIs ──────────────────────────────────────────────────────────────────────

total_quotes    = df["QUOTE_GROUP_ID"].nunique()
total_ships     = int(df["WON"].sum())
actual_wr       = df["WON"].mean()
predicted_wr    = df_pred["WIN_PROBABILITY"].mean() if using_model else actual_wr
total_revenue   = df[df["WON"] == 1]["REVENUE"].sum()
total_grate     = df[df["WON"] == 1]["GRATE"].sum()
avg_grate_ship  = (total_grate / total_ships) if total_ships else 0
won_kg          = df.loc[df["WON"] == 1, "QUOTE_WGT_KG"].sum()
avg_grate_kg    = (total_grate / won_kg) if won_kg else 0

low_wr_mask     = lane_df["ACTUAL_WIN_RATE"] < 0.30
low_wr_lanes    = int(low_wr_mask.sum())
opp_mask        = lane_df["RECOMMENDATION"] == "Lower price selectively"
opp_count       = int(opp_mask.sum())
est_grate_lift  = lane_df.loc[opp_mask, "EXPECTED_GRATE_LIFT"].sum()
est_ship_lift   = lane_df.loc[opp_mask, "EXPECTED_SHIPMENT_LIFT"].sum()

st.subheader("Key Performance Indicators")
r1 = st.columns(4)
r2 = st.columns(4)

with r1[0]: kpi_card("Total Quotes",       f"{total_quotes:,}")
with r1[1]: kpi_card("Total Shipments",    f"{total_ships:,}")
with r1[2]: kpi_card("Actual Win Rate",    format_pct(actual_wr))
with r1[3]: kpi_card("Predicted Win Rate", format_pct(predicted_wr),
                     delta=f"{'Model active' if using_model else 'Fallback mode'}")

with r2[0]: kpi_card("Revenue",              format_currency(total_revenue))
with r2[1]: kpi_card("GRATE",                format_currency(total_grate))
with r2[2]: kpi_card("Avg GRATE/Shipment",   format_currency(avg_grate_ship))
with r2[3]: kpi_card("Avg GRATE/KG",         format_currency(avg_grate_kg))

st.divider()
r3 = st.columns(4)
with r3[0]: kpi_card("Low Win-Rate Lanes",           str(low_wr_lanes))
with r3[1]: kpi_card("Opportunity Lanes",            str(opp_count))
with r3[2]: kpi_card("Est. GRATE Lift (opps)",       format_currency(est_grate_lift))
with r3[3]: kpi_card("Est. Shipment Lift (opps)",    f"{est_ship_lift:.0f}")

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(win_rate_trend(df_pred), use_container_width=True)
with c2:
    st.plotly_chart(grate_trend(df_pred), use_container_width=True)

st.plotly_chart(quote_vs_shipment_volume(df_pred), use_container_width=True)

st.plotly_chart(top_opportunity_lanes(lane_df), use_container_width=True)

c3, c4, c5 = st.columns(3)
with c3:
    st.plotly_chart(win_rate_by_dimension(df_pred, "PRODUCT",      "Win Rate by Product"),
                    use_container_width=True)
with c4:
    st.plotly_chart(win_rate_by_dimension(df_pred, "RATE_TYPE",    "Win Rate by Rate Type"),
                    use_container_width=True)
with c5:
    if "QUOTE_SOURCE" in df_pred.columns:
        st.plotly_chart(win_rate_by_dimension(df_pred, "QUOTE_SOURCE", "Win Rate by Quote Source"),
                        use_container_width=True)

st.plotly_chart(scatter_win_rate_vs_grate(lane_df), use_container_width=True)
