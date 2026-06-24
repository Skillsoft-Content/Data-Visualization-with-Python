import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st

from src.utils import get_sample_data, format_currency, format_pct
from src.data_cleaning import clean_data
from src.feature_engineering import run_feature_engineering
from src.predict import predict_win_probability, add_confidence_level, add_fallback_win_probability, model_exists
from src.visualizations import (
    drilldown_monthly, distribution_chart, mix_chart, won_vs_lost_comparison,
    win_rate_by_dimension,
)

st.set_page_config(page_title="Lane Drilldown", layout="wide")
st.title("Lane Drilldown")


# ── Data setup ────────────────────────────────────────────────────────────────

def _get_pred_df() -> pd.DataFrame:
    if "df_pred" not in st.session_state:
        if "df" not in st.session_state:
            raw = get_sample_data()
            clean, _, excl = clean_data(raw)
            st.session_state["df"]          = clean
            st.session_state["excluded_df"] = excl
        df = st.session_state["df"]
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


df_pred = _get_pred_df()

# ── Lane selector ─────────────────────────────────────────────────────────────

all_lanes = sorted(df_pred["LANE"].dropna().unique().tolist()) if "LANE" in df_pred.columns else []
default_lane = st.session_state.get("selected_lane", all_lanes[0] if all_lanes else None)

selected_lane = st.selectbox(
    "Select lane",
    all_lanes,
    index=all_lanes.index(default_lane) if default_lane in all_lanes else 0,
)
st.session_state["selected_lane"] = selected_lane

lane_df = df_pred[df_pred["LANE"] == selected_lane].copy() if "LANE" in df_pred.columns else pd.DataFrame()

if lane_df.empty:
    st.warning(f"No data found for lane: {selected_lane}")
    st.stop()

# ── Header KPIs ───────────────────────────────────────────────────────────────

n_quotes   = lane_df["QUOTE_GROUP_ID"].nunique()
n_ships    = int(lane_df["WON"].sum())
actual_wr  = lane_df["WON"].mean()
pred_wr    = lane_df["WIN_PROBABILITY"].mean() if "WIN_PROBABILITY" in lane_df else actual_wr
total_gr   = lane_df[lane_df["WON"] == 1]["GRATE"].sum()
avg_kg     = lane_df["QUOTE_WGT_KG"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Quotes",          f"{n_quotes:,}")
c2.metric("Shipments",       f"{n_ships:,}")
c3.metric("Actual Win Rate", format_pct(actual_wr))
c4.metric("Predicted WR",    format_pct(pred_wr))
c5.metric("Total GRATE",     format_currency(total_gr))

st.divider()

# ── Monthly trends ────────────────────────────────────────────────────────────

st.subheader("Monthly Trends")
fig_vol, fig_wr, fig_gr = drilldown_monthly(lane_df)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_vol, use_container_width=True)
with col2:
    st.plotly_chart(fig_wr, use_container_width=True)

st.plotly_chart(fig_gr, use_container_width=True)

# ── Mix charts ────────────────────────────────────────────────────────────────

st.subheader("Mix Analysis")
mix_cols = [
    ("CUSTOMER_NAME",        "Customer Mix"),
    ("RATE_TYPE",            "Rate Type Mix"),
    ("QUOTE_SOURCE",         "Quote Source Mix"),
    ("CUSTOMER_CLASSIFICATION", "Customer Classification Mix"),
]
available_mixes = [(c, t) for c, t in mix_cols if c in lane_df.columns]

if available_mixes:
    mc_cols = st.columns(min(len(available_mixes), 3))
    for i, (col, title) in enumerate(available_mixes[:3]):
        with mc_cols[i % 3]:
            st.plotly_chart(mix_chart(lane_df, col, title), use_container_width=True)

# ── Distributions ─────────────────────────────────────────────────────────────

st.subheader("Distributions")
d1, d2, d3 = st.columns(3)
with d1:
    if "QUOTE_WGT_KG" in lane_df.columns:
        st.plotly_chart(distribution_chart(lane_df, "QUOTE_WGT_KG", "Weight Distribution (KG)"),
                        use_container_width=True)
with d2:
    if "DISCOUNT_PCT" in lane_df.columns:
        st.plotly_chart(distribution_chart(lane_df, "DISCOUNT_PCT", "Discount Distribution", is_pct=True),
                        use_container_width=True)
with d3:
    if "PRICE_PER_KG" in lane_df.columns:
        st.plotly_chart(distribution_chart(lane_df, "PRICE_PER_KG", "Price per KG Distribution"),
                        use_container_width=True)

# ── Won vs lost comparison ────────────────────────────────────────────────────

st.subheader("Won vs Lost Comparison")
st.plotly_chart(won_vs_lost_comparison(lane_df), use_container_width=True)

# ── Quote-level table ─────────────────────────────────────────────────────────

st.subheader("Quote-Level Detail")

quote_cols = {
    "DIGITAL_QUOTE_ID":  "Quote ID",
    "QUOTE_GROUP_ID":    "Group ID",
    "QUOTE_DATE":        "Date",
    "CUSTOMER_NAME":     "Customer",
    "LASSO_ID":          "Customer ID",
    "PRODUCT":           "Product",
    "SERVICE_PAIR":      "Service Pair",
    "RATE_TYPE":         "Rate Type",
    "QUOTE_SOURCE":      "Source",
    "QUOTE_WGT_KG":      "KG",
    "NET_QUOTE_PRICE_USD":"Net Price ($)",
    "PRICE_PER_KG":      "Price/KG ($)",
    "DISCOUNT_PCT":      "Disc %",
    "WIN_PROBABILITY":   "Pred. WR",
    "WON":               "Won",
    "REVENUE":           "Revenue ($)",
    "GRATE":             "GRATE ($)",
    "CONFIDENCE_LEVEL":  "Confidence",
}

avail = {k: v for k, v in quote_cols.items() if k in lane_df.columns}
q_out = lane_df[list(avail.keys())].rename(columns=avail).copy()

# Format
for c in ["Disc %", "Pred. WR"]:
    if c in q_out.columns:
        q_out[c] = q_out[c].apply(format_pct)
for c in ["Net Price ($)", "Revenue ($)", "GRATE ($)", "Price/KG ($)"]:
    if c in q_out.columns:
        q_out[c] = q_out[c].apply(format_currency)

st.dataframe(q_out, use_container_width=True, height=400)

# Store for downstream
st.session_state["drilldown_lane"]   = selected_lane
st.session_state["drilldown_quotes"] = lane_df
