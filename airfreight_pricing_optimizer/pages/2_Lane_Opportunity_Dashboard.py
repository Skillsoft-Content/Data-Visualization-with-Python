import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st

from src.utils import get_sample_data, format_currency, format_pct
from src.data_cleaning import clean_data
from src.feature_engineering import run_feature_engineering
from src.predict import predict_win_probability, add_confidence_level, add_fallback_win_probability, model_exists
from src.recommendation_engine import compute_lane_summary, apply_recommendations

st.set_page_config(page_title="Lane Opportunity Dashboard", layout="wide")
st.title("Lane Opportunity Dashboard")


def _get_df():
    if "df" not in st.session_state:
        raw = get_sample_data()
        clean, _, excl = clean_data(raw)
        st.session_state["df"]          = clean
        st.session_state["excluded_df"] = excl
    return st.session_state["df"]


def _get_lane_summary(df):
    if "lane_summary" not in st.session_state:
        df_fe = run_feature_engineering(df.copy(), reference_df=df)
        if model_exists():
            try:
                df_fe = predict_win_probability(df_fe)
                df_fe = add_confidence_level(df_fe)
            except Exception:
                df_fe = add_fallback_win_probability(df_fe)
        else:
            df_fe = add_fallback_win_probability(df_fe)
        ls = compute_lane_summary(df_fe)
        ls = apply_recommendations(ls)
        st.session_state["lane_summary"] = ls
        st.session_state["df_pred"]      = df_fe
    return st.session_state["lane_summary"]


df      = _get_df()
lane_df = _get_lane_summary(df)

# ── Filters ─────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

products      = ["All"] + sorted(lane_df["PRODUCT"].dropna().unique().tolist())
rate_types    = ["All"] + sorted(lane_df["RATE_TYPE"].dropna().unique().tolist())
service_pairs = ["All"] + sorted(lane_df["SERVICE_PAIR"].dropna().unique().tolist())
recs          = ["All"] + sorted(lane_df["RECOMMENDATION"].dropna().unique().tolist())

sel_product = st.sidebar.selectbox("Product",        products)
sel_rt      = st.sidebar.selectbox("Rate type",      rate_types)
sel_sp      = st.sidebar.selectbox("Service pair",   service_pairs)
sel_rec     = st.sidebar.selectbox("Recommendation", recs)

st.sidebar.divider()
min_quotes = st.sidebar.slider("Min quote count",  0, 200, 5,   step=5)
max_wr     = st.sidebar.slider("Max win rate (%)", 0, 100, 100, step=5) / 100
min_grate  = st.sidebar.number_input("Min total GRATE ($)", value=0, step=1000)

sort_by = st.sidebar.selectbox("Sort by", [
    "Lowest win rate", "Highest expected GRATE lift", "Highest shipment lift",
    "Highest quote volume", "Highest GRATE opportunity", "Highest confidence",
])

# ── Apply filters ───────────────────────────────────────────────────────────────
filt = lane_df.copy()
if sel_product != "All": filt = filt[filt["PRODUCT"]       == sel_product]
if sel_rt      != "All": filt = filt[filt["RATE_TYPE"]      == sel_rt]
if sel_sp      != "All": filt = filt[filt["SERVICE_PAIR"]   == sel_sp]
if sel_rec     != "All": filt = filt[filt["RECOMMENDATION"] == sel_rec]
filt = filt[(filt["QUOTE_COUNT"] >= min_quotes) &
            (filt["ACTUAL_WIN_RATE"] <= max_wr) &
            (filt["GRATE"] >= min_grate)]

sort_map = {
    "Lowest win rate":             ("ACTUAL_WIN_RATE",       True),
    "Highest expected GRATE lift": ("EXPECTED_GRATE_LIFT",   False),
    "Highest shipment lift":       ("EXPECTED_SHIPMENT_LIFT",False),
    "Highest quote volume":        ("QUOTE_COUNT",           False),
    "Highest GRATE opportunity":   ("EXPECTED_GRATE",        False),
    "Highest confidence":          ("CONFIDENCE_LEVEL",      False),
}
scol, sasc = sort_map[sort_by]
if scol in filt.columns:
    filt = filt.sort_values(scol, ascending=sasc)

st.markdown(f"**{len(filt):,} lanes** match current filters")

display_cols = {
    "LANE":"Lane","PRODUCT":"Product","SERVICE_PAIR":"Service Pair","RATE_TYPE":"Rate Type",
    "QUOTE_COUNT":"Quotes","SHIPMENT_COUNT":"Shipments","ACTUAL_WIN_RATE":"Actual WR",
    "PREDICTED_WIN_RATE":"Predicted WR","REVENUE":"Revenue ($)","GRATE":"GRATE ($)",
    "GRATE_PER_SHIPMENT":"GRATE/Ship ($)","GRATE_PER_KG":"GRATE/KG ($)",
    "AVG_KG":"Avg KG","AVG_PRICE_PER_KG":"Price/KG ($)","AVG_DISCOUNT_PCT":"Avg Disc (%)",
    "EXPECTED_SHIPMENTS":"Exp. Ships","EXPECTED_GRATE":"Exp. GRATE ($)",
    "SCENARIO_EXPECTED_SHIPMENTS":"Scen. Ships","SCENARIO_EXPECTED_GRATE":"Scen. GRATE ($)",
    "EXPECTED_SHIPMENT_LIFT":"Ship. Lift","EXPECTED_GRATE_LIFT":"GRATE Lift ($)",
    "RECOMMENDATION":"Recommendation","CONFIDENCE_LEVEL":"Confidence","REASON":"Reason",
}
avail = {k: v for k, v in display_cols.items() if k in filt.columns}
out   = filt[list(avail.keys())].rename(columns=avail).copy()

for c in ["Actual WR","Predicted WR","Avg Disc (%)"]:
    if c in out: out[c] = out[c].apply(format_pct)
for c in ["Revenue ($)","GRATE ($)","GRATE/Ship ($)","GRATE/KG ($)","Price/KG ($)",
          "Exp. GRATE ($)","Scen. GRATE ($)","GRATE Lift ($)"]:
    if c in out: out[c] = out[c].apply(format_currency)

st.dataframe(out, use_container_width=True, height=600)

st.divider()
lane_options = filt["LANE"].tolist() if "LANE" in filt.columns else []
if lane_options:
    selected = st.selectbox("Select a lane to drill into:", [""] + lane_options)
    if selected:
        st.session_state["selected_lane"] = selected
        st.success(f"Lane **{selected}** selected. Navigate to **Drilldown** to explore.")
