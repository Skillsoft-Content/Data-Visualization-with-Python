import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils import get_sample_data, format_currency, format_pct
from src.data_cleaning import clean_data
from src.feature_engineering import run_feature_engineering
from src.predict import add_fallback_win_probability, model_exists, predict_win_probability, add_confidence_level
from src.recommendation_engine import compute_lane_summary, apply_recommendations
from src.scenario_engine import run_scenario
from src.config import THRESHOLDS

st.set_page_config(page_title="Scenario Simulator", layout="wide")
st.title("Scenario Simulator")
st.markdown(
    "Select a lane and adjust pricing parameters to model the expected impact "
    "on win probability, shipment volume, and total GRATE."
)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _ensure_lane_summary() -> pd.DataFrame:
    if "lane_summary" not in st.session_state:
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
        ls = compute_lane_summary(df_fe)
        ls = apply_recommendations(ls)
        st.session_state["lane_summary"] = ls
        st.session_state["df_pred"]      = df_fe
    return st.session_state["lane_summary"]


lane_df = _ensure_lane_summary()

# ── Lane selector ─────────────────────────────────────────────────────────────

col_a, col_b, col_c = st.columns(3)

with col_a:
    all_lanes    = sorted(lane_df["LANE"].dropna().unique().tolist())
    default_lane = st.session_state.get("selected_lane", all_lanes[0] if all_lanes else None)
    sel_lane     = st.selectbox("Lane", all_lanes,
                                index=all_lanes.index(default_lane) if default_lane in all_lanes else 0)

lane_mask = lane_df["LANE"] == sel_lane
sub       = lane_df[lane_mask]

products   = sorted(sub["PRODUCT"].dropna().unique().tolist()) if not sub.empty else []
rate_types = sorted(sub["RATE_TYPE"].dropna().unique().tolist()) if not sub.empty else []

with col_b:
    sel_prod = st.selectbox("Product",   ["All"] + products)
with col_c:
    sel_rt   = st.selectbox("Rate type", ["All"] + rate_types)

if sel_prod != "All": sub = sub[sub["PRODUCT"]   == sel_prod]
if sel_rt   != "All": sub = sub[sub["RATE_TYPE"] == sel_rt]

if sub.empty:
    st.warning("No matching lane/product/rate-type combination found.")
    st.stop()

row = sub.iloc[0]

# ── Current state display ─────────────────────────────────────────────────────

st.divider()
st.subheader("Current State")

r1 = st.columns(5)
r1[0].metric("Quotes",         f"{int(row['QUOTE_COUNT']):,}")
r1[1].metric("Shipments",      f"{int(row['SHIPMENT_COUNT']):,}")
r1[2].metric("Actual Win Rate",format_pct(row["ACTUAL_WIN_RATE"]))
r1[3].metric("Predicted WR",   format_pct(row.get("PREDICTED_WIN_RATE", row["ACTUAL_WIN_RATE"])))
r1[4].metric("GRATE/Shipment", format_currency(row["GRATE_PER_SHIPMENT"]))

# ── Scenario controls ─────────────────────────────────────────────────────────

st.divider()
st.subheader("Pricing Scenario")

sc1, sc2, sc3 = st.columns(3)
with sc1:
    price_change = st.slider(
        "Base price change (%)", -30, 20, -5, step=1,
        help="Negative = price reduction"
    ) / 100
with sc2:
    disc_change = st.slider(
        "Additional discount (%)", 0, 25, 0, step=1,
        help="Additional discount on top of current pricing"
    ) / 100
with sc3:
    elasticity = st.slider(
        "Price elasticity",
        0.5, 3.0,
        float(THRESHOLDS["price_elasticity"]),
        step=0.1,
        help="How sensitive win probability is to price changes. Higher = more elastic.",
    )

sc4, sc5, sc6 = st.columns(3)
with sc4:
    max_sac = st.slider(
        "Max GRATE sacrifice (%)", 5, 50,
        int(THRESHOLDS["max_grate_sacrifice_pct"] * 100), step=5,
        help="Maximum % of GRATE/shipment we're willing to give up"
    ) / 100
with sc5:
    min_sl = st.slider(
        "Min shipment lift required", 0.0, 20.0,
        float(THRESHOLDS["min_shipment_lift"]), step=0.5,
    )
with sc6:
    st.markdown("**Total price change**")
    total_pc = price_change + disc_change
    color = "green" if total_pc < 0 else "red"
    st.markdown(f"<h2 style='color:{color}'>{total_pc:.0%}</h2>", unsafe_allow_html=True)

# ── Run scenario ──────────────────────────────────────────────────────────────

result = run_scenario(
    row.to_dict(),
    price_change_pct=price_change,
    discount_change_pct=disc_change,
    max_grate_sacrifice_pct=max_sac,
    min_shipment_lift=min_sl,
    elasticity=elasticity,
)

# ── Results display ───────────────────────────────────────────────────────────

st.divider()
st.subheader("Scenario Results")

rec = result["recommendation"]
rec_colors = {
    "Good opportunity":                "success",
    "Too risky – negative GRATE lane": "error",
    "GRATE risk too high":             "error",
    "Not enough shipment lift":        "warning",
    "Expected GRATE lift is negative": "warning",
    "Insufficient data":               "info",
    "Needs pricing review":            "info",
}
badge = rec_colors.get(rec, "info")
getattr(st, badge)(f"**Recommendation: {rec}**")

res_cols = st.columns(4)
res_cols[0].metric("Current Win Prob",  format_pct(result["current_win_prob"]))
res_cols[1].metric("Scenario Win Prob", format_pct(result["scenario_win_prob"]),
                   delta=f"{result['scenario_win_prob'] - result['current_win_prob']:+.1%}")
res_cols[2].metric("Current Exp. Ships", f"{result['current_expected_shipments']:.1f}")
res_cols[3].metric("Scenario Exp. Ships",f"{result['scenario_expected_shipments']:.1f}",
                   delta=f"{result['shipment_lift']:+.1f}")

res_cols2 = st.columns(4)
res_cols2[0].metric("Current Exp. GRATE",  format_currency(result["current_expected_grate"]))
res_cols2[1].metric("Scenario Exp. GRATE", format_currency(result["scenario_expected_grate"]),
                    delta=f"{format_currency(result['grate_lift'])}")
res_cols2[2].metric("GRATE Sacrifice/Ship",format_currency(result["grate_sacrifice_per_shipment"]))
res_cols2[3].metric("Expected GRATE Lift", format_currency(result["grate_lift"]))

# ── Sensitivity chart ─────────────────────────────────────────────────────────

st.divider()
st.subheader("Price Sensitivity Analysis")
st.markdown("How expected GRATE changes across a range of price changes.")

price_range = [p / 100 for p in range(-30, 21, 2)]
sens_data   = []
for pc in price_range:
    r = run_scenario(row.to_dict(), price_change_pct=pc, discount_change_pct=0,
                     max_grate_sacrifice_pct=max_sac, min_shipment_lift=min_sl,
                     elasticity=elasticity)
    sens_data.append({
        "Price Change (%)":  pc,
        "Expected GRATE":    r["scenario_expected_grate"],
        "Expected Shipments":r["scenario_expected_shipments"],
        "Win Probability":   r["scenario_win_prob"],
    })
sens_df = pd.DataFrame(sens_data)

fig = go.Figure()
fig.add_scatter(x=sens_df["Price Change (%)"], y=sens_df["Expected GRATE"],
                mode="lines+markers", name="Expected GRATE", line=dict(color="#4C78A8"))
fig.add_vline(x=price_change, line_dash="dash", line_color="red",
              annotation_text="Current scenario")
fig.update_layout(
    template="plotly_white", title="Expected GRATE vs Price Change",
    xaxis_title="Price Change (%)", yaxis_title="Expected GRATE ($)",
    xaxis_tickformat=".0%",
)
st.plotly_chart(fig, use_container_width=True)

fig2 = go.Figure()
fig2.add_scatter(x=sens_df["Price Change (%)"], y=sens_df["Win Probability"],
                 mode="lines+markers", name="Win Probability", line=dict(color="#72B7B2"))
fig2.update_layout(
    template="plotly_white", title="Win Probability vs Price Change",
    xaxis_title="Price Change (%)", yaxis_title="Win Probability",
    xaxis_tickformat=".0%", yaxis_tickformat=".0%",
)
st.plotly_chart(fig2, use_container_width=True)
