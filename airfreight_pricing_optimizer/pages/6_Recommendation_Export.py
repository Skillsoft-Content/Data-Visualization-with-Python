import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
import pandas as pd
import streamlit as st

from src.utils import get_sample_data
from src.data_cleaning import clean_data
from src.feature_engineering import run_feature_engineering
from src.predict import predict_win_probability, add_confidence_level, add_fallback_win_probability, model_exists
from src.recommendation_engine import compute_lane_summary, apply_recommendations
from src.export_utils import to_csv_bytes, to_excel_bytes

st.set_page_config(page_title="Recommendation Export", layout="wide")
st.title("Recommendation Export")


def _get_pred_df():
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


def _get_lane_summary(df_pred):
    if "lane_summary" not in st.session_state:
        ls = compute_lane_summary(df_pred)
        ls = apply_recommendations(ls)
        st.session_state["lane_summary"] = ls
    return st.session_state["lane_summary"]


df_pred = _get_pred_df()
lane_df = _get_lane_summary(df_pred)
ts      = datetime.now().strftime("%Y%m%d_%H%M")


def _row(label, desc, df, stem):
    c1,c2,c3 = st.columns([3,1,1])
    with c1: st.markdown(f"**{label}**  \n{desc} — {len(df):,} rows")
    with c2:
        st.download_button("CSV",   to_csv_bytes(df),
                           file_name=f"{stem}_{ts}.csv",   mime="text/csv",         key=f"csv_{stem}")
    with c3:
        st.download_button("Excel", to_excel_bytes({label[:31]: df}),
                           file_name=f"{stem}_{ts}.xlsx",  mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"xlsx_{stem}")


st.subheader("Lane-Level Exports")
_row("All Lanes",         "Full lane summary with recommendations", lane_df, "lane_all")
_row("Opportunity Lanes", "Lanes recommended for selective price reduction",
     lane_df[lane_df["RECOMMENDATION"]=="Lower price selectively"], "lane_opportunities")
_row("Review Lanes",      "Lanes flagged for manual pricing review",
     lane_df[lane_df["RECOMMENDATION"]=="Needs pricing review"], "lane_review")

st.divider()
st.subheader("Quote-Level Exports")
qcols = ["QUOTE_GROUP_ID","DIGITAL_QUOTE_ID","QUOTE_DATE","CUSTOMER_NAME","LASSO_ID",
         "ORIG_COUNTRY","DEST_COUNTRY","ORIG_SVC","DEST_SVC","PRODUCT","RATE_TYPE",
         "QUOTE_SOURCE","QUOTE_WGT_KG","NET_QUOTE_PRICE_USD","PRICE_PER_KG",
         "DISCOUNT_PCT","WIN_PROBABILITY","CONFIDENCE_LEVEL","WON","REVENUE","GRATE","LANE"]
q_export = df_pred[[c for c in qcols if c in df_pred.columns]].copy()
_row("Quote Predictions", "All quotes with predicted win probability", q_export, "quote_predictions")
_row("Won Quotes",        "Converted quotes with GRATE and revenue",
     q_export[q_export["WON"]==1], "won_quotes")

st.divider()
st.subheader("Full Export (All Sheets)")
st.download_button(
    "Download Everything (Excel)",
    data=to_excel_bytes({
        "Lane Recommendations": lane_df,
        "Quote Predictions":    q_export,
        "Opportunities":        lane_df[lane_df["RECOMMENDATION"]=="Lower price selectively"],
        "Review Lanes":         lane_df[lane_df["RECOMMENDATION"]=="Needs pricing review"],
    }),
    file_name=f"airfreight_full_export_{ts}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary",
)

st.divider()
st.subheader("Excluded / Flagged Rows")
excl_df = st.session_state.get("excluded_df", pd.DataFrame())
if not excl_df.empty:
    _row("Excluded Rows", "Rows removed during cleaning with reason", excl_df, "excluded_rows")
else:
    st.info("No excluded rows.")

metrics = st.session_state.get("model_metrics")
if metrics:
    st.divider()
    st.subheader("Model Metrics")
    importable = {k:v for k,v in metrics.items()
                  if k not in ("confusion_matrix","roc_curve","calibration","lift_chart","feature_importance")}
    _row("Model Metrics", "ROC AUC, F1, Brier Score and metadata",
         pd.DataFrame([importable]), "model_metrics")
    if "feature_importance" in metrics:
        _row("Feature Importance", "Ranked feature importance scores",
             metrics["feature_importance"], "feature_importance")
