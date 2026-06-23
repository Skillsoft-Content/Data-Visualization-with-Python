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
st.markdown("Download lane-level recommendations, quote-level predictions, and model artefacts.")


# ── Data helpers ──────────────────────────────────────────────────────────────

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


def _get_lane_summary(df_pred: pd.DataFrame) -> pd.DataFrame:
    if "lane_summary" not in st.session_state:
        ls = compute_lane_summary(df_pred)
        ls = apply_recommendations(ls)
        st.session_state["lane_summary"] = ls
    return st.session_state["lane_summary"]


df_pred = _get_pred_df()
lane_df = _get_lane_summary(df_pred)

ts = datetime.now().strftime("%Y%m%d_%H%M")


# ── Helper to render a download row ──────────────────────────────────────────

def _download_row(label: str, description: str, df: pd.DataFrame, filename_stem: str):
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(f"**{label}**  \n{description} — {len(df):,} rows")
    with col2:
        st.download_button(
            "CSV",
            data=to_csv_bytes(df),
            file_name=f"{filename_stem}_{ts}.csv",
            mime="text/csv",
            key=f"csv_{filename_stem}",
        )
    with col3:
        st.download_button(
            "Excel",
            data=to_excel_bytes({label[:31]: df}),
            file_name=f"{filename_stem}_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"xlsx_{filename_stem}",
        )


# ── Export sections ───────────────────────────────────────────────────────────

st.subheader("Lane-Level Exports")
_download_row(
    "All Lanes",
    "Full lane summary with recommendations and scenario GRATE lifts",
    lane_df,
    "lane_all",
)

opp_df = lane_df[lane_df["RECOMMENDATION"] == "Lower price selectively"].copy()
_download_row(
    "Opportunity Lanes",
    "Only lanes recommended for selective price reduction",
    opp_df,
    "lane_opportunities",
)

review_df = lane_df[lane_df["RECOMMENDATION"] == "Needs pricing review"].copy()
_download_row(
    "Manual Review Lanes",
    "Lanes flagged for manual pricing review",
    review_df,
    "lane_review",
)

st.divider()
st.subheader("Quote-Level Exports")

quote_export_cols = [
    "QUOTE_GROUP_ID", "DIGITAL_QUOTE_ID", "DIGITAL_CONFIRMATION_ID",
    "QUOTE_DATE", "CUSTOMER_NAME", "LASSO_ID",
    "ORIG_COUNTRY", "DEST_COUNTRY", "ORIG_SVC", "DEST_SVC",
    "PRODUCT", "RATE_TYPE", "QUOTE_SOURCE",
    "QUOTE_WGT_KG", "NET_QUOTE_PRICE_USD", "PRICE_PER_KG",
    "DISCOUNT_PCT", "DISCOUNT_AMT_USD",
    "WIN_PROBABILITY", "CONFIDENCE_LEVEL", "CONFIDENCE_SCORE",
    "WON", "REVENUE", "GRATE", "LANE",
]
q_export = df_pred[[c for c in quote_export_cols if c in df_pred.columns]].copy()
_download_row(
    "Quote Predictions",
    "All quotes with predicted win probability and confidence",
    q_export,
    "quote_predictions",
)

won_q = q_export[q_export["WON"] == 1].copy()
_download_row(
    "Won Quotes",
    "Converted quotes only (WON = 1) with GRATE and revenue",
    won_q,
    "won_quotes",
)

st.divider()
st.subheader("Full Data Export")

all_sheets = {
    "Lane Recommendations": lane_df,
    "Quote Predictions":    q_export,
    "Opportunities":        opp_df,
    "Review Lanes":         review_df,
}

excel_bytes = to_excel_bytes(all_sheets)
st.download_button(
    "Download All Sheets (Excel)",
    data=excel_bytes,
    file_name=f"airfreight_full_export_{ts}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary",
)

st.divider()
st.subheader("Excluded / Flagged Rows")

excl_df = st.session_state.get("excluded_df", pd.DataFrame())
if not excl_df.empty:
    _download_row(
        "Excluded Rows",
        "Rows removed during data cleaning with reason",
        excl_df,
        "excluded_rows",
    )
    with st.expander(f"Preview excluded rows ({len(excl_df):,})"):
        st.dataframe(excl_df.head(200), use_container_width=True)
else:
    st.info("No excluded rows recorded.")

st.divider()
st.subheader("Model Metrics")

metrics = st.session_state.get("model_metrics")
if metrics:
    importable_metrics = {
        k: v for k, v in metrics.items()
        if k not in ("confusion_matrix", "roc_curve", "calibration", "lift_chart", "feature_importance")
    }
    metrics_df = pd.DataFrame([importable_metrics])
    _download_row(
        "Model Metrics",
        "ROC AUC, Precision, Recall, F1, Brier Score and metadata",
        metrics_df,
        "model_metrics",
    )

    if "feature_importance" in metrics:
        fi_df = metrics["feature_importance"]
        _download_row(
            "Feature Importance",
            "Model feature importance scores ranked by importance",
            fi_df,
            "feature_importance",
        )
else:
    st.info("Train a model on the Executive Summary page to enable model metric exports.")
