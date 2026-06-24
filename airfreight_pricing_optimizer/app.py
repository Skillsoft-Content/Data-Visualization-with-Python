"""
Airfreight Win Rate & Pricing Optimization -- main entry point.
Run with:  streamlit run app.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import streamlit as st

from src.utils import get_sample_data, format_currency, format_pct
from src.data_loader import load_data, test_connection
from src.data_cleaning import clean_data
from src.db_utils import check_environment

st.set_page_config(
    page_title="Airfreight Win Rate Optimizer",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar: data source selector ────────────────────────────────────────────

st.sidebar.title("✈ Airfreight Win Rate")
st.sidebar.markdown("**Pricing Optimization Tool**")
st.sidebar.divider()

data_source = st.sidebar.radio(
    "Data source",
    ["Demo data", "Upload CSV", "SQL Server"],
    index=0,
)

filepath   = None
date_from  = None
date_to    = None
row_limit  = None
source_key = "demo"

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload quote_data.csv", type=["csv"])
    if uploaded:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.getvalue())
            filepath = tmp.name
    source_key = "csv"

elif data_source == "SQL Server":
    date_from = st.sidebar.date_input("From date", value=pd.Timestamp("2024-01-01"))
    date_to   = st.sidebar.date_input("To date",   value=pd.Timestamp("2024-12-31"))
    row_limit = st.sidebar.number_input(
        "Row limit (0 = all)", min_value=0, value=100_000, step=10_000,
    )
    source_key = "sql"

    # ── Connection diagnostics panel ──────────────────────────────────────
    with st.sidebar.expander("Connection Diagnostics", expanded=True):
        env = check_environment()

        if env.get("is_store_python"):
            st.error("🚨 Microsoft Store Python detected")
            st.markdown(
                "**This will block SQL Server access.**\n\n"
                "Run `setup_windows.bat` for the one-time fix."
            )
        else:
            st.success(f"Python OK: `{env['python_executable'][:40]}`")

        pyodbc_status = env.get("pyodbc_status", "UNKNOWN")
        if pyodbc_status == "OK":
            st.success(f"pyodbc {env.get('pyodbc_version', '')} installed")
            if env.get("sql_server_driver_found"):
                st.success("ODBC Driver 17 for SQL Server: found")
            else:
                st.error("ODBC Driver 17 not found")
                st.markdown("Download: https://aka.ms/downloadmsodbcsql")
        elif pyodbc_status == "PERMISSION_ERROR":
            st.error("pyodbc: Access Denied")
            st.markdown("Run `setup_windows.bat` to fix the Windows alias.")
        elif pyodbc_status == "NOT_INSTALLED":
            st.error("pyodbc not installed")
            st.code("pip install pyodbc")
        else:
            st.warning(f"pyodbc: {pyodbc_status}")
            if "pyodbc_error" in env:
                st.caption(env["pyodbc_error"])

        if st.button("Test Connection"):
            with st.spinner("Testing..."):
                ok, msg = test_connection()
            if ok:
                st.success(msg)
            else:
                st.error("Connection failed")
                st.markdown(msg)


# ── Load data ────────────────────────────────────────────────────────────────

if st.sidebar.button("Load / Refresh Data", type="primary"):
    st.cache_data.clear()
    for key in ["df", "excluded_df", "df_fe", "df_pred", "lane_summary", "model_metrics"]:
        st.session_state.pop(key, None)


@st.cache_data(show_spinner="Loading data...")
def _load(source, path, dfrom, dto, limit):
    if source == "demo":
        raw = get_sample_data(5_000)
    elif source == "csv":
        raw = load_data(source="csv", filepath=path)
    else:
        raw = load_data(
            source="sql", date_from=str(dfrom), date_to=str(dto),
            limit=int(limit) if limit else None,
        )
    return clean_data(raw)


conn_error_msg = None
try:
    limit_val = row_limit if data_source == "SQL Server" else None
    df, errors, excluded = _load(source_key, filepath, date_from, date_to, limit_val)
except RuntimeError as e:
    conn_error_msg = str(e)
    df, errors, excluded = _load("demo", None, None, None, None)
except Exception as ex:
    conn_error_msg = f"Unexpected error: {ex}"
    df, errors, excluded = _load("demo", None, None, None, None)

if errors:
    for e in errors:
        st.sidebar.error(e)
else:
    st.session_state["df"]          = df
    st.session_state["excluded_df"] = excluded

# Sidebar stats
if "df" in st.session_state:
    d = st.session_state["df"]
    st.sidebar.divider()
    st.sidebar.markdown(f"**Quotes:** {d['QUOTE_GROUP_ID'].nunique():,}")
    st.sidebar.markdown(f"**Shipments:** {int(d['WON'].sum()):,}")
    st.sidebar.markdown(f"**Win rate:** {d['WON'].mean():.1%}")
    st.sidebar.markdown(f"**Excluded rows:** {len(st.session_state.get('excluded_df', [])):,}")
    if "model_metrics" in st.session_state:
        auc = st.session_state["model_metrics"].get("roc_auc", 0)
        st.sidebar.markdown(f"**Model AUC:** {auc:.3f}")

# ── Home page ────────────────────────────────────────────────────────────────

st.title("Airfreight Win Rate & Pricing Optimization")

if conn_error_msg:
    st.warning(
        "Could not connect to SQL Server. Running on **Demo data**.  \n"
        "See sidebar diagnostics for the fix."
    )
    with st.expander("Connection Error Details", expanded=True):
        st.markdown(conn_error_msg)

st.markdown(
    "Select a data source in the sidebar and click **Load / Refresh Data** to begin."
)
st.divider()

col1, col2, col3 = st.columns(3)
with col1: st.info("**1 · Executive Summary**\n\nKPIs, monthly trends, and top opportunity lanes.")
with col2: st.info("**2 · Lane Opportunity Dashboard**\n\nFilter and sort all lanes. Identify where to act.")
with col3: st.info("**3 · Drilldown**\n\nSelect a lane for monthly trends and quote-level detail.")

col4, col5, col6 = st.columns(3)
with col4: st.info("**4 · Scenario Simulator**\n\nModel the GRATE impact of price changes.")
with col5: st.info("**5 · Model Performance**\n\nROC AUC, calibration, lift, feature importance.")
with col6: st.info("**6 · Recommendation Export**\n\nDownload lane recommendations and predictions.")
