"""
Data loading layer.
Supports two sources:
  - 'csv' : reads quote_data.csv (or a user-supplied path)
  - 'sql' : connects to SQL Server and joins quotes + bookings

SQL failures are caught early and translated into human-readable messages
via db_utils so the app always falls back cleanly to demo data.
"""

import urllib.parse
import pandas as pd
from src.config import DB_CONFIG, QUOTE_TABLE, BOOKING_TABLE, DATA_PATH


# ── pyodbc guard ──────────────────────────────────────────────────────────────

def _import_pyodbc():
    """Import pyodbc and raise RuntimeError with a helpful message on failure."""
    try:
        import pyodbc
        return pyodbc
    except ImportError as e:
        raise RuntimeError(
            "pyodbc is not installed. Run: pip install pyodbc"
        ) from e
    except (PermissionError, OSError) as e:
        from src.db_utils import interpret_connection_error
        raise RuntimeError(interpret_connection_error(e)) from e
    except Exception as e:
        from src.db_utils import interpret_connection_error
        raise RuntimeError(interpret_connection_error(e)) from e


# ── Engine ────────────────────────────────────────────────────────────────────

def get_db_engine():
    _import_pyodbc()  # validate access before sqlalchemy tries to use pyodbc
    try:
        from sqlalchemy import create_engine
        params = urllib.parse.quote_plus(
            f"DRIVER={{{DB_CONFIG['driver']}}};"
            f"SERVER={DB_CONFIG['server']};"
            f"DATABASE={DB_CONFIG['database']};"
            f"Trusted_Connection=yes;"
        )
        return create_engine(f"mssql+pyodbc:///?odbc_connect={params}", echo=False)
    except RuntimeError:
        raise
    except Exception as e:
        from src.db_utils import interpret_connection_error
        raise RuntimeError(interpret_connection_error(e)) from e


def test_connection() -> tuple:
    """
    Quick connection test. Returns (success: bool, message: str).
    Safe to call from the UI without crashing the app.
    """
    try:
        from sqlalchemy import text
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connection successful."
    except RuntimeError as e:
        return False, str(e)
    except Exception as e:
        from src.db_utils import interpret_connection_error
        return False, interpret_connection_error(e)


# ── SQL query ─────────────────────────────────────────────────────────────────

def _build_query(date_from=None, date_to=None, limit=None):
    where_parts = ["q.HEAVY_SHOPPER_FLAG = 'N'"]
    if date_from:
        where_parts.append(f"q.QUOTE_DATE >= '{date_from}'")
    if date_to:
        where_parts.append(f"q.QUOTE_DATE <= '{date_to}'")

    where_clause = "WHERE " + " AND ".join(where_parts)
    top_clause   = f"TOP ({limit})" if limit else ""

    return f"""
    SELECT {top_clause}
        q.QUOTE_GROUP_ID,
        q.QUOTE_OPP_ID,
        q.DIGITAL_CONFIRMATION_ID,
        q.DIGITAL_QUOTE_ID,
        q.QUOTE_DATE,
        q.QUOTE_REQUEST_DATE,
        q.QUOTE_YEAR,
        q.QUOTE_MONTH,
        q.WK_ENDING_DATE,
        q.QUOTE_TYPE,
        q.QUOTE_STATUS,
        q.LASSO_ID,
        q.USER_EMAIL,
        q.CUSTOMER_NAME,
        q.ORIG_REGION,
        q.ORIG_COUNTRY,
        q.ORIG_SVC,
        q.DEST_REGION,
        q.DEST_COUNTRY,
        q.DEST_SVC,
        q.PRODUCT,
        q.PRIMARY_PRODUCT_FLAG,
        q.SERVICE_TYPE,
        q.MOVEMENT_TYPE,
        q.INCOTERM,
        q.FINAL_QUOTE_WGT         AS QUOTE_WGT_KG,
        q.UOM,
        q.NET_QUOTE_PRICE_USD,
        q.ORIG_QUOTE_PRICE_USD,
        q.SHELL_PA                AS PA_RATE,
        q.RATE_TYPE,
        q.CURRENCY,
        q.DISCOUNT_FLAG,
        q.DISCOUNT_CLASSIFICATION,
        q.DISCOUNT_TYPE,
        q.OFFERED_DISCOUNT_FACTOR    AS DISCOUNT_PCT,
        q.OFFERED_DISCOUNT_AMT_USD   AS DISCOUNT_AMT_USD,
        q.HEAVY_SHOPPER_FLAG,
        q.AI_QUOTE_INDICATOR,
        q.API_QUOTE_INDICATOR,
        q.PAYOR_TYPE,
        COALESCE(b.SHIPMENT_COUNT, 0) AS SHIPMENT_COUNT,
        COALESCE(b.TOTAL_TEP_REV, 0)  AS REVENUE,
        COALESCE(b.TOTAL_GRATE,   0)  AS GRATE,
        b.AVG_SHIP_KG                 AS SHIP_WGT_KG,
        b.CUSTOMER_CLASSIFICATION,
        COALESCE(b.FFS_MARGIN,    0)  AS FFS_MARGIN,
        CASE
            WHEN COALESCE(b.SHIPMENT_COUNT, 0) > 0 THEN 1
            ELSE 0
        END AS WON
    FROM {QUOTE_TABLE} q
    LEFT JOIN (
        SELECT
            DIGITAL_PRQ_NBR,
            DIGITAL_BOOKING_ACCTG_YEAR,
            COUNT(DISTINCT SHIPMENT_DIM_FK)      AS SHIPMENT_COUNT,
            SUM(TEP_REV)                          AS TOTAL_TEP_REV,
            SUM(GRATE)                            AS TOTAL_GRATE,
            AVG(DIGITAL_SHIP_WGT_KG)             AS AVG_SHIP_KG,
            MAX(DIGITAL_CUSTOMER_CLASSIFICATION)  AS CUSTOMER_CLASSIFICATION,
            SUM(FFS_MARGIN)                       AS FFS_MARGIN
        FROM {BOOKING_TABLE}
        GROUP BY DIGITAL_PRQ_NBR, DIGITAL_BOOKING_ACCTG_YEAR
    ) b
        ON  q.DIGITAL_CONFIRMATION_ID = b.DIGITAL_PRQ_NBR
        AND q.QUOTE_YEAR              = b.DIGITAL_BOOKING_ACCTG_YEAR
    {where_clause}
    """


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_from_sql(date_from=None, date_to=None, limit=None) -> pd.DataFrame:
    from sqlalchemy import text
    engine = get_db_engine()
    query  = _build_query(date_from=date_from, date_to=date_to, limit=limit)
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def load_from_csv(filepath=None) -> pd.DataFrame:
    path = filepath or DATA_PATH
    df   = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
    return df


def load_data(source: str = "csv", filepath=None,
             date_from=None, date_to=None, limit=None) -> pd.DataFrame:
    if source == "sql":
        return load_from_sql(date_from=date_from, date_to=date_to, limit=limit)
    return load_from_csv(filepath=filepath)
