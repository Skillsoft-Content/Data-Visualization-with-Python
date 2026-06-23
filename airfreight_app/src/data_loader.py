"""
Data loading layer.
Supports two sources:
  - 'csv'  : reads quote_data.csv (or a user-supplied path)
  - 'sql'  : connects to SQL Server and joins quotes + bookings

The SQL query aggregates bookings per quote confirmation so that the result is
always one row per quote group (no fan-out from multi-shipment bookings).
"""

import urllib.parse
import pandas as pd
from src.config import DB_CONFIG, QUOTE_TABLE, BOOKING_TABLE, DATA_PATH


# ── Database connection ───────────────────────────────────────────────────────

def get_db_engine():
    from sqlalchemy import create_engine
    server   = DB_CONFIG["server"]
    database = DB_CONFIG["database"]
    driver   = DB_CONFIG["driver"]
    params   = urllib.parse.quote_plus(
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"Trusted_Connection=yes;"
    )
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}", echo=False)


# ── SQL query builder ─────────────────────────────────────────────────────────

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
        -- Quote identifiers
        q.QUOTE_GROUP_ID,
        q.QUOTE_OPP_ID,
        q.DIGITAL_CONFIRMATION_ID,
        q.DIGITAL_QUOTE_ID,

        -- Dates
        q.QUOTE_DATE,
        q.QUOTE_REQUEST_DATE,
        q.QUOTE_YEAR,
        q.QUOTE_MONTH,
        q.WK_ENDING_DATE,

        -- Quote metadata
        q.QUOTE_TYPE,
        q.QUOTE_STATUS,

        -- Customer
        q.LASSO_ID,
        q.USER_EMAIL,
        q.CUSTOMER_NAME,

        -- Geography
        q.ORIG_REGION,
        q.ORIG_COUNTRY,
        q.ORIG_SVC,
        q.DEST_REGION,
        q.DEST_COUNTRY,
        q.DEST_SVC,

        -- Product / service
        q.PRODUCT,
        q.PRIMARY_PRODUCT_FLAG,
        q.SERVICE_TYPE,
        q.MOVEMENT_TYPE,
        q.INCOTERM,

        -- Weight
        q.FINAL_QUOTE_WGT  AS QUOTE_WGT_KG,
        q.UOM,

        -- Pricing
        q.NET_QUOTE_PRICE_USD,
        q.ORIG_QUOTE_PRICE_USD,
        q.SHELL_PA          AS PA_RATE,
        q.RATE_TYPE,
        q.CURRENCY,

        -- Discounts
        q.DISCOUNT_FLAG,
        q.DISCOUNT_CLASSIFICATION,
        q.DISCOUNT_TYPE,
        q.OFFERED_DISCOUNT_FACTOR     AS DISCOUNT_PCT,
        q.OFFERED_DISCOUNT_AMT_USD    AS DISCOUNT_AMT_USD,

        -- Quote source flags
        q.HEAVY_SHOPPER_FLAG,
        q.AI_QUOTE_INDICATOR,
        q.API_QUOTE_INDICATOR,
        q.PAYOR_TYPE,

        -- Booking aggregates (NULL for lost quotes)
        COALESCE(b.SHIPMENT_COUNT, 0)  AS SHIPMENT_COUNT,
        COALESCE(b.TOTAL_TEP_REV, 0)   AS REVENUE,
        COALESCE(b.TOTAL_GRATE,   0)   AS GRATE,
        b.AVG_SHIP_KG                  AS SHIP_WGT_KG,
        b.CUSTOMER_CLASSIFICATION,
        COALESCE(b.FFS_MARGIN, 0)      AS FFS_MARGIN,

        -- Target: 1 = WON (converted to shipment), 0 = LOST
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
        GROUP BY
            DIGITAL_PRQ_NBR,
            DIGITAL_BOOKING_ACCTG_YEAR
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
        df = pd.read_sql(text(query), conn)
    return df


def load_from_csv(filepath=None) -> pd.DataFrame:
    path = filepath or DATA_PATH
    df = pd.read_csv(path, low_memory=False)
    # Normalize to upper-case column names for consistent downstream handling
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
    return df


def load_data(source: str = "csv", filepath=None, date_from=None, date_to=None, limit=None) -> pd.DataFrame:
    """
    source: 'csv' | 'sql'
    """
    if source == "sql":
        return load_from_sql(date_from=date_from, date_to=date_to, limit=limit)
    return load_from_csv(filepath=filepath)
