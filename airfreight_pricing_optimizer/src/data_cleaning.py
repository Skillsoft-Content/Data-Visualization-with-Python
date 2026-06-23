import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "QUOTE_GROUP_ID", "QUOTE_DATE", "ORIG_COUNTRY", "DEST_COUNTRY",
    "PRODUCT", "QUOTE_WGT_KG", "NET_QUOTE_PRICE_USD", "WON",
]

OPTIONAL_DEFAULTS = {
    "ORIG_SVC": "UNKNOWN", "DEST_SVC": "UNKNOWN",
    "ORIG_REGION": "UNKNOWN", "DEST_REGION": "UNKNOWN",
    "RATE_TYPE": "UNKNOWN", "DISCOUNT_PCT": 0.0, "DISCOUNT_AMT_USD": 0.0,
    "DISCOUNT_FLAG": "N", "DISCOUNT_TYPE": "UNKNOWN",
    "DISCOUNT_CLASSIFICATION": "UNKNOWN", "CUSTOMER_NAME": "UNKNOWN",
    "LASSO_ID": "UNKNOWN", "REVENUE": 0.0, "GRATE": 0.0, "FFS_MARGIN": 0.0,
    "SHIPMENT_COUNT": 0, "PA_RATE": np.nan, "AI_QUOTE_INDICATOR": "N",
    "API_QUOTE_INDICATOR": "N", "CUSTOMER_CLASSIFICATION": "UNKNOWN",
    "SERVICE_TYPE": "UNKNOWN", "MOVEMENT_TYPE": "UNKNOWN",
    "INCOTERM": "UNKNOWN", "PAYOR_TYPE": "UNKNOWN", "CURRENCY": "USD",
    "USER_EMAIL": "UNKNOWN", "PRIMARY_PRODUCT_FLAG": "Y",
    "QUOTE_TYPE": "UNKNOWN", "QUOTE_STATUS": "UNKNOWN",
}


def validate_required_columns(df):
    return [f"Required column missing: {c}" for c in REQUIRED_COLUMNS if c not in df.columns]


def ensure_won_column(df):
    if "WON" not in df.columns:
        if "SHIPMENT_COUNT" in df.columns:
            df["WON"] = (df["SHIPMENT_COUNT"] > 0).astype(int)
        elif "SHIPMENT_ID" in df.columns:
            df["WON"] = df["SHIPMENT_ID"].notna().astype(int)
        else:
            df["WON"] = 0
    else:
        df["WON"] = pd.to_numeric(df["WON"], errors="coerce").fillna(0).astype(int)
    return df


def parse_dates(df):
    for col in ["QUOTE_DATE", "QUOTE_REQUEST_DATE", "WK_ENDING_DATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def fill_optional_columns(df):
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        elif default is not np.nan:
            df[col] = df[col].fillna(default)
    return df


def derive_year_month(df):
    if "QUOTE_YEAR" not in df.columns and "QUOTE_DATE" in df.columns:
        df["QUOTE_YEAR"] = df["QUOTE_DATE"].dt.year
    if "QUOTE_MONTH" not in df.columns and "QUOTE_DATE" in df.columns:
        df["QUOTE_MONTH"] = df["QUOTE_DATE"].dt.month
    return df


def clean_string_columns(df):
    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col].astype(str).str.strip().str.upper()
            .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
            .fillna("UNKNOWN")
        )
    return df


def flag_excluded_rows(df):
    df["EXCLUDE_FLAG"]   = False
    df["EXCLUDE_REASON"] = ""

    def _flag(mask, reason):
        new = mask & ~df["EXCLUDE_FLAG"]
        df.loc[new, "EXCLUDE_FLAG"]   = True
        df.loc[new, "EXCLUDE_REASON"] = reason

    _flag(df["QUOTE_WGT_KG"].isna()        | (df["QUOTE_WGT_KG"] <= 0),          "Zero or missing weight")
    _flag(df["NET_QUOTE_PRICE_USD"].isna()  | (df["NET_QUOTE_PRICE_USD"] <= 0),   "Zero or missing price")
    _flag(df["QUOTE_DATE"].isna(),                                                  "Missing quote date")
    _flag(df["ORIG_COUNTRY"].isna()         | df["DEST_COUNTRY"].isna(),           "Missing origin or destination")
    return df


def clean_data(df):
    df = df.copy()
    df = ensure_won_column(df)
    df = parse_dates(df)
    errors = validate_required_columns(df)
    if errors:
        return df, errors, pd.DataFrame()
    df = fill_optional_columns(df)
    df = derive_year_month(df)
    df = clean_string_columns(df)
    df = flag_excluded_rows(df)
    excluded = df[df["EXCLUDE_FLAG"]].copy()
    clean    = df[~df["EXCLUDE_FLAG"]].copy().reset_index(drop=True)
    return clean, [], excluded
