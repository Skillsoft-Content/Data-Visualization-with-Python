import pandas as pd
import numpy as np


def add_lane_features(df):
    df["LANE"]         = df["ORIG_COUNTRY"].str.strip() + " - " + df["DEST_COUNTRY"].str.strip()
    df["SERVICE_PAIR"] = df["ORIG_SVC"].str.strip()     + " - " + df["DEST_SVC"].str.strip()
    df["REGION_PAIR"]  = df["ORIG_REGION"].str.strip()  + " - " + df["DEST_REGION"].str.strip()
    return df


def add_price_features(df):
    kg = df["QUOTE_WGT_KG"].replace(0, np.nan)
    df["PRICE_PER_KG"]   = df["NET_QUOTE_PRICE_USD"] / kg
    df["REVENUE_PER_KG"] = df["REVENUE"] / kg.where(df["WON"] == 1)
    df["GRATE_PER_KG"]   = df["GRATE"]   / kg.where(df["WON"] == 1)
    disc = df["DISCOUNT_PCT"].fillna(0)
    df["DISCOUNT_BUCKET"] = pd.cut(
        disc,
        bins=[-np.inf, 0.001, 0.05, 0.10, 0.20, np.inf],
        labels=["NO_DISCOUNT", "1-5%", "5-10%", "10-20%", "20%+"],
    ).astype(str)
    pa = df["PA_RATE"].where(df["PA_RATE"] > 0)
    df["PRICE_VS_PA"]   = df["NET_QUOTE_PRICE_USD"] / pa
    df["ABOVE_PA_FLAG"] = (df["PRICE_VS_PA"] >= 1.0).astype(float)
    return df


def add_time_features(df):
    dt = pd.to_datetime(df["QUOTE_DATE"], errors="coerce")
    df["QUOTE_QUARTER"]     = dt.dt.quarter
    df["QUOTE_DAY_OF_WEEK"] = dt.dt.dayofweek
    df["QUOTE_YEAR_MONTH"]  = dt.dt.to_period("M").astype(str)
    return df


def add_quote_source(df):
    ai  = df["AI_QUOTE_INDICATOR"].astype(str).str.upper()
    api = df["API_QUOTE_INDICATOR"].astype(str).str.upper()
    df["QUOTE_SOURCE"] = np.select([ai == "Y", api == "Y"], ["IQA", "API"], default="MANUAL")
    return df


def add_historical_features(df, reference_df):
    global_wr = reference_df["WON"].mean()

    def _merge(stats, on):
        return df.merge(stats, on=on, how="left")

    lane_stats = (
        reference_df.groupby("LANE")
        .agg(LANE_WIN_RATE=("WON","mean"), LANE_QUOTE_COUNT=("QUOTE_GROUP_ID","nunique"),
             LANE_SHIPMENT_COUNT=("WON","sum"), LANE_AVG_GRATE=("GRATE","mean"),
             LANE_AVG_REVENUE=("REVENUE","mean"), LANE_AVG_KG=("QUOTE_WGT_KG","mean"))
        .reset_index()
    )
    df = _merge(lane_stats, "LANE")

    cust_stats = (
        reference_df.groupby("LASSO_ID")
        .agg(CUST_WIN_RATE=("WON","mean"), CUST_QUOTE_COUNT=("QUOTE_GROUP_ID","nunique"))
        .reset_index()
    )
    df = _merge(cust_stats, "LASSO_ID")

    cl_stats = (
        reference_df.groupby(["LASSO_ID","LANE"])
        .agg(CUST_LANE_WIN_RATE=("WON","mean")).reset_index()
    )
    df = _merge(cl_stats, ["LASSO_ID","LANE"])

    rt_stats   = reference_df.groupby("RATE_TYPE").agg(RATE_TYPE_WIN_RATE=("WON","mean")).reset_index()
    prod_stats = reference_df.groupby("PRODUCT").agg(PRODUCT_WIN_RATE=("WON","mean")).reset_index()
    df = _merge(rt_stats, "RATE_TYPE")
    df = _merge(prod_stats, "PRODUCT")

    for col in df.columns:
        if col.endswith("_WIN_RATE"):
            df[col] = df[col].fillna(global_wr)
        elif "_COUNT" in col or "_SHIPMENT" in col:
            df[col] = df[col].fillna(0)
        elif col.startswith("LANE_AVG_") or col.startswith("CUST_"):
            df[col] = df[col].fillna(0)
    return df


def add_customer_type(df):
    cnt = df.get("CUST_QUOTE_COUNT", pd.Series(0, index=df.index)).fillna(0)
    df["CUSTOMER_TYPE"] = pd.cut(
        cnt, bins=[-np.inf, 0, 5, 20, np.inf],
        labels=["NEW","EMERGING","ESTABLISHED","LOYAL"]
    ).astype(str)
    return df


def add_relative_price_features(df):
    lane_avg = df.get("LANE_AVG_REVENUE", pd.Series(np.nan, index=df.index))
    df["PRICE_VS_LANE_AVG"]        = df["NET_QUOTE_PRICE_USD"] / lane_avg.where(lane_avg > 0)
    df["ABOVE_LANE_AVG_PRICE_FLAG"]= (df["PRICE_VS_LANE_AVG"] >= 1.0).astype(float)
    return df


def run_feature_engineering(df, reference_df=None):
    if reference_df is None:
        reference_df = df
    df = add_lane_features(df)
    df = add_price_features(df)
    df = add_time_features(df)
    df = add_quote_source(df)
    df = add_historical_features(df, reference_df)
    df = add_customer_type(df)
    df = add_relative_price_features(df)
    return df
