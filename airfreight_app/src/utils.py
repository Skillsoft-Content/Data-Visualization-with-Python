"""Shared helpers: formatting, KPI cards, and synthetic demo data generation."""

import pandas as pd
import numpy as np
import streamlit as st


# ── Formatting ────────────────────────────────────────────────────────────────

def format_currency(val, prefix="$"):
    if pd.isna(val):
        return "N/A"
    if abs(val) >= 1_000_000:
        return f"{prefix}{val / 1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"{prefix}{val / 1_000:.1f}K"
    return f"{prefix}{val:.0f}"


def format_pct(val):
    if pd.isna(val):
        return "N/A"
    return f"{val:.1%}"


def kpi_card(label: str, value: str, delta: str = None):
    st.metric(label=label, value=value, delta=delta)


# ── Demo data ─────────────────────────────────────────────────────────────────

def get_sample_data(n_rows: int = 5_000, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic airfreight quote dataset with realistic distributions.
    Used when no real data source is available.
    """
    rng = np.random.default_rng(seed)

    origins = ["US", "DE", "GB", "CN", "FR", "JP", "SG", "AU", "CA", "MX", "NL", "HK"]
    dests   = ["US", "DE", "GB", "CN", "FR", "JP", "SG", "AU", "CA", "MX", "NL", "HK"]
    svcs    = ["ONT", "2DA", "GFP", "WXS", "ESP"]
    regions = ["AMERICAS", "EUROPE", "ASIA PACIFIC", "MIDDLE EAST & AFRICA"]
    products     = ["WW EXPRESS", "WW EXPEDITED", "WW SAVER"]
    rate_types   = ["MARKET", "PA", "SPOT", "NEGOTIATED"]
    incoterms    = ["DDP", "DAP", "EXW", "FOB", "CIP"]
    payor_types  = ["SHIPPER", "CONSIGNEE", "THIRD_PARTY"]
    move_types   = ["EXPORT", "IMPORT"]
    disc_types   = ["FLAT", "PCT", "NONE"]
    cust_class   = ["STRATEGIC", "COMMERCIAL", "SME", "SPOT"]
    customers    = [f"CUST_{i:04d}" for i in range(250)]

    dates  = pd.date_range("2023-01-01", "2024-12-31", periods=n_rows)
    orig   = rng.choice(origins, n_rows)
    dest   = rng.choice(dests,   n_rows)
    same   = orig == dest
    dest[same] = rng.choice([d for d in dests if d != "US"], same.sum())

    weight    = np.abs(rng.lognormal(5.0, 1.2, n_rows)).clip(1, 5_000)
    base_px   = weight * rng.uniform(2.5, 9.0, n_rows)
    disc_pct  = rng.choice([0, 0, 0, 0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20], n_rows)
    net_px    = base_px * (1 - disc_pct)

    product   = rng.choice(products,   n_rows, p=[0.60, 0.25, 0.15])
    rate_type = rng.choice(rate_types, n_rows, p=[0.40, 0.25, 0.20, 0.15])
    customer  = rng.choice(customers,  n_rows)
    pa_rate   = base_px * rng.uniform(0.88, 1.12, n_rows)

    # Simulate win probability with realistic drivers
    win_base = (
        0.28
        + (rate_type == "PA")         * 0.18
        + (rate_type == "NEGOTIATED") * 0.12
        + (rate_type == "MARKET")     * 0.06
        + (disc_pct >= 0.05)          * 0.08
        + (disc_pct >= 0.10)          * 0.06
        + rng.normal(0, 0.10, n_rows)
    ).clip(0.03, 0.97)

    won      = rng.binomial(1, win_base).astype(int)
    grate    = np.where(won, net_px * rng.uniform(0.04, 0.22, n_rows), 0.0)
    revenue  = np.where(won, net_px, 0.0)

    orig_svcs = rng.choice(svcs, n_rows)
    dest_svcs = rng.choice(svcs, n_rows)

    df = pd.DataFrame({
        "QUOTE_GROUP_ID":          [f"QG{i:07d}" for i in range(n_rows)],
        "DIGITAL_QUOTE_ID":        [f"DQ{i:07d}" for i in range(n_rows)],
        "DIGITAL_CONFIRMATION_ID": [f"CONF{i:07d}" for i in range(n_rows)],
        "QUOTE_OPP_ID":            [f"OPP{i:07d}" for i in range(n_rows)],
        "QUOTE_DATE":              dates,
        "QUOTE_REQUEST_DATE":      dates,
        "WK_ENDING_DATE":          dates,
        "QUOTE_YEAR":              dates.year,
        "QUOTE_MONTH":             dates.month,
        "QUOTE_TYPE":              rng.choice(["SPOT", "CONTRACT", "PREBOOK"], n_rows),
        "QUOTE_STATUS":            rng.choice(["COMPLETED", "EXPIRED"], n_rows, p=[0.9, 0.1]),
        "LASSO_ID":                customer,
        "CUSTOMER_NAME":           customer,
        "USER_EMAIL":              [f"user{rng.integers(1, 25)}@ups.com" for _ in range(n_rows)],
        "ORIG_COUNTRY":            orig,
        "DEST_COUNTRY":            dest,
        "ORIG_SVC":                orig_svcs,
        "DEST_SVC":                dest_svcs,
        "ORIG_REGION":             rng.choice(regions, n_rows),
        "DEST_REGION":             rng.choice(regions, n_rows),
        "PRODUCT":                 product,
        "PRIMARY_PRODUCT_FLAG":    rng.choice(["Y", "N"], n_rows, p=[0.85, 0.15]),
        "SERVICE_TYPE":            rng.choice(["EXPRESS", "STANDARD", "ECONOMY"], n_rows),
        "MOVEMENT_TYPE":           rng.choice(move_types, n_rows),
        "INCOTERM":                rng.choice(incoterms, n_rows),
        "QUOTE_WGT_KG":            weight,
        "UOM":                     "KG",
        "NET_QUOTE_PRICE_USD":     net_px,
        "ORIG_QUOTE_PRICE_USD":    base_px,
        "PA_RATE":                 pa_rate,
        "RATE_TYPE":               rate_type,
        "CURRENCY":                "USD",
        "DISCOUNT_FLAG":           (disc_pct > 0).astype(str),
        "DISCOUNT_CLASSIFICATION": rng.choice(["STANDARD", "SPECIAL", "NONE"], n_rows),
        "DISCOUNT_TYPE":           rng.choice(disc_types, n_rows),
        "DISCOUNT_PCT":            disc_pct,
        "DISCOUNT_AMT_USD":        base_px * disc_pct,
        "HEAVY_SHOPPER_FLAG":      "N",
        "AI_QUOTE_INDICATOR":      rng.choice(["Y", "N"], n_rows, p=[0.32, 0.68]),
        "API_QUOTE_INDICATOR":     rng.choice(["Y", "N"], n_rows, p=[0.12, 0.88]),
        "PAYOR_TYPE":              rng.choice(payor_types, n_rows),
        "WON":                     won,
        "SHIPMENT_COUNT":          won,
        "REVENUE":                 revenue,
        "GRATE":                   grate,
        "FFS_MARGIN":              np.where(won, grate * rng.uniform(0.3, 0.7, n_rows), 0.0),
        "SHIP_WGT_KG":             np.where(won, weight, np.nan),
        "CUSTOMER_CLASSIFICATION": rng.choice(cust_class, n_rows, p=[0.15, 0.35, 0.30, 0.20]),
    })

    return df
