"""
Recommendation engine.
Aggregates quote-level data to lane level, then classifies each lane and
computes scenario-based GRATE lift estimates.
"""

import pandas as pd
import numpy as np

from src.config import THRESHOLDS, RECOMMENDATIONS
from src.scenario_engine import run_scenario


# ── Lane-level aggregation ────────────────────────────────────────────────────

def compute_lane_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (LANE, PRODUCT, SERVICE_PAIR, RATE_TYPE).
    Uses WIN_PROBABILITY if available, falls back to ACTUAL_WIN_RATE.
    """
    prob_col = "WIN_PROBABILITY" if "WIN_PROBABILITY" in df.columns else "WON"

    # All quotes
    quote_agg = (
        df.groupby(["LANE", "PRODUCT", "SERVICE_PAIR", "RATE_TYPE"], dropna=False)
        .agg(
            QUOTE_COUNT        = ("QUOTE_GROUP_ID",  "nunique"),
            SHIPMENT_COUNT     = ("WON",             "sum"),
            ACTUAL_WIN_RATE    = ("WON",             "mean"),
            PREDICTED_WIN_RATE = (prob_col,          "mean"),
            AVG_KG             = ("QUOTE_WGT_KG",    "mean"),
            AVG_PRICE_PER_KG   = ("PRICE_PER_KG",    "mean"),
            AVG_DISCOUNT_PCT   = ("DISCOUNT_PCT",     "mean"),
        )
        .reset_index()
    )

    # Won quotes only (for GRATE / revenue metrics)
    won_df  = df[df["WON"] == 1]
    won_agg = (
        won_df.groupby(["LANE", "PRODUCT", "SERVICE_PAIR", "RATE_TYPE"], dropna=False)
        .agg(
            REVENUE                = ("REVENUE", "sum"),
            GRATE                  = ("GRATE",   "sum"),
            AVG_GRATE_PER_SHIPMENT = ("GRATE",   "mean"),
        )
        .reset_index()
    )

    agg = quote_agg.merge(
        won_agg,
        on=["LANE", "PRODUCT", "SERVICE_PAIR", "RATE_TYPE"],
        how="left",
    )
    agg[["REVENUE", "GRATE", "AVG_GRATE_PER_SHIPMENT"]] = (
        agg[["REVENUE", "GRATE", "AVG_GRATE_PER_SHIPMENT"]].fillna(0)
    )

    # Derived
    agg["GRATE_PER_SHIPMENT"] = np.where(
        agg["SHIPMENT_COUNT"] > 0,
        agg["GRATE"] / agg["SHIPMENT_COUNT"],
        0.0,
    )
    agg["GRATE_PER_KG"] = np.where(
        (agg["SHIPMENT_COUNT"] > 0) & (agg["AVG_KG"] > 0),
        agg["GRATE"] / (agg["SHIPMENT_COUNT"] * agg["AVG_KG"]),
        0.0,
    )
    agg["WIN_PROBABILITY"]    = agg["PREDICTED_WIN_RATE"]
    agg["EXPECTED_SHIPMENTS"] = agg["QUOTE_COUNT"] * agg["PREDICTED_WIN_RATE"]
    agg["EXPECTED_GRATE"]     = agg["EXPECTED_SHIPMENTS"] * agg["GRATE_PER_SHIPMENT"]

    return agg


# ── Recommendation assignment ─────────────────────────────────────────────────

def apply_recommendations(
    lane_df: pd.DataFrame,
    default_price_change: float = -0.05,
) -> pd.DataFrame:
    """
    Classify every lane row and attach scenario output columns.
    default_price_change: the discount scenario used to compute expected lifts.
    """
    t = THRESHOLDS
    recs, conf_levels, reasons = [], [], []
    s_ships, s_grates, s_slift, s_glift = [], [], [], []

    for _, row in lane_df.iterrows():
        qc       = row["QUOTE_COUNT"]
        wr       = row["ACTUAL_WIN_RATE"]
        gps      = row["GRATE_PER_SHIPMENT"]
        pred_wr  = row.get("PREDICTED_WIN_RATE", wr)

        if qc < t["insufficient_data_quotes"]:
            rec    = RECOMMENDATIONS["insufficient_data"]
            reason = f"Only {int(qc)} quotes (min {t['insufficient_data_quotes']})"
            conf   = "Low"
            so = (0, 0, 0, 0)

        elif gps < t["min_grate_per_shipment"] and gps < 0:
            rec    = RECOMMENDATIONS["no_discount_grate"]
            reason = f"GRATE/shipment ${gps:,.0f} – negative, do not discount"
            conf   = "High"
            so = (0, 0, 0, 0)

        elif wr >= 0.80:
            rec    = RECOMMENDATIONS["increase"]
            reason = f"Win rate {wr:.0%} is very strong – pricing power opportunity"
            conf   = "High" if qc >= t["min_quote_count"] else "Medium"
            so = (0, 0, 0, 0)

        elif wr >= t["high_win_rate"]:
            rec    = RECOMMENDATIONS["maintain"]
            reason = f"Win rate {wr:.0%} is healthy – maintain pricing"
            conf   = "High" if qc >= t["min_quote_count"] else "Medium"
            so = (0, 0, 0, 0)

        elif wr < t["low_win_rate"] and qc >= t["min_quote_count"] and gps >= 0:
            scenario = run_scenario(
                row.to_dict(), price_change_pct=default_price_change
            )
            sl, gl = scenario["shipment_lift"], scenario["grate_lift"]
            sr = scenario["recommendation"]

            if sr == "Good opportunity":
                rec    = RECOMMENDATIONS["lower_price"]
                reason = (f"Win rate {wr:.0%} is low; scenario shows "
                          f"+{sl:.1f} shipments and +${gl:,.0f} GRATE")
                conf   = "High" if qc >= t["min_quote_count"] * 2 else "Medium"
            elif "GRATE risk" in sr:
                rec    = RECOMMENDATIONS["no_discount_grate"]
                reason = f"GRATE sacrifice would exceed {t['max_grate_sacrifice_pct']:.0%} tolerance"
                conf   = "Medium"
            else:
                rec    = RECOMMENDATIONS["review"]
                reason = f"Low win rate {wr:.0%} but scenario uplift insufficient – manual review"
                conf   = "Low"

            so = (
                scenario["scenario_expected_shipments"],
                scenario["scenario_expected_grate"],
                sl, gl,
            )

        elif wr < t["low_win_rate"] and qc < t["min_quote_count"]:
            rec    = RECOMMENDATIONS["review"]
            reason = f"Win rate {wr:.0%} is low but only {int(qc)} quotes – needs more volume"
            conf   = "Low"
            so = (0, 0, 0, 0)

        else:
            rec    = RECOMMENDATIONS["maintain"]
            reason = f"Win rate {wr:.0%} and GRATE within acceptable range"
            conf   = "Medium"
            so = (0, 0, 0, 0)

        recs.append(rec)
        conf_levels.append(conf)
        reasons.append(reason)
        s_ships.append(so[0])
        s_grates.append(so[1])
        s_slift.append(so[2])
        s_glift.append(so[3])

    lane_df = lane_df.copy()
    lane_df["RECOMMENDATION"]              = recs
    lane_df["CONFIDENCE_LEVEL"]            = conf_levels
    lane_df["REASON"]                      = reasons
    lane_df["SCENARIO_EXPECTED_SHIPMENTS"] = s_ships
    lane_df["SCENARIO_EXPECTED_GRATE"]     = s_grates
    lane_df["EXPECTED_SHIPMENT_LIFT"]      = s_slift
    lane_df["EXPECTED_GRATE_LIFT"]         = s_glift

    return lane_df
