"""
Scenario engine.
Models the effect of a price or discount change on win probability and
expected GRATE using a logit-shift elasticity approach.

Key assumption: a 1% price reduction shifts the log-odds of winning
by `elasticity` units (configurable in config.THRESHOLDS).
"""

import numpy as np
from src.config import THRESHOLDS


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return np.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def estimate_new_win_prob(
    base_prob: float,
    price_change_pct: float,
    elasticity: float = None,
) -> float:
    """
    price_change_pct: negative = price reduction (e.g. -0.05 = 5% cheaper)
    Returns new win probability after the price shift.
    """
    if elasticity is None:
        elasticity = THRESHOLDS["price_elasticity"]
    # Price decrease → more competitive → log-odds increase
    logit_shift = -elasticity * price_change_pct
    return _sigmoid(_logit(base_prob) + logit_shift)


def run_scenario(
    lane_row: dict,
    price_change_pct: float = 0.0,
    discount_change_pct: float = 0.0,
    max_grate_sacrifice_pct: float = None,
    min_shipment_lift: float = None,
    elasticity: float = None,
) -> dict:
    """
    Evaluate a single pricing scenario for one lane.

    Parameters
    ----------
    lane_row              : lane-level summary row (dict)
    price_change_pct      : proposed change in base price (e.g. -0.05 = -5%)
    discount_change_pct   : additional discount on top of current (e.g. 0.05 = +5%)
    max_grate_sacrifice_pct : max fraction of GRATE/shipment we're willing to sacrifice
    min_shipment_lift     : minimum expected additional shipments to justify the change
    elasticity            : log-odds sensitivity to 1% price change

    Returns
    -------
    dict with current and scenario KPIs plus a plain-text recommendation.
    """
    max_sac = max_grate_sacrifice_pct or THRESHOLDS["max_grate_sacrifice_pct"]
    min_sl  = min_shipment_lift       or THRESHOLDS["min_shipment_lift"]

    quote_count  = float(lane_row.get("QUOTE_COUNT", 0))
    current_wr   = float(lane_row.get("WIN_PROBABILITY",
                         lane_row.get("PREDICTED_WIN_RATE",
                         lane_row.get("ACTUAL_WIN_RATE", 0.30))))
    grate_per_ship = float(lane_row.get("GRATE_PER_SHIPMENT",
                           lane_row.get("AVG_GRATE_PER_SHIPMENT", 0.0)))

    # Current state
    cur_exp_ships = quote_count * current_wr
    cur_exp_grate = cur_exp_ships * grate_per_ship

    # Net price change (reduction wins more but also reduces GRATE margin)
    total_pc  = price_change_pct + discount_change_pct
    new_wr    = estimate_new_win_prob(current_wr, total_pc, elasticity=elasticity)

    # GRATE per shipment moves proportionally with price (simplified)
    if grate_per_ship > 0 and total_pc < 0:
        new_gps = grate_per_ship * (1 + total_pc)
    else:
        new_gps = grate_per_ship

    scen_exp_ships = quote_count * new_wr
    scen_exp_grate = scen_exp_ships * new_gps

    ship_lift        = scen_exp_ships - cur_exp_ships
    grate_lift       = scen_exp_grate - cur_exp_grate
    grate_sacrifice  = grate_per_ship - new_gps  # positive = we give up GRATE per ship

    recommendation = _classify_scenario(
        quote_count=quote_count,
        grate_per_ship=grate_per_ship,
        grate_sacrifice=grate_sacrifice,
        ship_lift=ship_lift,
        grate_lift=grate_lift,
        max_sac_pct=max_sac,
        min_sl=min_sl,
    )

    return {
        "current_win_prob":           current_wr,
        "scenario_win_prob":          new_wr,
        "current_expected_shipments": cur_exp_ships,
        "scenario_expected_shipments":scen_exp_ships,
        "current_expected_grate":     cur_exp_grate,
        "scenario_expected_grate":    scen_exp_grate,
        "shipment_lift":              ship_lift,
        "grate_lift":                 grate_lift,
        "grate_sacrifice_per_shipment": grate_sacrifice,
        "recommendation":             recommendation,
    }


def _classify_scenario(
    quote_count, grate_per_ship, grate_sacrifice,
    ship_lift, grate_lift, max_sac_pct, min_sl,
) -> str:
    if quote_count < THRESHOLDS["insufficient_data_quotes"]:
        return "Insufficient data"
    if grate_per_ship < 0:
        return "Too risky – negative GRATE lane"
    if grate_per_ship > 0 and grate_sacrifice > grate_per_ship * max_sac_pct:
        return "GRATE risk too high"
    if ship_lift < min_sl:
        return "Not enough shipment lift"
    if grate_lift <= THRESHOLDS["min_grate_lift"]:
        return "Expected GRATE lift is negative"
    if grate_lift > 0 and ship_lift >= min_sl:
        return "Good opportunity"
    return "Needs pricing review"
