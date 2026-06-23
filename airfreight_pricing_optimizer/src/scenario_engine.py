import numpy as np
from src.config import THRESHOLDS


def _logit(p):
    return np.log(np.clip(p, 1e-6, 1-1e-6) / (1 - np.clip(p, 1e-6, 1-1e-6)))

def _sigmoid(x):
    return float(1 / (1 + np.exp(-x)))


def estimate_new_win_prob(base_prob, price_change_pct, elasticity=None):
    if elasticity is None:
        elasticity = THRESHOLDS["price_elasticity"]
    return _sigmoid(_logit(float(base_prob)) + (-elasticity * price_change_pct))


def run_scenario(lane_row, price_change_pct=0.0, discount_change_pct=0.0,
                max_grate_sacrifice_pct=None, min_shipment_lift=None, elasticity=None):
    max_sac = max_grate_sacrifice_pct or THRESHOLDS["max_grate_sacrifice_pct"]
    min_sl  = min_shipment_lift       or THRESHOLDS["min_shipment_lift"]

    quote_count    = float(lane_row.get("QUOTE_COUNT", 0))
    current_wr     = float(lane_row.get("WIN_PROBABILITY",
                           lane_row.get("PREDICTED_WIN_RATE",
                           lane_row.get("ACTUAL_WIN_RATE", 0.30))))
    grate_per_ship = float(lane_row.get("GRATE_PER_SHIPMENT",
                           lane_row.get("AVG_GRATE_PER_SHIPMENT", 0.0)))

    cur_ships = quote_count * current_wr
    cur_grate = cur_ships  * grate_per_ship
    total_pc  = price_change_pct + discount_change_pct
    new_wr    = estimate_new_win_prob(current_wr, total_pc, elasticity=elasticity)
    new_gps   = grate_per_ship * (1 + total_pc) if grate_per_ship > 0 and total_pc < 0 else grate_per_ship

    scen_ships = quote_count * new_wr
    scen_grate = scen_ships  * new_gps
    ship_lift  = scen_ships - cur_ships
    grate_lift = scen_grate - cur_grate
    sacrifice  = grate_per_ship - new_gps

    rec = _classify(quote_count, grate_per_ship, sacrifice, ship_lift, grate_lift, max_sac, min_sl)
    return {
        "current_win_prob": current_wr, "scenario_win_prob": new_wr,
        "current_expected_shipments": cur_ships, "scenario_expected_shipments": scen_ships,
        "current_expected_grate": cur_grate, "scenario_expected_grate": scen_grate,
        "shipment_lift": ship_lift, "grate_lift": grate_lift,
        "grate_sacrifice_per_shipment": sacrifice, "recommendation": rec,
    }


def _classify(qc, gps, sacrifice, ship_lift, grate_lift, max_sac, min_sl):
    if qc < THRESHOLDS["insufficient_data_quotes"]:  return "Insufficient data"
    if gps < 0:                                        return "Too risky – negative GRATE lane"
    if gps > 0 and sacrifice > gps * max_sac:         return "GRATE risk too high"
    if ship_lift < min_sl:                             return "Not enough shipment lift"
    if grate_lift <= THRESHOLDS["min_grate_lift"]:    return "Expected GRATE lift is negative"
    if grate_lift > 0 and ship_lift >= min_sl:         return "Good opportunity"
    return "Needs pricing review"
