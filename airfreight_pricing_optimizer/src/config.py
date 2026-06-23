"""
Central configuration: database, column mappings, business thresholds, model settings.
All tunable parameters live here so no business logic is scattered through the app.
"""

# ── Database ──────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "server":   r"SVRP000F968A.us.ups.com\WWBIAP21",
    "database": "IAF_Compliance",
    "driver":   "ODBC Driver 17 for SQL Server",
}

QUOTE_TABLE   = "[IAF_Compliance].[dbo].[UFH_AIR_QUOTE_FULL_DETAILS]"
BOOKING_TABLE = "[IAF_Compliance].[dbo].[UFH_AIR_BOOKINGS]"

# ── Business thresholds ───────────────────────────────────────────────────────
THRESHOLDS = {
    "low_win_rate":  0.30,
    "high_win_rate": 0.65,
    "min_quote_count":           10,
    "insufficient_data_quotes":   5,
    "min_grate_per_shipment":   0.0,
    "max_grate_sacrifice_pct":  0.15,
    "min_shipment_lift":  0.5,
    "min_grate_lift":     0.0,
    "max_recommended_discount_pct": 0.25,
    "min_confidence_for_recommendation": 0.50,
    "price_elasticity": 1.5,
    "test_months": 3,
    "win_prob_threshold": 0.50,
}

# ── Recommendation labels ─────────────────────────────────────────────────────
RECOMMENDATIONS = {
    "lower_price":        "Lower price selectively",
    "maintain":           "Maintain price",
    "increase":           "Increase price",
    "review":             "Needs pricing review",
    "insufficient_data":  "Insufficient data",
    "no_discount_grate":  "Do not discount – GRATE risk",
}

# ── Model configuration ───────────────────────────────────────────────────────
MODEL_CONFIG = {
    "model_path":          "models/best_model.pkl",
    "feature_cols_path":   "models/feature_cols.pkl",
    "label_encoder_path":  "models/label_encoders.pkl",
    "metrics_path":        "models/model_metrics.pkl",
    "n_estimators":        300,
    "max_depth":           6,
    "learning_rate":       0.05,
    "random_state":        42,
    "cv_folds":            3,
}

# ── File paths ────────────────────────────────────────────────────────────────
DATA_PATH   = "data/quote_data.csv"
MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs"
