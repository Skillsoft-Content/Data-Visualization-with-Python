"""
Model training pipeline.
  1. Time-based train / test split (no random leakage)
  2. Historical features computed on training set only, then applied to test set
  3. Candidates: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM
  4. Each model is calibrated with isotonic regression
  5. Best model selected on CV ROC-AUC and persisted to disk
"""

import os
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration        import CalibratedClassifierCV
from sklearn.preprocessing      import LabelEncoder
from sklearn.model_selection    import cross_val_score

from src.config import MODEL_CONFIG, THRESHOLDS

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ── Feature lists ─────────────────────────────────────────────────────────────

CATEGORICAL_FEATURES = [
    "ORIG_COUNTRY", "DEST_COUNTRY", "ORIG_SVC", "DEST_SVC",
    "PRODUCT", "RATE_TYPE", "LANE", "SERVICE_PAIR", "REGION_PAIR",
    "QUOTE_SOURCE", "DISCOUNT_BUCKET", "CUSTOMER_TYPE",
    "MOVEMENT_TYPE", "INCOTERM", "PAYOR_TYPE",
]

NUMERIC_FEATURES = [
    "QUOTE_WGT_KG",
    "NET_QUOTE_PRICE_USD",
    "PRICE_PER_KG",
    "DISCOUNT_PCT",
    "ABOVE_PA_FLAG",
    "PRICE_VS_PA",
    "ABOVE_LANE_AVG_PRICE_FLAG",
    "PRICE_VS_LANE_AVG",
    "LANE_WIN_RATE",
    "LANE_QUOTE_COUNT",
    "LANE_AVG_GRATE",
    "LANE_AVG_KG",
    "CUST_WIN_RATE",
    "CUST_QUOTE_COUNT",
    "CUST_LANE_WIN_RATE",
    "RATE_TYPE_WIN_RATE",
    "PRODUCT_WIN_RATE",
    "QUOTE_MONTH",
    "QUOTE_QUARTER",
    "QUOTE_DAY_OF_WEEK",
]


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "UNKNOWN"
        df[col] = df[col].astype(str).fillna("UNKNOWN")

        if fit:
            le = LabelEncoder()
            le.fit(list(df[col].unique()) + ["UNKNOWN"])
            encoders[col] = le

        le = encoders[col]
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else "UNKNOWN")
        df[col] = le.transform(df[col])

    return df, encoders


def get_feature_matrix(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    available    = [f for f in all_features if f in df.columns]

    df_enc, encoders = encode_categoricals(df, encoders=encoders, fit=fit)
    X = df_enc[available].copy()

    for col in available:
        if col in NUMERIC_FEATURES:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X, available, encoders


# ── Train / test split ────────────────────────────────────────────────────────

def time_split(df: pd.DataFrame, test_months: int = None):
    n  = test_months or THRESHOLDS["test_months"]
    df = df.sort_values("QUOTE_DATE")
    cutoff = df["QUOTE_DATE"].max() - pd.DateOffset(months=n)
    return df[df["QUOTE_DATE"] <= cutoff].copy(), df[df["QUOTE_DATE"] > cutoff].copy()


# ── Candidate models ──────────────────────────────────────────────────────────

def _candidates():
    rs  = MODEL_CONFIG["random_state"]
    ne  = MODEL_CONFIG["n_estimators"]
    md  = MODEL_CONFIG["max_depth"]
    lr  = MODEL_CONFIG["learning_rate"]

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=rs, class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=ne, max_depth=md, random_state=rs,
            class_weight="balanced", n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=ne, max_depth=md, learning_rate=lr, random_state=rs,
        ),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=ne, max_depth=md, learning_rate=lr,
            random_state=rs, eval_metric="logloss", verbosity=0,
        )
    if HAS_LGB:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=ne, max_depth=md, learning_rate=lr,
            random_state=rs, n_jobs=-1, verbose=-1,
        )
    return models


# ── Main training function ────────────────────────────────────────────────────

def train_and_select_best(df: pd.DataFrame, progress_callback=None):
    """
    Full pipeline: feature engineering → split → train → calibrate → select.
    Returns (model, name, cv_scores, feature_cols, encoders, test_df, X_test, y_test).
    """
    from src.feature_engineering import run_feature_engineering

    df = df[df["QUOTE_DATE"].notna()].copy()
    df = run_feature_engineering(df, reference_df=df)

    train_df, test_df = time_split(df)
    if len(train_df) < 100:
        raise ValueError(f"Training set too small: {len(train_df)} rows after time split.")

    # Re-compute historical features using only training data
    train_df = run_feature_engineering(train_df, reference_df=train_df)
    test_df  = run_feature_engineering(test_df,  reference_df=train_df)

    X_train, feature_cols, encoders = get_feature_matrix(train_df, fit=True)
    y_train = train_df["WON"].values

    X_test, _, _ = get_feature_matrix(test_df, encoders=encoders, fit=False)
    missing_cols  = [c for c in feature_cols if c not in X_test.columns]
    for c in missing_cols:
        X_test[c] = 0
    X_test = X_test[feature_cols]
    y_test = test_df["WON"].values

    candidates  = _candidates()
    cv_scores   = {}
    best_score  = -np.inf
    best_name   = None
    best_model  = None
    n_models    = len(candidates)

    for i, (name, base_model) in enumerate(candidates.items()):
        if progress_callback:
            progress_callback(i / n_models, f"Training {name}…")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calibrated = CalibratedClassifierCV(
                base_model, cv=MODEL_CONFIG["cv_folds"], method="isotonic"
            )
            calibrated.fit(X_train, y_train)
            scores    = cross_val_score(calibrated, X_train, y_train,
                                        cv=3, scoring="roc_auc")
            mean_auc  = float(scores.mean())
            cv_scores[name] = mean_auc

            if mean_auc > best_score:
                best_score = mean_auc
                best_name  = name
                best_model = calibrated

    if progress_callback:
        progress_callback(1.0, f"Best model: {best_name} (AUC {best_score:.3f})")

    # Persist
    os.makedirs(MODELS_DIR := MODEL_CONFIG["model_path"].rsplit("/", 1)[0], exist_ok=True)
    joblib.dump(best_model,  MODEL_CONFIG["model_path"])
    joblib.dump(feature_cols, MODEL_CONFIG["feature_cols_path"])
    joblib.dump(encoders,    MODEL_CONFIG["label_encoder_path"])

    return best_model, best_name, cv_scores, feature_cols, encoders, test_df, X_test, y_test
