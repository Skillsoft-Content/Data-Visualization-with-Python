"""
Prediction layer.
Loads the persisted model and returns a DataFrame augmented with
WIN_PROBABILITY and CONFIDENCE_LEVEL columns.
"""

import numpy as np
import pandas as pd
import joblib
import os

from src.config import MODEL_CONFIG
from src.train_model import get_feature_matrix, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def model_exists() -> bool:
    return os.path.exists(MODEL_CONFIG["model_path"])


def load_model():
    model        = joblib.load(MODEL_CONFIG["model_path"])
    feature_cols = joblib.load(MODEL_CONFIG["feature_cols_path"])
    encoders     = joblib.load(MODEL_CONFIG["label_encoder_path"])
    return model, feature_cols, encoders


def predict_win_probability(
    df: pd.DataFrame,
    model=None,
    feature_cols: list = None,
    encoders: dict = None,
) -> pd.DataFrame:
    if model is None:
        model, feature_cols, encoders = load_model()

    X, _, _ = get_feature_matrix(df, encoders=encoders, fit=False)

    # Align to the exact columns the model was trained on
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols]

    probs = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["WIN_PROBABILITY"] = probs
    return df


def add_confidence_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Confidence = blend of:
      - How far the predicted probability is from 0.5 (model certainty)
      - How much historical data exists on the lane (volume confidence)
    """
    prob_conf = (df["WIN_PROBABILITY"] - 0.5).abs() * 2  # 0 at 0.5, 1 at extremes

    if "LANE_QUOTE_COUNT" in df.columns:
        vol_conf = (df["LANE_QUOTE_COUNT"].clip(upper=100) / 100).clip(0, 1)
    else:
        vol_conf = pd.Series(0.5, index=df.index)

    df["CONFIDENCE_SCORE"] = (prob_conf * 0.6 + vol_conf * 0.4).clip(0, 1)
    df["CONFIDENCE_LEVEL"]  = pd.cut(
        df["CONFIDENCE_SCORE"],
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    return df


def add_fallback_win_probability(df: pd.DataFrame) -> pd.DataFrame:
    """Used when no trained model is available – uses rolling lane win rate."""
    if "LANE_WIN_RATE" in df.columns:
        df["WIN_PROBABILITY"] = df["LANE_WIN_RATE"].fillna(df["WON"].mean())
    else:
        df["WIN_PROBABILITY"] = df["WON"].mean()
    df["CONFIDENCE_SCORE"] = 0.3
    df["CONFIDENCE_LEVEL"]  = "Low"
    return df
