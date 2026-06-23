"""
Model evaluation: metrics, calibration, lift, confusion matrix, feature importance.
All outputs are plain Python dicts / DataFrames so they can be cached in session state.
"""

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, brier_score_loss, confusion_matrix, roc_curve,
)
from sklearn.calibration import calibration_curve


def evaluate(model, X_test, y_test, feature_cols: list = None, model_name: str = "") -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Core metrics
    metrics = {
        "model_name":       model_name,
        "trained_at":       datetime.now().strftime("%Y-%m-%d %H:%M"),
        "roc_auc":          float(roc_auc_score(y_test, y_prob)),
        "precision":        float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_test, y_pred, zero_division=0)),
        "brier_score":      float(brier_score_loss(y_test, y_prob)),
        "n_test":           int(len(y_test)),
        "n_positive":       int(y_test.sum()),
        "n_negative":       int((1 - y_test).sum()),
        "actual_win_rate":  float(y_test.mean()),
        "predicted_win_rate": float(y_prob.mean()),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
    metrics["calibration"] = {
        "fraction_positives": frac_pos.tolist(),
        "mean_predicted":     mean_pred.tolist(),
    }

    # Lift by probability decile
    try:
        deciles = pd.qcut(y_prob, 10, labels=False, duplicates="drop")
        lift_df = pd.DataFrame({"decile": deciles, "actual": y_test, "prob": y_prob})
        lift_summary = (
            lift_df.groupby("decile")
            .agg(avg_prob=("prob", "mean"), actual_win_rate=("actual", "mean"), count=("actual", "count"))
            .reset_index()
        )
        metrics["lift_chart"] = lift_summary
    except Exception:
        metrics["lift_chart"] = pd.DataFrame()

    # Feature importance (attempt to extract from calibrated classifier)
    if feature_cols:
        try:
            base = model.calibrated_classifiers_[0].estimator
            if hasattr(base, "feature_importances_"):
                fi = pd.DataFrame({"feature": feature_cols, "importance": base.feature_importances_})
                metrics["feature_importance"] = fi.sort_values("importance", ascending=False)
            elif hasattr(base, "coef_"):
                fi = pd.DataFrame({"feature": feature_cols, "importance": np.abs(base.coef_[0])})
                metrics["feature_importance"] = fi.sort_values("importance", ascending=False)
        except Exception:
            pass

    return metrics
