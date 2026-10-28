"""
anomaly_detection.py
--------------------
Detects unusual power spikes and abnormal consumption patterns using:
  - Isolation Forest  (sklearn)
  - Z-score / IQR statistical method (fast, no training needed)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Isolation Forest
# ─────────────────────────────────────────────────────────────────────────────

def train_isolation_forest(
    df: pd.DataFrame,
    feature_cols: list,
    contamination: float = 0.03,
    save: bool = True,
) -> IsolationForest:
    """
    Train an Isolation Forest on the given features.

    contamination : expected fraction of anomalies in the dataset (≈3 %)
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].dropna())

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump({"model": model, "scaler": scaler,
                     "feature_cols": feature_cols}, ANOMALY_MODEL_PATH)

    return model, scaler


def detect_anomalies_iforest(
    df: pd.DataFrame,
    feature_cols: list | None = None,
) -> pd.DataFrame:
    """
    Label each row as anomaly (1) or normal (0) using Isolation Forest.
    Trains fresh if no saved model is found.
    """
    df = df.copy()
    if feature_cols is None:
        feature_cols = ["consumption_kwh", "hour", "weekday"]
        feature_cols = [c for c in feature_cols if c in df.columns]

    if os.path.exists(ANOMALY_MODEL_PATH):
        bundle  = joblib.load(ANOMALY_MODEL_PATH)
        model   = bundle["model"]
        scaler  = bundle["scaler"]
        cols    = bundle["feature_cols"]
    else:
        model, scaler = train_isolation_forest(df, feature_cols)
        cols = feature_cols

    valid_cols = [c for c in cols if c in df.columns]
    X = scaler.transform(df[valid_cols].fillna(0))
    preds = model.predict(X)           # +1 normal, -1 anomaly
    scores = model.score_samples(X)    # lower = more anomalous

    df["anomaly_iforest"]      = (preds == -1).astype(int)
    df["anomaly_score_iforest"] = np.round(scores, 4)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Statistical  (Z-score + IQR)
# ─────────────────────────────────────────────────────────────────────────────

def detect_anomalies_statistical(
    df: pd.DataFrame,
    col: str = "consumption_kwh",
    z_threshold: float = 3.0,
    iqr_factor: float = 2.5,
) -> pd.DataFrame:
    """
    Flag rows where `col` is an outlier via both Z-score AND IQR tests.

    A row is anomalous if it exceeds the Z-score threshold OR falls outside
    the IQR fence (more sensitive for skewed distributions).
    """
    df = df.copy()
    values = df[col].values.astype(float)

    # Z-score
    mean, std = np.nanmean(values), np.nanstd(values)
    z_scores  = np.abs((values - mean) / (std + 1e-9))

    # IQR
    q1, q3 = np.nanpercentile(values, 25), np.nanpercentile(values, 75)
    iqr    = q3 - q1
    lower  = q1 - iqr_factor * iqr
    upper  = q3 + iqr_factor * iqr
    iqr_flag = (values < lower) | (values > upper)

    df["z_score"]          = np.round(z_scores, 3)
    df["anomaly_stat"]     = ((z_scores > z_threshold) | iqr_flag).astype(int)
    df["anomaly_iqr_lower"] = lower
    df["anomaly_iqr_upper"] = upper
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Combined detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_all_anomalies(df: pd.DataFrame,
                         col: str = "consumption_kwh") -> pd.DataFrame:
    """Apply both methods and create a unified `is_anomaly` flag."""
    df = detect_anomalies_statistical(df, col)

    try:
        feature_cols = [c for c in ["consumption_kwh", "hour", "weekday"]
                        if c in df.columns]
        df = detect_anomalies_iforest(df, feature_cols)
        df["is_anomaly"] = (
            (df["anomaly_stat"] == 1) | (df["anomaly_iforest"] == 1)
        ).astype(int)
    except Exception:
        df["is_anomaly"] = df["anomaly_stat"]

    return df


def anomaly_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict for the dashboard."""
    if "is_anomaly" not in df.columns:
        df = detect_all_anomalies(df)
    n_total    = len(df)
    n_anomaly  = int(df["is_anomaly"].sum())
    pct        = round(n_anomaly / n_total * 100, 2) if n_total > 0 else 0
    worst_rows = df[df["is_anomaly"] == 1].nlargest(5, "consumption_kwh")
    return {
        "total_points":    n_total,
        "n_anomalies":     n_anomaly,
        "anomaly_pct":     pct,
        "worst_timestamps": worst_rows["timestamp"].tolist() if "timestamp" in df.columns else [],
        "worst_values":     worst_rows["consumption_kwh"].round(3).tolist(),
    }
