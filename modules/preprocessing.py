"""
preprocessing.py
----------------
Handles all data loading, cleaning, feature engineering, and normalisation
for the Smart Grid AI system.  Inspired by the REC paper methodology:
  - lag features up to 24 h
  - rolling averages (3 h, 6 h, 24 h)
  - weather merge
  - outlier removal
  - MinMax normalisation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_energy_data(path: str | None = None) -> pd.DataFrame:
    """Load the hourly electricity consumption CSV."""
    if path is None:
        path = os.path.join(DATA_DIR, "energy_data.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_weather_data(path: str | None = None) -> pd.DataFrame:
    """Load the hourly weather CSV."""
    if path is None:
        path = os.path.join(DATA_DIR, "weather_data.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Replace NaNs with linear interpolation (same approach as paper: interpolate
    previous + following measurement), then fill any remaining NaN with 0.
    """
    df = df.copy()
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def remove_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 3.0) -> pd.DataFrame:
    """Remove outliers beyond `factor * IQR` using interpolation (not deletion)."""
    df = df.copy()
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    mask = (df[col] < lower) | (df[col] > upper)
    df.loc[mask, col] = np.nan
    df[col] = df[col].interpolate(method="linear", limit_direction="both")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical + calendar features from the timestamp column."""
    df = df.copy()
    ts = df["timestamp"]
    df["hour"]       = ts.dt.hour
    df["day"]        = ts.dt.day
    df["month"]      = ts.dt.month
    df["weekday"]    = ts.dt.weekday
    df["is_weekend"] = (ts.dt.weekday >= 5).astype(int)
    df["quarter"]    = ts.dt.quarter
    # Cyclical encoding (avoids discontinuity at midnight / year-end)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_features(df: pd.DataFrame, col: str = "consumption_kwh",
                     lags: list | None = None) -> pd.DataFrame:
    """
    Add lag features.  Default: 1, 2, 3, 6, 12, 24, 48, 168 hours
    (mirrors paper's 'previous readings up to 24 h' + weekly pattern).
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 168]
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str = "consumption_kwh",
                         windows: list | None = None) -> pd.DataFrame:
    """Add rolling mean & std features."""
    if windows is None:
        windows = [3, 6, 24]
    df = df.copy()
    for w in windows:
        df[f"{col}_rolling_mean_{w}h"] = df[col].shift(1).rolling(w).mean()
        df[f"{col}_rolling_std_{w}h"]  = df[col].shift(1).rolling(w).std()
    return df


def merge_weather(energy_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge energy and weather on timestamp (nearest-hour join)."""
    merged = pd.merge_asof(
        energy_df.sort_values("timestamp"),
        weather_df[["timestamp", "temperature_c", "humidity_pct", "dni_wm2"]].sort_values("timestamp"),
        on="timestamp",
        direction="nearest"
    )
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(df: pd.DataFrame, feature_cols: list,
                   scaler_path: str | None = None,
                   fit: bool = True) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    MinMax-scale the specified columns.
    If `fit=True` train a new scaler (and save it); otherwise load existing.
    """
    df = df.copy()
    if not fit and scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = MinMaxScaler()
        scaler.fit(df[feature_cols])
        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)

    df[feature_cols] = scaler.transform(df[feature_cols])
    return df, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(energy_path: str | None = None,
                         weather_path: str | None = None,
                         drop_na: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      load → clean → merge → datetime features → lags → rolling → drop NaN
    Returns a ready-to-use feature matrix (no normalisation applied yet).
    """
    energy_df  = load_energy_data(energy_path)
    weather_df = load_weather_data(weather_path)

    # Clean
    energy_df  = handle_missing_values(energy_df, ["consumption_kwh"])
    energy_df  = remove_outliers_iqr(energy_df, "consumption_kwh")
    weather_df = handle_missing_values(weather_df, ["temperature_c", "humidity_pct", "dni_wm2"])

    # Merge
    df = merge_weather(energy_df, weather_df)

    # Features
    df = add_datetime_features(df)
    df = add_lag_features(df, "consumption_kwh")
    df = add_rolling_features(df, "consumption_kwh")

    if drop_na:
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


def get_feature_columns(df: pd.DataFrame,
                        target: str = "consumption_kwh") -> list:
    """Return all feature columns (everything except timestamp and target)."""
    exclude = {"timestamp", target}
    return [c for c in df.columns if c not in exclude]
