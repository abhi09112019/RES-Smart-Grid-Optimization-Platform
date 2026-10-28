"""
utils.py
--------
Shared utility functions used across the Smart Grid AI project.
"""

import numpy as np
import pandas as pd
import os


BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ─────────────────────────────────────────────────────────────────────────────
# Numeric helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_divide(numerator: float, denominator: float,
                default: float = 0.0) -> float:
    """Division that returns `default` when denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def co2_saved_kg(solar_kwh: float, grid_factor_kg_kwh: float = 0.233) -> float:
    """
    Estimate CO₂ saved by using solar instead of grid electricity.
    Default factor: UK average ≈ 0.233 kg CO₂/kWh (2023).
    """
    return round(solar_kwh * grid_factor_kg_kwh, 2)


def kwh_to_euros(kwh: float, rate: float) -> float:
    return round(kwh * rate, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Time-series helpers
# ─────────────────────────────────────────────────────────────────────────────

def resample_hourly(df: pd.DataFrame, value_col: str,
                    agg: str = "mean") -> pd.DataFrame:
    """Resample a DataFrame with a timestamp column to hourly frequency."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    resampled = df[[value_col]].resample("h").agg(agg).reset_index()
    return resampled


def last_n_days(df: pd.DataFrame, n: int = 7) -> pd.DataFrame:
    """Return only the last n days of data."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = df["timestamp"].max() - pd.Timedelta(days=n)
    return df[df["timestamp"] >= cutoff].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model management
# ─────────────────────────────────────────────────────────────────────────────

def models_exist() -> bool:
    """Check if the XGBoost model has been trained and saved."""
    xgb_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    return os.path.exists(xgb_path)


def list_saved_models() -> list:
    """Return names of all .pkl model files in the models/ directory."""
    if not os.path.isdir(MODEL_DIR):
        return []
    return [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers (for Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def format_kwh(value: float) -> str:
    if abs(value) >= 1_000:
        return f"{value/1_000:.1f} MWh"
    return f"{value:.2f} kWh"


def format_eur(value: float) -> str:
    return f"€{value:,.2f}"


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def soc_color(soc: float) -> str:
    """Return a hex colour for SOC indicator (green → yellow → red)."""
    if soc >= 0.6:
        return "#2ecc71"
    elif soc >= 0.3:
        return "#f39c12"
    else:
        return "#e74c3c"
