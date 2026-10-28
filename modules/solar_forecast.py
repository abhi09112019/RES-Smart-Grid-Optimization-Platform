"""
solar_forecast.py
-----------------
Analytical PV generation forecast based on the paper by Filik et al. (2017)
and directly mirrors the REC paper's PV model:

    solar_output = DNI × panel_area × efficiency × performance_coeff

Three scenarios — high, average, low — are supported.
"""

import numpy as np
import pandas as pd
import os

from modules.preprocessing import load_weather_data

# ─────────────────────────────────────────────────────────────────────────────
# PV Scenario Parameters  (from REC paper, Section 4.3)
# ─────────────────────────────────────────────────────────────────────────────

PV_SCENARIOS = {
    "high": {
        "panel_area_m2": 28.8,      # 18 panels × 1.6 m²
        "efficiency": 0.16,
        "performance_coeff": 0.95,
        "n_members": 10,
    },
    "average": {
        "panel_area_m2": 28.8 * 0.55,
        "efficiency": 0.16,
        "performance_coeff": 0.95,
        "n_members": 10,
    },
    "low": {
        "panel_area_m2": 28.8 * 0.1,   # one-tenth of high
        "efficiency": 0.16,
        "performance_coeff": 0.95,
        "n_members": 10,
    },
}

# DNI is in W/m² in the dataset → convert to kWh/m² per hour: divide by 1000
DNI_TO_KWH = 1 / 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Core formula
# ─────────────────────────────────────────────────────────────────────────────

def pv_output_kwh(dni_wm2: float | np.ndarray,
                  panel_area_m2: float,
                  efficiency: float,
                  performance_coeff: float = 0.95) -> np.ndarray:
    """
    Compute PV output in kWh for one hour.

    Formula:
        E_pv = DNI [kWh/m²] × A [m²] × η × PR

    where PR is the performance ratio (captures temperature/wiring losses).
    """
    dni_kwh = np.asarray(dni_wm2, dtype=float) * DNI_TO_KWH
    output  = dni_kwh * panel_area_m2 * efficiency * performance_coeff
    return np.clip(output, 0, None)


# ─────────────────────────────────────────────────────────────────────────────
# Forecast builders
# ─────────────────────────────────────────────────────────────────────────────

def forecast_solar(scenario: str = "average",
                   weather_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return a DataFrame with columns [timestamp, solar_kwh_per_member, solar_kwh_total].

    Parameters
    ----------
    scenario   : "high" | "average" | "low"
    weather_df : optional pre-loaded weather DataFrame (else loads from disk)
    """
    if scenario not in PV_SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(PV_SCENARIOS)}")

    cfg = PV_SCENARIOS[scenario]

    if weather_df is None:
        weather_df = load_weather_data()

    output_per_member = pv_output_kwh(
        weather_df["dni_wm2"].values,
        cfg["panel_area_m2"],
        cfg["efficiency"],
        cfg["performance_coeff"],
    )

    total_output = output_per_member * cfg["n_members"]

    result = pd.DataFrame({
        "timestamp":              weather_df["timestamp"].values,
        "dni_wm2":                weather_df["dni_wm2"].values,
        "temperature_c":          weather_df["temperature_c"].values,
        "solar_kwh_per_member":   np.round(output_per_member, 5),
        "solar_kwh_total":        np.round(total_output, 5),
        "scenario":               scenario,
    })
    return result


def forecast_all_scenarios(weather_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return combined DataFrame for all three scenarios (stacked)."""
    frames = [forecast_solar(s, weather_df) for s in ("high", "average", "low")]
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def solar_summary(scenario: str = "average") -> dict:
    """Key statistics for one scenario (matching Table 7 in the paper)."""
    df = forecast_solar(scenario)
    col = "solar_kwh_per_member"
    return {
        "scenario":  scenario,
        "mean_kwh":  round(df[col].mean(), 4),
        "std_kwh":   round(df[col].std(), 4),
        "max_kwh":   round(df[col].max(), 4),
        "total_kwh": round(df[col].sum(), 2),
    }


def get_daily_solar(scenario: str = "average") -> pd.DataFrame:
    """Aggregate hourly solar to daily totals per member."""
    df = forecast_solar(scenario)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily = df.groupby("date")["solar_kwh_per_member"].sum().reset_index()
    daily.columns = ["date", "daily_solar_kwh"]
    return daily
