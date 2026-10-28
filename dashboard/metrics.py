"""
metrics.py
----------
KPI card data builders for the Streamlit dashboard.
Each function returns a list of dicts:
  [{"label": str, "value": str, "delta": str | None, "color": str}, …]
"""

import pandas as pd
import numpy as np
from modules.utils import format_kwh, format_eur, format_pct, co2_saved_kg


def overview_metrics(
    consumption_df: pd.DataFrame,
    solar_df: pd.DataFrame,
    sim_df: pd.DataFrame,
) -> list[dict]:
    """Top-level overview KPIs."""
    total_demand  = consumption_df["consumption_kwh"].sum()
    total_solar   = solar_df["solar_kwh_total"].sum()
    total_bought  = sim_df["grid_buy_kwh"].sum()
    self_suff     = (1 - total_bought / total_demand) * 100 if total_demand else 0
    co2           = co2_saved_kg(total_solar)

    return [
        {"label": "Total Demand",       "value": format_kwh(total_demand),
         "delta": None, "color": "#2196F3"},
        {"label": "Total Solar Gen.",   "value": format_kwh(total_solar),
         "delta": None, "color": "#FFC107"},
        {"label": "Self-Sufficiency",   "value": format_pct(self_suff),
         "delta": None, "color": "#4CAF50"},
        {"label": "CO₂ Saved",          "value": f"{co2:,.0f} kg",
         "delta": None, "color": "#009688"},
    ]


def forecast_metrics(metrics_dict: dict) -> list[dict]:
    """Metrics from model evaluation results."""
    return [
        {"label": "MAE",  "value": f"{metrics_dict['MAE']:.4f} kWh",
         "delta": None, "color": "#2196F3"},
        {"label": "RMSE", "value": f"{metrics_dict['RMSE']:.4f} kWh",
         "delta": None, "color": "#FF9800"},
        {"label": "R² Score", "value": f"{metrics_dict['R2']:.4f}",
         "delta": None, "color": "#4CAF50"},
        {"label": "WAPE", "value": f"{metrics_dict['WAPE']:.2f}%",
         "delta": None, "color": "#9C27B0"},
    ]


def solar_metrics(scenario: str, summary: dict) -> list[dict]:
    return [
        {"label": "Scenario",       "value": scenario.title(),
         "delta": None, "color": "#FFC107"},
        {"label": "Mean Output",    "value": f"{summary['mean_kwh']:.4f} kWh/h",
         "delta": None, "color": "#FF9800"},
        {"label": "Peak Output",    "value": f"{summary['max_kwh']:.3f} kWh",
         "delta": None, "color": "#F44336"},
        {"label": "Annual Total",   "value": format_kwh(summary["total_kwh"]),
         "delta": None, "color": "#4CAF50"},
    ]


def battery_metrics(analytics: dict) -> list[dict]:
    return [
        {"label": "Total Grid Buy",   "value": format_kwh(analytics["total_grid_buy_kwh"]),
         "delta": None, "color": "#F44336"},
        {"label": "Total Grid Sell",  "value": format_kwh(analytics["total_grid_sell_kwh"]),
         "delta": None, "color": "#4CAF50"},
        {"label": "Avg SOC",          "value": f"{analytics['avg_soc_pct']:.1f}%",
         "delta": None, "color": "#2196F3"},
        {"label": "Self-Sufficiency", "value": f"{analytics['self_sufficiency_pct']:.1f}%",
         "delta": None, "color": "#009688"},
    ]


def optimisation_metrics(summary: dict) -> list[dict]:
    return [
        {"label": "Total Cost",         "value": format_eur(summary["total_cost_eur"]),
         "delta": None, "color": "#9C27B0"},
        {"label": "Grid Purchased",     "value": format_kwh(summary["total_grid_buy_kwh"]),
         "delta": None, "color": "#F44336"},
        {"label": "Renewable Share",    "value": format_pct(summary["renewable_share_pct"]),
         "delta": None, "color": "#FFC107"},
        {"label": "CO₂ Saved",          "value": f"{summary['co2_saved_kg']:,.0f} kg",
         "delta": None, "color": "#009688"},
    ]


def pricing_metrics(monthly_df: pd.DataFrame, savings: dict) -> list[dict]:
    avg_monthly = monthly_df["total_cost_eur"].mean() if len(monthly_df) > 0 else 0
    return [
        {"label": "Avg Monthly Bill",  "value": format_eur(avg_monthly),
         "delta": None, "color": "#9C27B0"},
        {"label": "Baseline Cost",     "value": format_eur(savings["baseline_cost_eur"]),
         "delta": None, "color": "#F44336"},
        {"label": "Optimised Cost",    "value": format_eur(savings["optimised_cost_eur"]),
         "delta": None, "color": "#4CAF50"},
        {"label": "Total Savings",     "value": f"{format_eur(savings['savings_eur'])} ({savings['savings_pct']:.1f}%)",
         "delta": None, "color": "#009688"},
    ]


def anomaly_metrics(summary: dict) -> list[dict]:
    return [
        {"label": "Total Data Points", "value": f"{summary['total_points']:,}",
         "delta": None, "color": "#2196F3"},
        {"label": "Anomalies Found",   "value": str(summary["n_anomalies"]),
         "delta": None, "color": "#F44336"},
        {"label": "Anomaly Rate",      "value": f"{summary['anomaly_pct']:.2f}%",
         "delta": None, "color": "#FF9800"},
        {"label": "Status",            "value": "⚠ Alert" if summary["anomaly_pct"] > 5 else "✅ Normal",
         "delta": None, "color": "#F44336" if summary["anomaly_pct"] > 5 else "#4CAF50"},
    ]
