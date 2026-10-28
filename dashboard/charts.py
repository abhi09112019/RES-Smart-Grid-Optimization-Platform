"""
charts.py
---------
All Plotly chart builders used by the Streamlit dashboard.
Each function returns a plotly.graph_objects.Figure.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Colour palette
C_BLUE   = "#2196F3"
C_GREEN  = "#4CAF50"
C_ORANGE = "#FF9800"
C_RED    = "#F44336"
C_PURPLE = "#9C27B0"
C_TEAL   = "#009688"
C_YELLOW = "#FFC107"

TEMPLATE = "plotly_dark"


# ─────────────────────────────────────────────────────────────────────────────
# Load forecast
# ─────────────────────────────────────────────────────────────────────────────

def forecast_vs_actual_chart(
    timestamps, actual, predicted, title: str = "Load Forecast vs Actual"
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=actual,
        name="Actual", line=dict(color=C_BLUE, width=2),
        mode="lines"
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=predicted,
        name="Predicted", line=dict(color=C_ORANGE, width=2, dash="dot"),
        mode="lines"
    ))
    fig.update_layout(
        template=TEMPLATE, title=title,
        xaxis_title="Time", yaxis_title="kWh",
        legend=dict(x=0.01, y=0.99), height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def next_24h_forecast_chart(forecast_df: pd.DataFrame) -> go.Figure:
    """Area chart for the 24-hour ahead forecast."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["forecast_kwh"],
        fill="tozeroy",
        fillcolor="rgba(33,150,243,0.2)",
        line=dict(color=C_BLUE, width=2),
        name="Forecast",
    ))
    fig.update_layout(
        template=TEMPLATE, title="24-Hour Load Forecast",
        xaxis_title="Hour", yaxis_title="kWh",
        height=320, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def feature_importance_chart(fi_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=fi_df["importance"], y=fi_df["feature"],
        orientation="h",
        marker_color=C_BLUE,
    ))
    fig.update_layout(
        template=TEMPLATE, title="Top Feature Importances",
        xaxis_title="Importance Score",
        height=380, margin=dict(l=40, r=20, t=50, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Solar forecast
# ─────────────────────────────────────────────────────────────────────────────

def solar_generation_chart(solar_df: pd.DataFrame) -> go.Figure:
    """Multi-scenario solar generation area chart."""
    fig = go.Figure()
    colors = {"high": C_YELLOW, "average": C_ORANGE, "low": C_RED}
    for scenario, color in colors.items():
        s = solar_df[solar_df["scenario"] == scenario]
        fig.add_trace(go.Scatter(
            x=s["timestamp"], y=s["solar_kwh_per_member"],
            name=f"{scenario.title()} PV",
            line=dict(color=color, width=1.5),
            mode="lines",
        ))
    fig.update_layout(
        template=TEMPLATE, title="PV Generation — All Scenarios (per member)",
        xaxis_title="Time", yaxis_title="kWh",
        height=380, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def solar_daily_bar(daily_df: pd.DataFrame, scenario: str = "average") -> go.Figure:
    fig = go.Figure(go.Bar(
        x=daily_df["date"], y=daily_df["daily_solar_kwh"],
        marker_color=C_YELLOW, name="Daily Solar",
    ))
    fig.update_layout(
        template=TEMPLATE, title=f"Daily Solar Generation — {scenario.title()}",
        xaxis_title="Date", yaxis_title="kWh/member/day",
        height=320, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def dni_vs_solar_chart(solar_df: pd.DataFrame) -> go.Figure:
    s = solar_df[solar_df["scenario"] == "high"].head(24 * 7)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=s["timestamp"], y=s["dni_wm2"],
        name="DNI (W/m²)", line=dict(color=C_YELLOW)), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=s["timestamp"], y=s["solar_kwh_per_member"],
        name="Solar Output (kWh)", line=dict(color=C_ORANGE)), secondary_y=True)
    fig.update_layout(
        template=TEMPLATE, title="DNI vs Solar Output (1 week, high scenario)",
        height=340, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Battery analytics
# ─────────────────────────────────────────────────────────────────────────────

def soc_over_time_chart(sim_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sim_df["timestamp"], y=sim_df["soc"] * 100,
        fill="tozeroy", fillcolor="rgba(76,175,80,0.2)",
        line=dict(color=C_GREEN, width=2),
        name="State of Charge (%)",
    ))
    fig.add_hline(y=20, line_dash="dot", line_color=C_RED,
                  annotation_text="Min SOC 20%")
    fig.add_hline(y=80, line_dash="dot", line_color=C_YELLOW,
                  annotation_text="Warn 80%")
    fig.update_layout(
        template=TEMPLATE, title="Battery State of Charge Over Time",
        xaxis_title="Time", yaxis_title="SOC (%)",
        yaxis=dict(range=[0, 105]),
        height=360, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def grid_buy_sell_chart(sim_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sim_df["timestamp"], y=sim_df["grid_buy_kwh"],
        name="Grid Buy", marker_color=C_RED, opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        x=sim_df["timestamp"], y=-sim_df["grid_sell_kwh"],
        name="Grid Sell", marker_color=C_GREEN, opacity=0.8,
    ))
    fig.update_layout(
        template=TEMPLATE, title="Grid Buy / Sell Activity",
        barmode="overlay", xaxis_title="Time", yaxis_title="kWh",
        height=340, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def energy_balance_area_chart(sim_df: pd.DataFrame) -> go.Figure:
    """Stacked area: demand, solar, battery discharge, grid buy."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sim_df["timestamp"], y=sim_df["demand_kwh"],
        name="Demand", line=dict(color=C_BLUE, width=2), mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=sim_df["timestamp"], y=sim_df["solar_kwh"],
        name="Solar", fill="tozeroy",
        fillcolor="rgba(255,193,7,0.25)",
        line=dict(color=C_YELLOW, width=1.5),
    ))
    fig.update_layout(
        template=TEMPLATE, title="Energy Balance: Demand vs Solar Generation",
        xaxis_title="Time", yaxis_title="kWh",
        height=360, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Pricing
# ─────────────────────────────────────────────────────────────────────────────

def price_trace_chart(price_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=price_df["timestamp"], y=price_df["buy_price_eur_kwh"],
        line=dict(color=C_ORANGE, width=1.5), fill="tozeroy",
        fillcolor="rgba(255,152,0,0.15)", name="Buy Price",
    ))
    fig.update_layout(
        template=TEMPLATE, title="Electricity Buy Price Over Time",
        xaxis_title="Time", yaxis_title="EUR/kWh",
        height=320, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def monthly_bill_chart(monthly_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=monthly_df["year_month"],
        y=monthly_df["total_cost_eur"],
        marker_color=C_PURPLE, name="Monthly Bill",
    ))
    fig.update_layout(
        template=TEMPLATE, title="Monthly Electricity Bill (EUR)",
        xaxis_title="Month", yaxis_title="EUR",
        height=320, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def savings_gauge(savings_pct: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=savings_pct,
        delta={"reference": 0},
        title={"text": "Cost Savings vs Baseline (%)"},
        gauge={
            "axis": {"range": [0, 15]},
            "bar": {"color": C_GREEN},
            "steps": [
                {"range": [0, 3],  "color": "#37474f"},
                {"range": [3, 7],  "color": "#455a64"},
                {"range": [7, 15], "color": "#546e7a"},
            ],
            "threshold": {
                "line": {"color": C_ORANGE, "width": 3},
                "thickness": 0.75,
                "value": savings_pct,
            },
        },
        number={"suffix": "%", "font": {"size": 36}},
    ))
    fig.update_layout(template=TEMPLATE, height=280,
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Energy source distribution
# ─────────────────────────────────────────────────────────────────────────────

def energy_source_pie(solar_kwh: float, grid_kwh: float,
                      battery_kwh: float) -> go.Figure:
    labels = ["Solar PV", "Grid", "Battery"]
    values = [max(0, solar_kwh), max(0, grid_kwh), max(0, battery_kwh)]
    colors = [C_YELLOW, C_RED, C_GREEN]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors),
        hole=0.45, textinfo="label+percent",
    ))
    fig.update_layout(
        template=TEMPLATE, title="Energy Source Distribution",
        height=320, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly detection
# ─────────────────────────────────────────────────────────────────────────────

def anomaly_chart(df: pd.DataFrame,
                  col: str = "consumption_kwh") -> go.Figure:
    normal   = df[df["is_anomaly"] == 0]
    anomalies = df[df["is_anomaly"] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["timestamp"], y=normal[col],
        mode="lines", name="Normal",
        line=dict(color=C_BLUE, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=anomalies["timestamp"], y=anomalies[col],
        mode="markers", name="Anomaly",
        marker=dict(color=C_RED, size=8, symbol="x"),
    ))
    if "anomaly_iqr_upper" in df.columns:
        fig.add_hline(
            y=df["anomaly_iqr_upper"].iloc[0],
            line_dash="dot", line_color=C_ORANGE,
            annotation_text="IQR Upper Fence",
        )
    fig.update_layout(
        template=TEMPLATE, title="Anomaly Detection — Energy Consumption",
        xaxis_title="Time", yaxis_title="kWh",
        height=380, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def zscore_distribution_chart(df: pd.DataFrame) -> go.Figure:
    if "z_score" not in df.columns:
        return go.Figure()
    fig = px.histogram(
        df, x="z_score", nbins=60,
        color_discrete_sequence=[C_BLUE],
        title="Z-Score Distribution",
        template=TEMPLATE,
    )
    fig.add_vline(x=3, line_dash="dash", line_color=C_RED,
                  annotation_text="Threshold (3σ)")
    fig.update_layout(height=320, margin=dict(l=40, r=20, t=50, b=40))
    return fig
