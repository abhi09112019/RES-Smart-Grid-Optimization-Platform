"""
app.py
------
AI-Powered Smart Grid Energy Management System
Streamlit dashboard — run with:  streamlit run app.py
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Internal modules ──────────────────────────────────────────────────────────
from modules.preprocessing  import build_feature_matrix, load_energy_data, load_weather_data
from modules.solar_forecast  import forecast_solar, forecast_all_scenarios, solar_summary, get_daily_solar
from modules.battery_manager import simulate_battery_algorithm, battery_analytics, BATTERY_SCENARIOS
from modules.optimization    import run_ml_algorithm, optimisation_summary
from modules.pricing_engine  import calculate_bill, daily_bill_summary, monthly_bill_summary, \
                                     compute_savings, generate_price_trace
from modules.anomaly_detection import detect_all_anomalies, anomaly_summary
from modules.utils           import models_exist, format_kwh, format_eur, format_pct, soc_color
from dashboard.charts        import (
    forecast_vs_actual_chart, next_24h_forecast_chart, feature_importance_chart,
    solar_generation_chart, solar_daily_bar, dni_vs_solar_chart,
    soc_over_time_chart, grid_buy_sell_chart, energy_balance_area_chart,
    price_trace_chart, monthly_bill_chart, savings_gauge,
    energy_source_pie, anomaly_chart, zscore_distribution_chart,
)
from dashboard.metrics       import (
    overview_metrics, forecast_metrics, solar_metrics, battery_metrics,
    optimisation_metrics, pricing_metrics, anomaly_metrics,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Grid AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #262b40 100%);
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid var(--card-color, #2196F3);
        margin-bottom: 8px;
    }
    .metric-label { font-size: 0.78rem; color: #9aa3b2; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 4px; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #f0f2f6; }
    .section-header {
        font-size: 1.2rem; font-weight: 600; color: #e0e4ef;
        border-bottom: 2px solid #2196F3; padding-bottom: 6px;
        margin: 16px 0 12px 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130; border-radius: 8px 8px 0 0;
        padding: 8px 18px; color: #9aa3b2;
    }
    .stTabs [aria-selected="true"] { background-color: #2196F3; color: white; }
    .banner {
        background: linear-gradient(90deg, #0d47a1 0%, #1565c0 40%, #0d47a1 100%);
        border-radius: 12px; padding: 20px 30px; margin-bottom: 20px;
        display: flex; align-items: center; gap: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# KPI card renderer
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, color: str = "#2196F3"):
    st.markdown(f"""
    <div class="metric-card" style="--card-color:{color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>""", unsafe_allow_html=True)


def kpi_row(metrics_list: list[dict]):
    cols = st.columns(len(metrics_list))
    for col, m in zip(cols, metrics_list):
        with col:
            kpi_card(m["label"], m["value"], m.get("color", "#2196F3"))


# ─────────────────────────────────────────────────────────────────────────────
# Cached data loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    energy_df  = load_energy_data()
    weather_df = load_weather_data()
    return energy_df, weather_df


@st.cache_data(show_spinner=False)
def load_feature_matrix():
    return build_feature_matrix()


@st.cache_data(show_spinner=False)
def cached_solar(scenario: str):
    return forecast_solar(scenario)


@st.cache_data(show_spinner=False)
def cached_all_solar():
    return forecast_all_scenarios()


@st.cache_data(show_spinner=False)
def cached_battery_sim(batt_cap, solar_scenario):
    energy_df, _ = load_data()
    solar_df = cached_solar(solar_scenario)
    merged = pd.merge(
        energy_df[["timestamp", "consumption_kwh"]],
        solar_df[["timestamp", "solar_kwh_total"]],
        on="timestamp", how="inner"
    )
    return simulate_battery_algorithm(
        merged["consumption_kwh"], merged["solar_kwh_total"],
        merged["timestamp"], batt_cap, n_members=10,
    )


@st.cache_data(show_spinner=False)
def cached_ml_sim(batt_cap, solar_scenario, tariff):
    energy_df, _ = load_data()
    solar_df = cached_solar(solar_scenario)
    return run_ml_algorithm(
        energy_df[["timestamp", "consumption_kwh"]],
        solar_df[["timestamp", "solar_kwh_total"]],
        battery_capacity_kwh=batt_cap,
        tariff=tariff,
    )


@st.cache_data(show_spinner=False)
def cached_anomalies():
    energy_df, _ = load_data()
    return detect_all_anomalies(energy_df)


@st.cache_data(show_spinner=False)
def cached_forecast():
    """Run XGBoost on the feature matrix and return test predictions."""
    if not models_exist():
        return None, None, None, None
    try:
        from modules.forecast import load_xgboost_model, get_feature_importances
        from sklearn.model_selection import train_test_split
        df = load_feature_matrix()
        from modules.preprocessing import get_feature_columns
        features = get_feature_columns(df, "consumption_kwh")
        X = df[features].values
        y = df["consumption_kwh"].values
        ts = df["timestamp"].values
        _, X_test, _, y_test, _, ts_test = train_test_split(
            X, y, ts, test_size=0.2, shuffle=False
        )
        bundle = load_xgboost_model()
        y_pred = bundle["model"].predict(X_test)
        fi_df = get_feature_importances()
        return ts_test, y_test, y_pred, fi_df
    except Exception as e:
        return None, None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Smart Grid AI")
    st.markdown("---")
    st.markdown("### Configuration")

    solar_scenario = st.selectbox("☀️ Solar Scenario",
                                   ["average", "high", "low"], index=0)
    battery_size   = st.selectbox("🔋 Battery Size",
                                   ["medium", "small", "large"], index=0)
    tariff         = st.selectbox("💰 Electricity Tariff",
                                   ["bihourly_daily", "bihourly_weekly", "simple"])

    batt_cap = BATTERY_SCENARIOS[battery_size]["capacity_kwh"]

    st.markdown("---")

    if not models_exist():
        st.warning("⚠️ Models not trained yet.\nRun `python train_model.py` first.")
    else:
        st.success("✅ Models loaded")

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "Based on the REC paper:\n"
        "_Matos et al. (2024) — Machine Learning-Based Electricity "
        "Consumption Forecast for Renewable Energy Communities_"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <div style="font-size:2.4rem">⚡</div>
  <div>
    <div style="font-size:1.5rem;font-weight:700;color:white">
      AI-Powered Smart Grid Energy Management System
    </div>
    <div style="color:#90caf9;font-size:0.9rem">
      XGBoost Forecasting · Solar PV Analytics · Battery Management · Cost Optimisation
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Pre-load data
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading data …"):
    energy_df, weather_df = load_data()
    solar_df  = cached_solar(solar_scenario)
    all_solar = cached_all_solar()

with st.spinner("Running simulations …"):
    sim_df    = cached_battery_sim(batt_cap, solar_scenario)
    ml_sim_df = cached_ml_sim(batt_cap, solar_scenario, tariff)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Overview",
    "📈 Load Forecast",
    "☀️ Solar Forecast",
    "🔋 Battery Analytics",
    "⚙️ Optimisation",
    "💰 Pricing",
    "🚨 Anomaly Detection",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">System Overview</div>',
                unsafe_allow_html=True)

    ov_metrics = overview_metrics(energy_df, solar_df, sim_df)
    kpi_row(ov_metrics)

    col1, col2 = st.columns(2)
    with col1:
        # Energy balance chart (last 7 days)
        recent = sim_df.tail(24 * 7)
        st.plotly_chart(energy_balance_area_chart(recent),
                        use_container_width=True)
    with col2:
        # Energy source distribution
        batt_ana   = battery_analytics(sim_df)
        solar_total = solar_df["solar_kwh_total"].sum()
        grid_total  = sim_df["grid_buy_kwh"].sum()
        batt_total  = batt_ana["total_discharged_kwh"]
        st.plotly_chart(energy_source_pie(solar_total, grid_total, batt_total),
                        use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Configuration**")
        st.info(f"""
        - **Solar Scenario:** {solar_scenario.title()}
        - **Battery Capacity:** {batt_cap:.1f} kWh/member
        - **Members:** 10
        - **Total Battery:** {batt_cap * 10:.0f} kWh
        - **Tariff:** {tariff.replace('_', ' ').title()}
        """)
    with col4:
        opt_sum = optimisation_summary(ml_sim_df)
        st.markdown("**Optimisation Summary**")
        st.info(f"""
        - **Total Cost:** {format_eur(opt_sum['total_cost_eur'])}
        - **Grid Purchased:** {format_kwh(opt_sum['total_grid_buy_kwh'])}
        - **Renewable Share:** {format_pct(opt_sum['renewable_share_pct'])}
        - **CO₂ Saved:** {opt_sum['co2_saved_kg']:,.0f} kg
        - **Self-Sufficiency:** {format_pct(opt_sum['self_sufficiency_pct'])}
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Load Forecast
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">Load Forecasting — XGBoost</div>',
                unsafe_allow_html=True)

    if not models_exist():
        st.warning("Run `python train_model.py` to train the forecasting model.")
        st.stop()

    ts_test, y_test, y_pred, fi_df = cached_forecast()

    if ts_test is not None:
        from modules.forecast import load_xgboost_model
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        xgb_metrics = {
            "MAE":  round(mean_absolute_error(y_test, y_pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "R2":   round(r2_score(y_test, y_pred), 4),
            "WAPE": round(np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test)) * 100, 2),
        }
        kpi_row(forecast_metrics(xgb_metrics))

        col1, col2 = st.columns([2, 1])
        with col1:
            # Show last 200 test points for readability
            n = min(500, len(ts_test))
            st.plotly_chart(
                forecast_vs_actual_chart(ts_test[-n:], y_test[-n:], y_pred[-n:]),
                use_container_width=True,
            )
        with col2:
            if fi_df is not None:
                st.plotly_chart(feature_importance_chart(fi_df),
                                use_container_width=True)

        # 24-hour ahead
        st.markdown('<div class="section-header">24-Hour Ahead Forecast</div>',
                    unsafe_allow_html=True)
        try:
            from modules.forecast import predict_next_hours
            fc_df = predict_next_hours(24)
            st.plotly_chart(next_24h_forecast_chart(fc_df), use_container_width=True)
            st.dataframe(fc_df.style.format({"forecast_kwh": "{:.4f}"}),
                         use_container_width=True, height=200)
        except Exception as e:
            st.info(f"24h forecast unavailable: {e}")
    else:
        st.info("Train the model first: `python train_model.py`")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Solar Forecast
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">Solar PV Generation Forecast</div>',
                unsafe_allow_html=True)

    summary = solar_summary(solar_scenario)
    kpi_row(solar_metrics(solar_scenario, summary))

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(solar_generation_chart(all_solar), use_container_width=True)
    with col2:
        daily_s = get_daily_solar(solar_scenario)
        st.plotly_chart(solar_daily_bar(daily_s, solar_scenario),
                        use_container_width=True)

    st.plotly_chart(dni_vs_solar_chart(all_solar), use_container_width=True)

    st.markdown("**Scenario Statistics**")
    rows = []
    for sc in ["high", "average", "low"]:
        s = solar_summary(sc)
        rows.append(s)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Battery Analytics
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">Battery Energy Management</div>',
                unsafe_allow_html=True)

    batt_ana = battery_analytics(sim_df)
    kpi_row(battery_metrics(batt_ana))

    col1, col2 = st.columns(2)
    with col1:
        # Show last 14 days for SOC chart
        recent_sim = sim_df.tail(24 * 14)
        st.plotly_chart(soc_over_time_chart(recent_sim), use_container_width=True)
    with col2:
        st.plotly_chart(grid_buy_sell_chart(recent_sim), use_container_width=True)

    st.markdown("**Battery Scenario Comparison**")
    comp_rows = []
    for sc_name, sc_cfg in BATTERY_SCENARIOS.items():
        cap = sc_cfg["capacity_kwh"]
        s = cached_battery_sim(cap, solar_scenario)
        a = battery_analytics(s)
        comp_rows.append({
            "Scenario": sc_name.title(),
            "Capacity (kWh)": cap,
            "Grid Buy (kWh)": a["total_grid_buy_kwh"],
            "Grid Sell (kWh)": a["total_grid_sell_kwh"],
            "Self-Sufficiency (%)": a["self_sufficiency_pct"],
            "Avg SOC (%)": a["avg_soc_pct"],
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Optimisation
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">AI Energy Optimisation</div>',
                unsafe_allow_html=True)

    opt_sum = optimisation_summary(ml_sim_df)
    kpi_row(optimisation_metrics(opt_sum))

    col1, col2 = st.columns(2)
    with col1:
        recent_ml = ml_sim_df.tail(24 * 7)
        st.plotly_chart(soc_over_time_chart(recent_ml), use_container_width=True)
    with col2:
        # Baseline vs ML cost comparison
        baseline_sim = cached_battery_sim(batt_cap, solar_scenario)
        bill_base = calculate_bill(
            baseline_sim["timestamp"],
            baseline_sim["grid_buy_kwh"],
            baseline_sim["grid_sell_kwh"],
            tariff,
        )
        bill_ml = calculate_bill(
            ml_sim_df["timestamp"],
            ml_sim_df["grid_buy_kwh"],
            ml_sim_df["grid_sell_kwh"],
            tariff,
        )
        savings = compute_savings(
            bill_base["net_cost_eur"].sum(),
            bill_ml["net_cost_eur"].sum(),
        )
        st.plotly_chart(savings_gauge(savings["savings_pct"]),
                        use_container_width=True)

    st.markdown("**Algorithm Comparison (Bi-hourly Tariff)**")
    algo_rows = []
    for algo_name, algo_df in [("Battery Algorithm", sim_df),
                                ("ML-Based Algorithm", ml_sim_df)]:
        b = calculate_bill(algo_df["timestamp"], algo_df["grid_buy_kwh"],
                           algo_df["grid_sell_kwh"], tariff)
        algo_rows.append({
            "Algorithm": algo_name,
            "Total Cost (EUR)": round(b["net_cost_eur"].sum(), 2),
            "Grid Buy (kWh)": round(algo_df["grid_buy_kwh"].sum(), 1),
            "Grid Sell (kWh)": round(algo_df["grid_sell_kwh"].sum(), 1),
        })
    st.dataframe(pd.DataFrame(algo_rows), use_container_width=True)

    st.plotly_chart(energy_source_pie(
        solar_df["solar_kwh_total"].sum(),
        ml_sim_df["grid_buy_kwh"].sum(),
        opt_sum["total_grid_buy_kwh"] * 0.15,   # estimated battery discharge
    ), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Pricing
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">Electricity Pricing & Bill Analytics</div>',
                unsafe_allow_html=True)

    # Calculate bills under selected tariff
    bill_df = calculate_bill(
        ml_sim_df["timestamp"],
        ml_sim_df["grid_buy_kwh"],
        ml_sim_df["grid_sell_kwh"],
        tariff,
    )
    bill_base_df = calculate_bill(
        sim_df["timestamp"],
        sim_df["grid_buy_kwh"],
        sim_df["grid_sell_kwh"],
        tariff,
    )
    savings = compute_savings(
        bill_base_df["net_cost_eur"].sum(),
        bill_df["net_cost_eur"].sum(),
    )
    monthly_df = monthly_bill_summary(bill_df)

    kpi_row(pricing_metrics(monthly_df, savings))

    col1, col2 = st.columns(2)
    with col1:
        price_trace = generate_price_trace(energy_df["timestamp"].head(24 * 7), tariff)
        st.plotly_chart(price_trace_chart(price_trace), use_container_width=True)
    with col2:
        st.plotly_chart(monthly_bill_chart(monthly_df), use_container_width=True)

    st.markdown("**Tariff Comparison**")
    tariff_rows = []
    for t_name in ["simple", "bihourly_daily", "bihourly_weekly"]:
        b_ml  = calculate_bill(ml_sim_df["timestamp"],
                               ml_sim_df["grid_buy_kwh"],
                               ml_sim_df["grid_sell_kwh"], t_name)
        b_batt = calculate_bill(sim_df["timestamp"],
                                sim_df["grid_buy_kwh"],
                                sim_df["grid_sell_kwh"], t_name)
        sav = compute_savings(b_batt["net_cost_eur"].sum(), b_ml["net_cost_eur"].sum())
        tariff_rows.append({
            "Tariff": t_name.replace("_", " ").title(),
            "Battery Algo (EUR)": round(b_batt["net_cost_eur"].sum(), 2),
            "ML Algo (EUR)":      round(b_ml["net_cost_eur"].sum(), 2),
            "Savings (EUR)":      sav["savings_eur"],
            "Savings (%)":        sav["savings_pct"],
        })
    st.dataframe(pd.DataFrame(tariff_rows), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Anomaly Detection
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-header">Anomaly Detection</div>',
                unsafe_allow_html=True)

    with st.spinner("Running anomaly detection …"):
        anomaly_df = cached_anomalies()

    a_summary = anomaly_summary(anomaly_df)
    kpi_row(anomaly_metrics(a_summary))

    col1, col2 = st.columns(2)
    with col1:
        # Show last 30 days for readability
        recent_ano = anomaly_df.tail(24 * 30)
        st.plotly_chart(anomaly_chart(recent_ano), use_container_width=True)
    with col2:
        st.plotly_chart(zscore_distribution_chart(anomaly_df),
                        use_container_width=True)

    st.markdown("**Top Anomalous Readings**")
    top_anom = anomaly_df[anomaly_df["is_anomaly"] == 1].nlargest(20, "consumption_kwh")
    if not top_anom.empty:
        display_cols = [c for c in ["timestamp", "consumption_kwh", "z_score",
                                     "anomaly_stat", "hour", "weekday"]
                        if c in top_anom.columns]
        st.dataframe(top_anom[display_cols].reset_index(drop=True),
                     use_container_width=True)
    else:
        st.info("No anomalies detected.")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Smart Grid AI · Inspired by Matos et al. (2024) · "
    "XGBoost | Isolation Forest | Analytical PV Model"
)
