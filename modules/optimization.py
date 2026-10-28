"""
optimization.py
---------------
ML-based energy management decision algorithm (Figure 6 of the REC paper).

Core logic mirrors Pholboon et al. (2015):
  • At midnight: forecast day-ahead demand & solar; pre-charge battery from
    cheap off-peak grid to cover predicted day-time deficit.
  • During day: consume solar first, store surplus, sell excess, buy deficit.
  • Heuristics δ1, δ2 control pre-charging amounts.

Also exposes a peak-shaving function and a cost-minimisation summary.
"""

import numpy as np
import pandas as pd

from modules.battery_manager import CommunityBattery, Battery


# ─────────────────────────────────────────────────────────────────────────────
# Pricing constants (Section 5.3 of the paper, EUR/kWh)
# ─────────────────────────────────────────────────────────────────────────────

SELL_PRICE_EUR = 0.03          # fixed grid sell price
BUY_PRICE_SIMPLE = 0.1583
BUY_PRICE_PEAK   = 0.1917
BUY_PRICE_OFFPEAK = 0.1044

# Peak hours (bi-hourly daily tariff): 09:00–21:00
PEAK_HOURS = set(range(9, 21))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_peak_hour(hour: int) -> bool:
    return hour in PEAK_HOURS


def buy_price_bihourly(hour: int) -> float:
    return BUY_PRICE_PEAK if is_peak_hour(hour) else BUY_PRICE_OFFPEAK


def compute_delta(consumption_series: np.ndarray, hours: np.ndarray) -> tuple:
    """
    δ1 = sum(consumption 09-18) / sum(consumption full day)
    δ2 = sum(consumption 06-09) / sum(consumption full day)
    """
    total = consumption_series.sum()
    if total == 0:
        return 0.45, 0.10
    peak_window    = consumption_series[np.isin(hours, range(9, 18))].sum()
    morning_window = consumption_series[np.isin(hours, range(6, 9))].sum()
    return float(peak_window / total), float(morning_window / total)


# ─────────────────────────────────────────────────────────────────────────────
# ML-based algorithm (paper Figure 6)
# ─────────────────────────────────────────────────────────────────────────────

def run_ml_algorithm(
    consumption_df: pd.DataFrame,
    solar_df: pd.DataFrame,
    battery_capacity_kwh: float = 6.4,
    n_members: int = 10,
    tariff: str = "bihourly_daily",   # "simple" | "bihourly_daily" | "bihourly_weekly"
) -> pd.DataFrame:
    """
    Simulate the ML-based REC management algorithm over the full dataset.

    Parameters
    ----------
    consumption_df : DataFrame with [timestamp, consumption_kwh]
    solar_df       : DataFrame with [timestamp, solar_kwh_total]
    battery_capacity_kwh : capacity per member (kWh)
    n_members      : number of REC members
    tariff         : electricity purchase tariff name

    Returns
    -------
    DataFrame with per-timestep decisions, costs, and SOC.
    """
    merged = pd.merge(
        consumption_df[["timestamp", "consumption_kwh"]],
        solar_df[["timestamp", "solar_kwh_total"]],
        on="timestamp", how="inner"
    ).sort_values("timestamp").reset_index(drop=True)

    merged["hour"] = pd.to_datetime(merged["timestamp"]).dt.hour

    community_batt = CommunityBattery(battery_capacity_kwh, n_members)
    records = []

    # Group by day for the midnight pre-charging decision
    merged["date"] = pd.to_datetime(merged["timestamp"]).dt.date

    for date, day_df in merged.groupby("date"):
        hours        = day_df["hour"].values
        demand       = day_df["consumption_kwh"].values
        solar        = day_df["solar_kwh_total"].values

        # ── Midnight pre-charge logic ────────────────────────────────────────
        total_demand = demand.sum()
        total_solar  = solar.sum()
        surplus_day  = total_solar - total_demand
        δ1, δ2 = compute_delta(demand, hours)

        batt_available = community_batt.total_capacity

        if total_solar > total_demand * δ1:
            # Enough solar — pre-charge only the gap
            if surplus_day < batt_available:
                precharge_amount = (batt_available - surplus_day) / INVERTER_EFFICIENCY_OPT
            else:
                precharge_amount = demand[hours < 9].sum() * δ2 / INVERTER_EFFICIENCY_OPT
        else:
            # Not enough solar — fill battery fully at midnight
            precharge_amount = community_batt.headroom_kwh / INVERTER_EFFICIENCY_OPT

        # Clamp pre-charge
        precharge_amount = max(0.0, min(precharge_amount, community_batt.headroom_kwh))
        midnight_buy = community_batt.charge(precharge_amount)

        # Midnight cost
        midnight_price = BUY_PRICE_OFFPEAK if tariff != "simple" else BUY_PRICE_SIMPLE
        midnight_cost  = midnight_buy * midnight_price

        # ── Intra-day loop ───────────────────────────────────────────────────
        for idx, row in day_df.iterrows():
            h       = int(row["hour"])
            dem     = float(row["consumption_kwh"])
            sol     = float(row["solar_kwh_total"])
            net     = sol - dem
            soc_bef = community_batt.soc

            if tariff == "simple":
                buy_p = BUY_PRICE_SIMPLE
            elif tariff == "bihourly_weekly":
                # Weekends off-peak all day
                dow = pd.to_datetime(row["timestamp"]).weekday()
                buy_p = BUY_PRICE_OFFPEAK if dow >= 5 else buy_price_bihourly(h)
            else:
                buy_p = buy_price_bihourly(h)

            if net >= 0:
                # Surplus → store then sell
                charged   = community_batt.charge(net)
                remaining = net - charged
                grid_buy  = 0.0
                grid_sell = max(0.0, remaining)
                batt_loss = charged * (1 - INVERTER_EFFICIENCY_OPT)
            else:
                # Deficit: after 10:00 draw battery first; before 10:00 buy grid
                needed = abs(net)
                if h >= 10:
                    from_batt = community_batt.discharge(needed)
                    grid_buy  = max(0.0, needed - from_batt)
                else:
                    grid_buy  = needed
                    from_batt = 0.0
                grid_sell = 0.0
                charged   = 0.0
                batt_loss = 0.0

            step_cost = grid_buy * buy_p - grid_sell * SELL_PRICE_EUR + batt_loss * buy_p

            records.append({
                "timestamp":     row["timestamp"],
                "demand_kwh":    round(dem, 4),
                "solar_kwh":     round(sol, 4),
                "soc":           round(community_batt.soc, 4),
                "grid_buy_kwh":  round(grid_buy, 4),
                "grid_sell_kwh": round(grid_sell, 4),
                "batt_charge":   round(charged, 4),
                "buy_price":     round(buy_p, 4),
                "step_cost_eur": round(step_cost, 5),
                "midnight_cost": round(midnight_cost if h == 0 else 0.0, 5),
            })
            midnight_cost = 0.0   # only record once per day

    return pd.DataFrame(records)


INVERTER_EFFICIENCY_OPT = 0.97


# ─────────────────────────────────────────────────────────────────────────────
# Peak shaving
# ─────────────────────────────────────────────────────────────────────────────

def apply_peak_shaving(
    demand_series: np.ndarray,
    solar_series: np.ndarray,
    peak_threshold_kw: float,
    battery: Battery,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shave demand spikes above `peak_threshold_kw` using the battery.
    Returns (adjusted_demand, battery_soc_trace).
    """
    adjusted = demand_series.copy().astype(float)
    soc_trace = []
    for i, (dem, sol) in enumerate(zip(demand_series, solar_series)):
        net_demand = dem - sol
        if net_demand > peak_threshold_kw:
            excess = net_demand - peak_threshold_kw
            delivered = battery.discharge(excess)
            adjusted[i] = dem - delivered
        soc_trace.append(battery.soc)
    return adjusted, np.array(soc_trace)


# ─────────────────────────────────────────────────────────────────────────────
# Cost optimisation summary
# ─────────────────────────────────────────────────────────────────────────────

def optimisation_summary(sim_df: pd.DataFrame) -> dict:
    """Compute cost and self-sufficiency KPIs from a simulation result."""
    total_cost     = sim_df["step_cost_eur"].sum() + sim_df["midnight_cost"].sum()
    total_bought   = sim_df["grid_buy_kwh"].sum()
    total_sold     = sim_df["grid_sell_kwh"].sum()
    total_demand   = sim_df["demand_kwh"].sum()
    total_solar    = sim_df["solar_kwh"].sum()
    self_suff_pct  = (1 - total_bought / total_demand) * 100 if total_demand > 0 else 0

    # CO2 savings: ~0.233 kg CO2/kWh (UK grid average)
    co2_saved_kg   = total_solar * 0.233

    return {
        "total_cost_eur":       round(total_cost, 2),
        "total_grid_buy_kwh":   round(total_bought, 2),
        "total_grid_sell_kwh":  round(total_sold, 2),
        "total_demand_kwh":     round(total_demand, 2),
        "total_solar_kwh":      round(total_solar, 2),
        "self_sufficiency_pct": round(self_suff_pct, 1),
        "co2_saved_kg":         round(co2_saved_kg, 1),
        "renewable_share_pct":  round(total_solar / total_demand * 100, 1)
                                if total_demand > 0 else 0,
    }
