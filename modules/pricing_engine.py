"""
pricing_engine.py
-----------------
Implements the three tariff structures from the REC paper (Section 5.3):
  1. Simple flat tariff
  2. Bi-hourly tariff – daily cycle (peak 09–21)
  3. Bi-hourly tariff – weekly cycle (peak on weekdays 09–21)

Also calculates daily / monthly bills and cost savings.
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Rate table  (EUR/kWh — from REC paper)
# ─────────────────────────────────────────────────────────────────────────────

TARIFFS = {
    "simple": {
        "buy":     0.1583,
        "sell":    0.03,
        "peak":    None,
        "offpeak": None,
    },
    "bihourly_daily": {
        "buy":     None,            # depends on hour
        "sell":    0.03,
        "peak":    0.1917,
        "offpeak": 0.1044,
        "peak_hours": set(range(9, 21)),
    },
    "bihourly_weekly": {
        "buy":     None,
        "sell":    0.03,
        "peak":    0.1917,
        "offpeak": 0.1044,
        "peak_hours": set(range(9, 21)),
        "peak_days": set(range(0, 5)),  # Mon–Fri
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-interval price lookup
# ─────────────────────────────────────────────────────────────────────────────

def get_buy_price(hour: int, weekday: int, tariff: str = "simple") -> float:
    """Return the buy price for a given hour / weekday under the specified tariff."""
    cfg = TARIFFS[tariff]

    if tariff == "simple":
        return cfg["buy"]

    in_peak_hours = hour in cfg["peak_hours"]

    if tariff == "bihourly_daily":
        return cfg["peak"] if in_peak_hours else cfg["offpeak"]

    if tariff == "bihourly_weekly":
        in_peak_day = weekday in cfg["peak_days"]
        return cfg["peak"] if (in_peak_hours and in_peak_day) else cfg["offpeak"]

    return cfg["buy"]


def get_sell_price(tariff: str = "simple") -> float:
    return TARIFFS[tariff]["sell"]


# ─────────────────────────────────────────────────────────────────────────────
# Bill calculator
# ─────────────────────────────────────────────────────────────────────────────

def calculate_bill(
    timestamps: pd.Series,
    grid_buy_kwh: pd.Series,
    grid_sell_kwh: pd.Series,
    tariff: str = "simple",
) -> pd.DataFrame:
    """
    Calculate per-interval electricity cost and return an annotated DataFrame.

    Parameters
    ----------
    timestamps     : datetime Series
    grid_buy_kwh   : energy purchased from grid each interval
    grid_sell_kwh  : energy sold to grid each interval
    tariff         : one of 'simple', 'bihourly_daily', 'bihourly_weekly'
    """
    ts = pd.to_datetime(timestamps)
    hours    = ts.dt.hour.values
    weekdays = ts.dt.weekday.values

    buy_prices  = np.array([get_buy_price(h, d, tariff)
                            for h, d in zip(hours, weekdays)])
    sell_price  = get_sell_price(tariff)

    buy_cost  = np.asarray(grid_buy_kwh)  * buy_prices
    sell_rev  = np.asarray(grid_sell_kwh) * sell_price
    net_cost  = buy_cost - sell_rev

    df = pd.DataFrame({
        "timestamp":    ts,
        "hour":         hours,
        "weekday":      weekdays,
        "grid_buy_kwh": np.asarray(grid_buy_kwh),
        "grid_sell_kwh":np.asarray(grid_sell_kwh),
        "buy_price":    buy_prices,
        "buy_cost_eur": np.round(buy_cost, 5),
        "sell_rev_eur": np.round(sell_rev, 5),
        "net_cost_eur": np.round(net_cost, 5),
    })
    return df


def daily_bill_summary(bill_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly bill DataFrame to daily totals."""
    bill_df = bill_df.copy()
    bill_df["date"] = bill_df["timestamp"].dt.date
    daily = bill_df.groupby("date").agg(
        total_buy_kwh   = ("grid_buy_kwh", "sum"),
        total_sell_kwh  = ("grid_sell_kwh", "sum"),
        total_cost_eur  = ("net_cost_eur", "sum"),
        avg_buy_price   = ("buy_price", "mean"),
    ).reset_index()
    daily["total_cost_eur"] = daily["total_cost_eur"].round(4)
    return daily


def monthly_bill_summary(bill_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly bill DataFrame to monthly totals."""
    bill_df = bill_df.copy()
    bill_df["year_month"] = bill_df["timestamp"].dt.to_period("M").astype(str)
    monthly = bill_df.groupby("year_month").agg(
        total_buy_kwh   = ("grid_buy_kwh", "sum"),
        total_sell_kwh  = ("grid_sell_kwh", "sum"),
        total_cost_eur  = ("net_cost_eur", "sum"),
    ).reset_index()
    monthly["total_cost_eur"] = monthly["total_cost_eur"].round(2)
    return monthly


def compute_savings(
    baseline_cost: float,
    optimised_cost: float,
) -> dict:
    """Compare baseline vs optimised bill."""
    savings_eur  = baseline_cost - optimised_cost
    savings_pct  = (savings_eur / baseline_cost * 100) if baseline_cost != 0 else 0
    return {
        "baseline_cost_eur":  round(baseline_cost, 2),
        "optimised_cost_eur": round(optimised_cost, 2),
        "savings_eur":        round(savings_eur, 2),
        "savings_pct":        round(savings_pct, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Price trace for dashboard
# ─────────────────────────────────────────────────────────────────────────────

def generate_price_trace(
    timestamps: pd.Series,
    tariff: str = "bihourly_daily",
) -> pd.DataFrame:
    """
    Return a DataFrame [timestamp, buy_price] for charting the tariff schedule.
    """
    ts = pd.to_datetime(timestamps)
    prices = [
        get_buy_price(h, d, tariff)
        for h, d in zip(ts.dt.hour, ts.dt.weekday)
    ]
    return pd.DataFrame({"timestamp": ts, "buy_price_eur_kwh": prices})
