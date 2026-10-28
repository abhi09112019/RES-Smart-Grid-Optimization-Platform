"""
battery_manager.py
------------------
Implements the battery charge / discharge logic described in the REC paper
(Section 4.3 & Figures 5–7).

Three capacity scenarios: small (1.6 kWh), medium (6.4 kWh), large (12.8 kWh)
per REC member.  Battery inverter efficiency: 97 %.
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BATTERY_SCENARIOS = {
    "small":  {"capacity_kwh": 1.6,  "n_members": 10},
    "medium": {"capacity_kwh": 6.4,  "n_members": 10},
    "large":  {"capacity_kwh": 12.8, "n_members": 10},
}

INVERTER_EFFICIENCY = 0.97   # 97 % round-trip
SOC_MIN = 0.10               # never discharge below 10 %
SOC_MAX = 1.00               # 100 % charge limit


# ─────────────────────────────────────────────────────────────────────────────
# Single-member battery class
# ─────────────────────────────────────────────────────────────────────────────

class Battery:
    """
    Models a single REC-member battery.

    State of Charge (SOC) is stored as absolute kWh (not percentage) internally
    and exposed as a 0–1 fraction via the `soc` property.
    """

    def __init__(self, capacity_kwh: float, initial_soc: float = 0.5):
        self.capacity    = capacity_kwh
        self._charge     = capacity_kwh * initial_soc   # kWh stored
        self.efficiency  = INVERTER_EFFICIENCY

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def soc(self) -> float:
        """State of Charge as a fraction [0, 1]."""
        return self._charge / self.capacity

    @property
    def available_kwh(self) -> float:
        """Energy that can still be discharged (respecting SOC_MIN)."""
        return max(0.0, self._charge - SOC_MIN * self.capacity)

    @property
    def headroom_kwh(self) -> float:
        """Space available for charging."""
        return max(0.0, SOC_MAX * self.capacity - self._charge)

    # ── Actions ──────────────────────────────────────────────────────────────

    def charge(self, amount_kwh: float) -> float:
        """
        Charge battery by `amount_kwh` (input energy, losses applied).
        Returns actual energy drawn from source (> amount stored).
        """
        storable = min(amount_kwh * self.efficiency, self.headroom_kwh)
        self._charge += storable
        energy_drawn = storable / self.efficiency
        return round(energy_drawn, 6)

    def discharge(self, amount_kwh: float) -> float:
        """
        Discharge battery to supply `amount_kwh` to load.
        Returns actual energy delivered (≤ amount_kwh due to losses + SOC floor).
        """
        deliverable = min(amount_kwh, self.available_kwh * self.efficiency)
        self._charge -= deliverable / self.efficiency
        self._charge = max(self._charge, SOC_MIN * self.capacity)
        return round(deliverable, 6)

    def reset(self, soc: float = 0.5):
        self._charge = self.capacity * soc


# ─────────────────────────────────────────────────────────────────────────────
# Community battery (aggregate)
# ─────────────────────────────────────────────────────────────────────────────

class CommunityBattery:
    """Aggregate battery representing all REC members combined."""

    def __init__(self, capacity_per_member_kwh: float, n_members: int = 10):
        self.total_capacity = capacity_per_member_kwh * n_members
        self.battery = Battery(self.total_capacity, initial_soc=0.5)

    @property
    def soc(self) -> float:
        return self.battery.soc

    @property
    def available_kwh(self) -> float:
        return self.battery.available_kwh

    @property
    def headroom_kwh(self) -> float:
        return self.battery.headroom_kwh

    def charge(self, amount: float) -> float:
        return self.battery.charge(amount)

    def discharge(self, amount: float) -> float:
        return self.battery.discharge(amount)

    def reset(self, soc: float = 0.5):
        self.battery.reset(soc)


# ─────────────────────────────────────────────────────────────────────────────
# Battery algorithm simulation (simple — paper Figure 5)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_battery_algorithm(
    consumption_series: pd.Series,
    solar_series: pd.Series,
    timestamps: pd.Series,
    battery_capacity_kwh: float = 6.4,
    n_members: int = 10,
    sell_price: float = 0.03,
) -> pd.DataFrame:
    """
    Simple batteries algorithm (Figure 5 of the paper):
      1. Consume production first.
      2. Surplus → charge battery; if still excess → sell to grid.
      3. Deficit  → discharge battery; if still short → buy from grid.

    Returns a per-timestep DataFrame with SOC, grid buy/sell, battery actions.
    """
    community_batt = CommunityBattery(battery_capacity_kwh, n_members)
    records = []

    for ts, demand, solar in zip(timestamps, consumption_series, solar_series):
        net = solar - demand          # positive = surplus, negative = deficit

        soc_before = community_batt.soc

        if net >= 0:
            # Surplus → try to store
            charged    = community_batt.charge(net)
            still_over = net - charged
            grid_buy   = 0.0
            grid_sell  = max(0.0, still_over)
        else:
            # Deficit → draw from battery first
            needed    = abs(net)
            from_batt = community_batt.discharge(needed)
            remaining = needed - from_batt
            grid_buy  = remaining
            grid_sell = 0.0
            charged   = 0.0

        records.append({
            "timestamp":     ts,
            "demand_kwh":    round(demand, 4),
            "solar_kwh":     round(solar, 4),
            "soc":           round(community_batt.soc, 4),
            "soc_before":    round(soc_before, 4),
            "grid_buy_kwh":  round(grid_buy, 4),
            "grid_sell_kwh": round(grid_sell, 4),
            "batt_charge":   round(charged if net >= 0 else 0.0, 4),
            "batt_discharge": round(0.0 if net >= 0 else abs(net) - grid_buy, 4),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Battery analytics helpers
# ─────────────────────────────────────────────────────────────────────────────

def battery_analytics(sim_df: pd.DataFrame) -> dict:
    """Summarise a simulation DataFrame."""
    total_bought  = sim_df["grid_buy_kwh"].sum()
    total_sold    = sim_df["grid_sell_kwh"].sum()
    total_charged = sim_df["batt_charge"].sum()
    total_discharged = sim_df["batt_discharge"].sum()
    avg_soc       = sim_df["soc"].mean()
    return {
        "total_grid_buy_kwh":    round(total_bought, 2),
        "total_grid_sell_kwh":   round(total_sold, 2),
        "total_charged_kwh":     round(total_charged, 2),
        "total_discharged_kwh":  round(total_discharged, 2),
        "avg_soc_pct":           round(avg_soc * 100, 1),
        "self_sufficiency_pct":  round(
            (1 - total_bought / sim_df["demand_kwh"].sum()) * 100, 1
        ),
    }
