"""
mini_grid.py
------------
CAPEX, OPEX, and LCOE for solar-diesel hybrid mini-grid technology.

Cost components:
  1. PV + diesel gen system  : capex_per_kw × peak_kW  [USD]
  2. Battery storage          : battery_cost_kwh × kWh_storage [USD]
  3. Connection costs         : capex_per_connection × num_households [USD]
  4. Annual OPEX              : opex_rate × system_capex [USD/year]
  5. Battery replacement      : at battery_replacement_year [USD]

Peak demand sizing:
  peak_kW = num_households × peak_demand_factor
  storage_kWh = peak_kW × battery_hours

Key parameters (from config):
  capex_per_kw          : 3,500 USD/kW
  capex_per_connection  : 300 USD
  opex_rate             : 3% of CAPEX/year
  lifetime_years        : 20
  battery_replacement_year : 10
  battery_cost_kwh      : 250 USD/kWh
  battery_hours         : 4 hours storage
  min_connections       : 50 (viability threshold)
  peak_demand_factor    : 0.25 kW/household

Notes:
  - Mini-grid is infeasible if num_connections < min_connections
  - Building density used as secondary viability check
"""

import numpy as np
import pandas as pd

from src.config import MINIGRID, DISCOUNT_RATE
from src.costs.lcoe_calculator import compute_lcoe


def _peak_demand_kw(num_households: float) -> float:
    """Estimate peak demand [kW] from number of households."""
    return num_households * MINIGRID["peak_demand_factor"]


def _battery_storage_kwh(peak_kw: float) -> float:
    """Battery storage sizing [kWh]."""
    return peak_kw * MINIGRID["battery_hours"]


def minigrid_capex(num_households: float) -> dict:
    """
    Total mini-grid CAPEX breakdown [USD].

    Returns dict with system_capex, connection_capex, battery_capex, total
    """
    peak_kw       = _peak_demand_kw(num_households)
    storage_kwh   = _battery_storage_kwh(peak_kw)

    system_capex  = peak_kw     * MINIGRID["capex_per_kw"]
    battery_capex = storage_kwh * MINIGRID["battery_cost_kwh"]
    conn_capex    = num_households * MINIGRID["capex_per_connection"]

    total = system_capex + battery_capex + conn_capex

    return {
        "system_capex":     system_capex,
        "battery_capex":    battery_capex,
        "connection_capex": conn_capex,
        "total_capex":      total,
        "peak_kw":          peak_kw,
    }


def minigrid_lcoe(
    num_households: float,
    demand_timeseries: list[float],
    num_connections: int = None,
) -> float:
    """
    LCOE for mini-grid [USD/kWh].

    Parameters
    ----------
    num_households    : number of households
    demand_timeseries : annual kWh demand series over planning horizon
    num_connections   : if provided, used for viability check

    Returns
    -------
    float : LCOE [USD/kWh], or np.inf if infeasible
    """
    # Feasibility check — minimum settlement size
    connections = num_connections if num_connections is not None else num_households
    if connections < MINIGRID["min_connections"]:
        return np.inf

    cap = minigrid_capex(num_households)
    total_capex  = cap["total_capex"]
    annual_opex  = total_capex * MINIGRID["opex_rate"]

    # Battery replacement mid-life
    battery_replacement = {
        MINIGRID["battery_replacement_year"]: cap["battery_capex"]
    }

    # Adjust for system losses
    loss_factor = 1 - MINIGRID["loss_rate"]
    energy_net  = [e * loss_factor for e in demand_timeseries]

    return compute_lcoe(
        capex=total_capex,
        annual_opex=annual_opex,
        energy_series=energy_net,
        discount_rate=DISCOUNT_RATE,
        lifetime_years=MINIGRID["lifetime_years"],
        replacement_costs=battery_replacement,
    )


def add_minigrid_lcoe(df: pd.DataFrame) -> pd.DataFrame:
    """Add mini-grid LCOE column to settlements DataFrame."""
    df = df.copy()

    capex_data = df.apply(
        lambda r: minigrid_capex(r["num_households"]),
        axis=1,
        result_type="expand",
    )
    df["minigrid_capex_usd"]   = capex_data["total_capex"]
    df["minigrid_peak_kw"]     = capex_data["peak_kw"]

    df["lcoe_minigrid"] = df.apply(
        lambda r: minigrid_lcoe(
            num_households=r["num_households"],
            demand_timeseries=r["demand_timeseries"],
            num_connections=r.get("num_connections", r["num_households"]),
        ),
        axis=1,
    )

    df["minigrid_feasible"] = df["lcoe_minigrid"] < np.inf

    return df
