"""
grid_extension.py
-----------------
CAPEX, OPEX, and LCOE for grid extension technology.

Cost components:
  1. MV line extension  : cost_per_km × distance_to_grid [USD]
  2. Connection costs   : cost_per_connection × num_households [USD]
  3. Annual OPEX        : opex_rate × total_capex [USD/year]

Key parameters (from config):
  capex_per_km_mv     : 45,000 USD/km  (MV 33kV line)
  capex_per_km_hv     : 80,000 USD/km  (HV 63kV+ spur)
  capex_per_connection: 400 USD
  opex_rate           : 2% of CAPEX/year
  lifetime_years      : 30
  loss_rate           : 12%
  max_viable_dist_km  : 25 km

Notes:
  - Distance used is distance_to_existing_transmission_lines (base case)
    or dist_to_existing_planned_transmission_lines_2017 (optimistic scenario)
  - Voltage of nearest line determines per-km cost
  - Settlements beyond max_viable_distance_km get LCOE = np.inf (infeasible)
"""

import numpy as np
import pandas as pd

from src.config import GRID, GENERAL, DISCOUNT_RATE, HORIZON
from src.costs.lcoe_calculator import compute_lcoe, compute_npc


def _line_cost_per_km(voltage_kv: float | None) -> float:
    """Return USD/km based on voltage level of nearest line."""
    if voltage_kv is not None and voltage_kv >= GRID["hv_threshold_kv"]:
        return GRID["capex_per_km_hv"]
    return GRID["capex_per_km_mv"]


def grid_capex(
    distance_km: float,
    num_households: float,
    voltage_kv: float | None = None,
) -> float:
    """
    Total grid extension CAPEX [USD].

    Parameters
    ----------
    distance_km    : distance from settlement to nearest MV/HV line [km]
    num_households : number of households to connect
    voltage_kv     : voltage of nearest line (determines cost/km)
    """
    line_cost   = _line_cost_per_km(voltage_kv) * distance_km
    conn_cost   = GRID["capex_per_connection"] * num_households
    return line_cost + conn_cost


def grid_lcoe(
    distance_km: float,
    num_households: float,
    demand_timeseries: list[float],
    voltage_kv: float | None = None,
    scenario: str = "existing",
) -> float:
    """
    LCOE for grid extension [USD/kWh].

    Parameters
    ----------
    distance_km       : km to nearest grid line
    num_households    : number of HH in settlement
    demand_timeseries : list of annual kWh demand over planning horizon
    voltage_kv        : voltage of nearest line
    scenario          : 'existing' or 'planned' (informational only)

    Returns
    -------
    float : LCOE [USD/kWh], or np.inf if infeasible
    """
    # Feasibility check
    if distance_km > GRID["max_viable_distance_km"] or distance_km <= 0:
        return np.inf

    capex      = grid_capex(distance_km, num_households, voltage_kv)
    annual_opex = capex * GRID["opex_rate"]

    # Adjust energy for transmission losses
    loss_factor = 1 - GRID["loss_rate"]
    energy_net  = [e * loss_factor for e in demand_timeseries]

    # Align to grid lifetime (may differ from planning horizon)
    lifetime = GRID["lifetime_years"]

    return compute_lcoe(
        capex=capex,
        annual_opex=annual_opex,
        energy_series=energy_net,
        discount_rate=DISCOUNT_RATE,
        lifetime_years=lifetime,
    )


def add_grid_lcoe(df: pd.DataFrame, dist_col: str = "distance_to_existing_transmission_lines") -> pd.DataFrame:
    """
    Add grid extension LCOE column to settlements DataFrame.

    Parameters
    ----------
    df       : settlements GeoDataFrame with demand_timeseries column
    dist_col : which distance column to use (base case vs optimistic scenario)
    """
    df = df.copy()

    df["grid_capex_usd"] = df.apply(
        lambda r: grid_capex(
            distance_km=r[dist_col],
            num_households=r["num_households"],
        ),
        axis=1,
    )

    df["lcoe_grid"] = df.apply(
        lambda r: grid_lcoe(
            distance_km=r[dist_col],
            num_households=r["num_households"],
            demand_timeseries=r["demand_timeseries"],
        ),
        axis=1,
    )

    df["grid_feasible"] = df[dist_col] <= GRID["max_viable_distance_km"]

    return df
