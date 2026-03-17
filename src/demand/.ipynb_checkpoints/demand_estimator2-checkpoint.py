"""
demand_estimator.py
-------------------
Estimates annual electricity demand per settlement.

Demand components:
  1. Residential  — households × kWh/HH/year (from MTF tier)
  2. Institutional — health facilities + schools × kWh/facility/year
  3. Productive use top-up — % uplift on residential

Formula:
  demand_year_t = (
      residential_t + institutional + productive_use_t
  )

  residential_t = num_hh
                  × electrification_rate_t
                  × tier_demand_kwh
                  × (1 + growth_rate)^t

  electrification_rate_t = linear interpolation from
      electrification_rate_initial → electrification_rate_target
      over planning_horizon_years
"""

import numpy as np
import pandas as pd

from src.config import DEMAND, GENERAL, AVG_HH_SIZE
from src.demand.mtf_tiers import assign_tiers_df, get_tier_demand


# ── Helpers ────────────────────────────────────────────────────────────────────

def _num_households(population: float) -> float:
    return population / AVG_HH_SIZE


def _electrification_rate(t: int) -> float:
    """
    Linear ramp from initial to target electrification rate over the horizon.
    t = 0 → initial rate
    t = horizon → target rate
    """
    r0  = DEMAND["electrification_rate_initial"]
    r1  = DEMAND["electrification_rate_target"]
    T   = GENERAL["planning_horizon_years"]
    return r0 + (r1 - r0) * min(t / T, 1.0)


def _residential_demand(num_hh: float, tier: int, t: int) -> float:
    """kWh/year residential demand at year t."""
    tier_kwh   = get_tier_demand(tier)
    elec_rate  = _electrification_rate(t)
    growth     = (1 + DEMAND["demand_growth_rate"]) ** t
    return num_hh * elec_rate * tier_kwh * growth


def _institutional_demand(row: pd.Series) -> float:
    """kWh/year for health + education facilities (constant over time)."""
    inst = DEMAND["institutional"]
    n_health = row.get("num_health_facilities", 0) or 0
    n_edu    = row.get("num_education_facilities", 0) or 0
    return (
        n_health * inst["health_facility_kwh_year"]
        + n_edu   * inst["school_kwh_year"]
    )


def _productive_use_demand(residential_kwh: float) -> float:
    """Productive use as a fixed uplift on residential demand."""
    return residential_kwh * DEMAND["institutional"]["productive_use_factor"]


# ── Main functions ─────────────────────────────────────────────────────────────

def estimate_demand_year(row: pd.Series, t: int) -> float:
    """
    Total kWh/year demand for a single settlement at year t.

    Parameters
    ----------
    row : pd.Series
        One row from the settlements GeoDataFrame (must include mtf_tier).
    t   : int
        Year offset from base year (0 = base year).

    Returns
    -------
    float : total annual demand in kWh
    """
    num_hh   = _num_households(row["population"])
    tier     = int(row["mtf_tier"])

    res  = _residential_demand(num_hh, tier, t)
    inst = _institutional_demand(row)
    prod = _productive_use_demand(res)

    return res + inst + prod


def estimate_demand_timeseries(row: pd.Series) -> list[float]:
    """Return list of annual demand values over the full planning horizon."""
    T = GENERAL["planning_horizon_years"]
    return [estimate_demand_year(row, t) for t in range(T + 1)]


def add_demand_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the settlements GeoDataFrame with demand estimates.

    Adds columns:
      mtf_tier              — assigned tier (1–4)
      num_households        — estimated number of households
      demand_year0_kwh      — demand at base year (kWh/year)
      demand_year15_kwh     — demand at end of horizon (kWh/year)
      demand_peak_kwh       — maximum demand over horizon
      demand_timeseries     — list of annual values (for NPV calcs)
    """
    # 1. Assign MTF tiers
    df = assign_tiers_df(df)

    # 2. Number of households
    df["num_households"] = df["population"].apply(_num_households)

    # 3. Demand at t=0 and t=horizon
    T = GENERAL["planning_horizon_years"]
    df["demand_year0_kwh"]  = df.apply(lambda r: estimate_demand_year(r, 0), axis=1)
    df["demand_yearT_kwh"]  = df.apply(lambda r: estimate_demand_year(r, T), axis=1)

    # 4. Full timeseries (stored as list — used in LCOE calc)
    df["demand_timeseries"] = df.apply(estimate_demand_timeseries, axis=1)

    # 5. Demand components at t=0 for transparency
    df["demand_residential_kwh"] = df.apply(
        lambda r: _residential_demand(_num_households(r["population"]), int(r["mtf_tier"]), 0),
        axis=1
    )
    df["demand_institutional_kwh"] = df.apply(_institutional_demand, axis=1)
    df["demand_productive_kwh"]    = df["demand_residential_kwh"] * DEMAND["institutional"]["productive_use_factor"]

    return df
