"""
lcoe_calculator.py
------------------
Core NPV and LCOE engine shared by all technology modules.

LCOE (Levelised Cost of Energy) formula:
─────────────────────────────────────────────────────────────────────────
         NPV(all costs over lifetime)
LCOE = ─────────────────────────────────    [USD / kWh]
         NPV(all energy produced over lifetime)

Where:
  NPV(X) = Σ  X_t / (1 + r)^t    for t = 0 … T

Cost stream:
  t=0  : CAPEX (capital expenditure)
  t>0  : OPEX  (annual operating costs)
  t=k  : replacement costs (e.g. battery at year 10)

Energy stream:
  t=0  : 0  (no energy produced in investment year)
  t>0  : annual_energy_kwh (may grow over time)
─────────────────────────────────────────────────────────────────────────

NPC (Net Present Cost) is also computed for comparison:
  NPC = NPV(all costs)   [USD]
"""

import numpy as np
from typing import Union


# ── Core NPV ───────────────────────────────────────────────────────────────────

def npv(cash_flows: list[float], discount_rate: float) -> float:
    """
    Net Present Value of a cash flow series.

    Parameters
    ----------
    cash_flows    : list where index t is the cash flow at year t
    discount_rate : annual discount rate (e.g. 0.10 for 10%)

    Returns
    -------
    float : NPV in same currency as cash_flows
    """
    return sum(
        cf / (1.0 + discount_rate) ** t
        for t, cf in enumerate(cash_flows)
    )


# ── LCOE ──────────────────────────────────────────────────────────────────────

def compute_lcoe(
    capex: float,
    annual_opex: float,
    energy_series: list[float],
    discount_rate: float,
    lifetime_years: int,
    replacement_costs: dict[int, float] | None = None,
) -> float:
    """
    Levelised Cost of Energy [USD/kWh].

    Parameters
    ----------
    capex             : total upfront capital cost [USD]
    annual_opex       : constant annual O&M cost [USD/year]
    energy_series     : list of length (lifetime_years+1); index 0 = 0,
                        index t>0 = energy produced at year t [kWh]
    discount_rate     : annual discount rate
    lifetime_years    : technology lifetime
    replacement_costs : optional dict {year: cost} for mid-life replacements

    Returns
    -------
    float : LCOE [USD/kWh], or np.inf if energy == 0
    """
    T = lifetime_years

    # Build cost stream
    costs = [capex] + [annual_opex] * T

    # Add replacement costs
    if replacement_costs:
        for yr, cost in replacement_costs.items():
            if 0 < yr <= T:
                costs[yr] += cost

    # Truncate or pad energy series to match lifetime
    energy = list(energy_series)
    if len(energy) < T + 1:
        energy = energy + [energy[-1]] * (T + 1 - len(energy))
    energy = [0.0] + energy[1: T + 1]   # t=0 always 0

    npv_cost   = npv(costs,  discount_rate)
    npv_energy = npv(energy, discount_rate)

    if npv_energy <= 0:
        return np.inf

    return npv_cost / npv_energy


# ── NPC ───────────────────────────────────────────────────────────────────────

def compute_npc(
    capex: float,
    annual_opex: float,
    lifetime_years: int,
    discount_rate: float,
    replacement_costs: dict[int, float] | None = None,
) -> float:
    """
    Net Present Cost [USD] — useful for comparing technologies at fixed output.

    Parameters
    ----------
    capex, annual_opex, lifetime_years, discount_rate : as above
    replacement_costs : optional dict {year: cost}

    Returns
    -------
    float : NPC [USD]
    """
    T = lifetime_years
    costs = [capex] + [annual_opex] * T

    if replacement_costs:
        for yr, cost in replacement_costs.items():
            if 0 < yr <= T:
                costs[yr] += cost

    return npv(costs, discount_rate)


# ── Cost per connection ────────────────────────────────────────────────────────

def cost_per_connection(npc: float, num_connections: int) -> float:
    """NPC divided by number of connections [USD/connection]."""
    if num_connections <= 0:
        return np.inf
    return npc / num_connections
