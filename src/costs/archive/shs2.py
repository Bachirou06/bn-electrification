"""
shs.py
------
CAPEX, OPEX, and LCOE for Solar Home Systems (SHS).

Kit assignment — POST-SELECTION based on RWI (no MTF tier needed)
=================================================================
Kit size is assigned AFTER technology selection, not before.
This removes the circularity where kit capacity was driving the
demand assumption.

  mean_rwi >= 0.5  →  Tier-3 kit ($350) — higher wealth, larger loads
  mean_rwi <  0.5  →  Tier-2 kit ($150) — lower wealth, basic loads

LCOE energy denominator — VIDA demand (not kit rated capacity)
=============================================================
LCOE = (kit_cost × CRF + OPEX) / demand_per_connection_kwh

The denominator is VIDA's satellite-derived demand per connection,
not the kit's rated kWh/year. This means LCOE reflects actual demand.
"""

import numpy as np
import pandas as pd

from src.config import SHS as SHS_PARAMS, DISCOUNT_RATE, HORIZON
from src.costs.lcoe_calculator import compute_lcoe


# ── Kit assignment from RWI (post-selection) ──────────────────────────────────

def shs_kit_from_rwi(mean_rwi: float) -> str:
    """Assign SHS kit tier from RWI. Called AFTER technology selection."""
    return "tier_3" if (pd.notna(mean_rwi) and mean_rwi >= 0.5) else "tier_2"


# ── LCOE using VIDA demand per connection ─────────────────────────────────────

def shs_lcoe_vida(
    demand_timeseries: list,
    num_households: float,
    mean_rwi: float = 0.0,
) -> float:
    """
    LCOE for SHS [USD/kWh].

    Kit cost determined by RWI.
    Energy denominator = VIDA demand_timeseries / num_households.

    Parameters
    ----------
    demand_timeseries : VIDA annual kWh demand over planning horizon
    num_households    : number of households
    mean_rwi          : Relative Wealth Index for kit assignment
    """
    shs_key = shs_kit_from_rwi(mean_rwi)
    params  = SHS_PARAMS[shs_key]

    capex    = params["capex_per_unit"]
    opex     = params["opex_per_unit_year"]
    lifetime = params["lifetime_years"]

    # Replacement schedule over planning horizon
    replacement_costs = {}
    yr = lifetime
    while yr <= HORIZON:
        replacement_costs[yr] = capex
        yr += lifetime

    # Energy per household from VIDA — safe conversion from any type
    import numpy as _np
    if isinstance(demand_timeseries, str):
        import ast as _ast
        demand_timeseries = _ast.literal_eval(demand_timeseries)
    if demand_timeseries is None:
        demand_timeseries = []
    ts = [float(x) for x in demand_timeseries]
    n_hh = max(num_households, 1)
    if len(ts) > 0 and sum(ts) > 0:
        energy_series = [d / n_hh for d in ts]
    else:
        energy_series = [0.0] + [params["kwh_per_year"]] * HORIZON

    return compute_lcoe(
        capex=capex,
        annual_opex=opex,
        energy_series=energy_series,
        discount_rate=DISCOUNT_RATE,
        lifetime_years=HORIZON,
        replacement_costs=replacement_costs,
    )


# ── DataFrame-level function ──────────────────────────────────────────────────

def add_shs_lcoe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SHS LCOE columns to settlements DataFrame.

    Kit type from RWI (post-selection label — does not affect LCOE calculation).
    LCOE denominator = VIDA demand_timeseries / num_households.

    New columns:
      lcoe_shs          — LCOE [USD/kWh]
      shs_capex_usd     — total SHS CAPEX (kit_cost x num_households)
      shs_kit           — 'tier_2' or 'tier_3' (kit recommendation)
                          NOTE: this is a planning label, not an LCOE input
      shs_feasible      — always True (SHS is always a fallback option)
    """
    df = df.copy()

    rwi_col = "mean_rwi"
    rwi     = df[rwi_col].fillna(0.0) if rwi_col in df.columns else pd.Series(0.0, index=df.index)

    # LCOE — demand-driven, kit cost from RWI
    df["lcoe_shs"] = df.apply(
        lambda r: shs_lcoe_vida(
            demand_timeseries = r.get("demand_timeseries", []),
            num_households    = r.get("num_households", 1) or 1,
            mean_rwi          = r.get(rwi_col, 0.0) or 0.0,
        ),
        axis=1,
    )

    # Kit recommendation (post-selection label)
    df["shs_kit"] = rwi.apply(shs_kit_from_rwi)

    # CAPEX = kit_cost x num_households
    kit_costs = {"tier_2": SHS_PARAMS["tier_2"]["capex_per_unit"],
                 "tier_3": SHS_PARAMS["tier_3"]["capex_per_unit"]}
    df["shs_capex_usd"] = df.apply(
        lambda r: kit_costs[r["shs_kit"]] * (r.get("num_households", 1) or 1),
        axis=1,
    )

    df["shs_feasible"] = True
    return df
