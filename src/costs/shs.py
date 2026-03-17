"""
shs.py — Solar Home Systems LCOE for Benin electrification model.

Kit assignment (post-selection, from mean_rwi):
  mean_rwi >= 0.5  →  Tier-3 ($350, 365 kWh/yr, 7yr lifetime)
  mean_rwi <  0.5  →  Tier-2 ($150,  73 kWh/yr, 5yr lifetime)

LCOE denominator = min(VIDA demand per HH, kit rated kWh/year)
Kit cannot deliver more than its rated output — capping prevents
artificially low LCOE for high-demand settlements.

Sources: GOGLA West Africa 2023; MTF Tier definitions (ESMAP 2015)
"""

import ast
import numpy as np
import pandas as pd

from src.config import SHS as SHS_PARAMS, DISCOUNT_RATE, HORIZON
from src.costs.lcoe_calculator import compute_lcoe


def _to_list(ts):
    """Convert any demand_timeseries type to a plain Python float list."""
    if ts is None:
        return []
    if isinstance(ts, str):
        try:
            ts = ast.literal_eval(ts)
        except Exception:
            return []
    # numpy array, list, or any iterable
    try:
        return [float(x) for x in list(ts)]
    except Exception:
        return []


def shs_kit_from_rwi(mean_rwi):
    """Assign SHS kit tier from RWI."""
    try:
        return "tier_3" if (mean_rwi is not None and float(mean_rwi) >= 0.5) else "tier_2"
    except (TypeError, ValueError):
        return "tier_2"


def shs_lcoe_vida(demand_timeseries, num_households, mean_rwi=0.0):
    """
    LCOE for SHS [USD/kWh].
    Energy denominator capped at kit rated capacity.
    """
    shs_key  = shs_kit_from_rwi(mean_rwi)
    params   = SHS_PARAMS[shs_key]
    capex    = params["capex_per_unit"]
    opex     = params["opex_per_unit_year"]
    lifetime = params["lifetime_years"]
    kit_kwh  = params["kwh_per_year"]

    # Replacement schedule
    replacement_costs = {}
    yr = lifetime
    while yr <= HORIZON:
        replacement_costs[yr] = capex
        yr += lifetime

    # Safe conversion
    ts   = _to_list(demand_timeseries)
    n_hh = max(float(num_households), 1.0)

    if ts and sum(ts) > 0:
        vida_per_hh   = [d / n_hh for d in ts]
        energy_series = [min(v, kit_kwh * (1.04 ** t))
                         for t, v in enumerate(vida_per_hh)]
    else:
        energy_series = [0.0] + [kit_kwh] * HORIZON

    return compute_lcoe(
        capex=capex,
        annual_opex=opex,
        energy_series=energy_series,
        discount_rate=DISCOUNT_RATE,
        lifetime_years=HORIZON,
        replacement_costs=replacement_costs,
    )


def add_shs_lcoe(df: pd.DataFrame) -> pd.DataFrame:
    """Add SHS LCOE columns to settlements DataFrame."""
    df = df.copy()

    rwi_col = "mean_rwi"
    rwi = df[rwi_col].fillna(0.0) if rwi_col in df.columns else pd.Series(0.0, index=df.index)

    df["lcoe_shs"] = df.apply(
        lambda r: shs_lcoe_vida(
            demand_timeseries = r.get("demand_timeseries"),
            num_households    = r.get("num_households", 1) or 1,
            mean_rwi          = r.get(rwi_col, 0.0),
        ),
        axis=1,
    )

    df["shs_kit"] = rwi.apply(shs_kit_from_rwi)

    kit_costs = {
        "tier_2": SHS_PARAMS["tier_2"]["capex_per_unit"],
        "tier_3": SHS_PARAMS["tier_3"]["capex_per_unit"],
    }
    df["shs_capex_usd"] = df.apply(
        lambda r: kit_costs[r["shs_kit"]] * (r.get("num_households", 1) or 1),
        axis=1,
    )

    df["shs_feasible"] = True

    fin  = df["lcoe_shs"].replace(np.inf, np.nan).dropna()
    print(f"SHS LCOE computed: {len(fin):,} valid  "
          f"median=${fin.median():.3f}/kWh  "
          f"mean=${fin.mean():.3f}/kWh")
    return df
