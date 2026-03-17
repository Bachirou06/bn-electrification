"""
shs.py
------
CAPEX, OPEX, and LCOE for Solar Home Systems (SHS).

SHS are individual household-level solar systems — no shared infrastructure.
Two tiers are modelled:
  Tier 2 SHS (~10–50W): basic lighting, phone charging, small fan
  Tier 3 SHS (~50–200W): lighting, TV, productive appliances

Assignment logic:
  - MTF tier 1–2 → SHS Tier 2 (150 USD/unit)
  - MTF tier 3+  → SHS Tier 3 (350 USD/unit)

Cost components (per household):
  CAPEX  : unit cost of SHS kit [USD]
  OPEX   : annual maintenance / replacement parts [USD/year]
  Replacement: full system replacement at end of lifetime

Key parameters (from config):
  tier_2: capex=150, opex=15/year, lifetime=5 years
  tier_3: capex=350, opex=30/year, lifetime=7 years

Notes:
  - SHS lifetime is shorter than grid/mini-grid → multiple replacement cycles
  - SHS always feasible (no minimum size), but less attractive for large
    dense settlements (mini-grid economies of scale)
"""

import numpy as np
import pandas as pd

from src.config import SHS as SHS_PARAMS, DISCOUNT_RATE, HORIZON
from src.costs.lcoe_calculator import compute_lcoe


def _shs_tier_for_mtf(mtf_tier: int) -> str:
    """Map MTF demand tier to SHS product tier."""
    return "tier_3" if mtf_tier >= 3 else "tier_2"


def shs_capex_per_hh(mtf_tier: int) -> float:
    """Return SHS CAPEX per household based on MTF tier."""
    shs_tier = _shs_tier_for_mtf(mtf_tier)
    return SHS_PARAMS[shs_tier]["capex_per_unit"]


def shs_lcoe_per_hh(mtf_tier: int) -> float:
    """
    LCOE per household for SHS [USD/kWh].

    Accounts for multiple replacement cycles over the planning horizon.
    """
    shs_key  = _shs_tier_for_mtf(mtf_tier)
    params   = SHS_PARAMS[shs_key]

    capex    = params["capex_per_unit"]
    opex     = params["opex_per_unit_year"]
    lifetime = params["lifetime_years"]
    kwh_yr   = params["kwh_per_year"]

    # Build replacement schedule over planning horizon
    # System replaced at each end-of-lifetime within the horizon
    replacement_costs = {}
    yr = lifetime
    while yr <= HORIZON:
        replacement_costs[yr] = capex
        yr += lifetime

    # Constant annual energy
    energy_series = [0.0] + [kwh_yr] * HORIZON

    return compute_lcoe(
        capex=capex,
        annual_opex=opex,
        energy_series=energy_series,
        discount_rate=DISCOUNT_RATE,
        lifetime_years=HORIZON,          # evaluate over full planning horizon
        replacement_costs=replacement_costs,
    )


def shs_lcoe_settlement(
    num_households: float,
    mtf_tier: int,
) -> float:
    """
    Settlement-level SHS LCOE [USD/kWh].
    Since SHS is per-HH, LCOE is the same regardless of settlement size.
    """
    return shs_lcoe_per_hh(mtf_tier)


def add_shs_lcoe(df: pd.DataFrame) -> pd.DataFrame:
    """Add SHS LCOE column to settlements DataFrame."""
    df = df.copy()

    df["shs_tier"]       = df["mtf_tier"].apply(_shs_tier_for_mtf)
    df["shs_capex_usd"]  = df.apply(
        lambda r: shs_capex_per_hh(int(r["mtf_tier"])) * r["num_households"],
        axis=1,
    )
    df["lcoe_shs"] = df.apply(
        lambda r: shs_lcoe_settlement(r["num_households"], int(r["mtf_tier"])),
        axis=1,
    )

    # SHS always feasible
    df["shs_feasible"] = True

    return df
