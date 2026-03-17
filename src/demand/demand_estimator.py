"""
demand_estimator.py
-------------------
Estimates annual electricity demand per settlement using VIDA satellite data.

DEMAND MODEL — TWO SEPARATE GROWTH COMPONENTS
==============================================
Previously the model used a single 4% growth rate that conflated two
distinct phenomena. This version separates them:

  1. demand_intensity_growth_rate (3%/yr)
     Income-driven growth in kWh *per household* — existing households
     buy fans, TVs, fridges as incomes rise.
     Derivation: Benin GDP ~4%/yr × SSA income elasticity ~0.75 ≈ 3%/yr
     Source: Foster & Briceño-Garmendia (2010); IEA Africa Energy Outlook 2022

  2. population_growth_rate (2.7%/yr) × rural_unelec_share (40%)
     New households forming each year that need first-time connections.
     Effective new-HH rate in unelectrified settlements ≈ 1.1%/yr
     Source: UN World Population Prospects 2024 (medium-fertility)

  Combined effective demand growth ≈ (1.03 × 1.011) - 1 ≈ 4.1%/yr
  — close to the old 4% but now decomposed into two transparent, defensible
  and separately testable components.

TIER PROGRESSION FOR NEWLY CONNECTED HOUSEHOLDS
================================================
Newly electrified households do not immediately reach their assigned MTF
tier. ESMAP MTF surveys show ~80% of newly connected rural households
upgrade from Tier-1 to Tier-2 within 5 years. We model this as a linear
ramp-up from Tier-1 demand to the assigned tier demand over
`years_to_target_tier` (default 5 years).

This is only applied to *unelectrified* settlements being connected —
already-electrified settlements use full tier demand from t=0.

SHS PRICE DECLINE
=================
SHS kit prices have fallen ~8-10%/yr over 2018-2023. We apply a
conservative 5%/yr decline through 2030 (floor: 70% of 2025 price).
This is passed into shs.py via the demand module so that LCOE for SHS
reflects realistic forward pricing.
"""

import pandas as pd
import numpy as np
from src.config import DEMAND, GENERAL, SHS as SHS_CONFIG


# ── Growth helpers ─────────────────────────────────────────────────────────────

def _intensity_factor(t: int) -> float:
    """
    Compound income-driven per-HH demand growth at year t.
    Rate: demand_intensity_growth_rate (3%/yr) from config.
    """
    return (1 + DEMAND["demand_intensity_growth_rate"]) ** t


def _hh_growth_factor(t: int) -> float:
    """
    Household count growth factor at year t.
    Only the fraction of population growth in unelectrified areas.
    Rate: population_growth_rate × rural_unelec_share ≈ 1.1%/yr
    """
    effective_rate = (
        DEMAND["population_growth_rate"] * DEMAND["rural_unelec_share"]
    )
    return (1 + effective_rate) ** t


def _combined_growth_factor(t: int) -> float:
    """
    Combined growth: intensity × household count.
    ≈ (1.03 × 1.011)^t — replaces old (1.04)^t
    """
    return _intensity_factor(t) * _hh_growth_factor(t)


# ── Tier progression (newly connected HH ramp up from Tier-1) ─────────────────

def _tier_progression_factor(t: int, assigned_tier_kwh: float,
                              is_unelectrified: bool = True) -> float:
    """
    Fraction of assigned-tier demand applicable at year t.

    For unelectrified settlements newly being connected:
      - t=0: Tier-1 demand (4.3 kWh/HH/yr) — just connected, no appliances yet
      - t=years_to_target: full assigned tier demand
      - Linear interpolation between

    For already-electrified settlements: factor = 1.0 always.

    Returns a multiplier (0 < factor <= 1.0) on assigned_tier_kwh.
    """
    if not is_unelectrified:
        return 1.0

    cfg = DEMAND.get("tier_progression", {})
    if not cfg.get("enabled", True):
        return 1.0

    years_to_target = cfg.get("years_to_target_tier", 5)
    tier1_kwh = DEMAND["mtf_tiers"]["tier_1"]

    if assigned_tier_kwh <= tier1_kwh:
        return 1.0   # already at Tier-1 — no ramp needed

    if t >= years_to_target:
        return 1.0   # fully ramped up

    # Linear interpolation from Tier-1 to assigned tier
    tier1_fraction = tier1_kwh / assigned_tier_kwh
    ramp = tier1_fraction + (1.0 - tier1_fraction) * (t / years_to_target)
    return min(ramp, 1.0)


# ── Productive use uplift ──────────────────────────────────────────────────────

def _productive_use_factor(row: pd.Series) -> float:
    """
    Demand uplift for settlements with productive use anchor loads.

    Qualifying criteria (any one is sufficient):
      - has_health_facility  : health centre refrigeration + 24h power
      - has_education_facility: school evening lighting + admin
      - cropland land cover  : grain milling, irrigation, cold storage

    Uplift factor and coverage from config.productive_use section.
    Default: 1.30 (30% uplift) for 25% of settlements.

    Returns 1.0 if productive use is disabled in config or not applicable.
    """
    pu_cfg = DEMAND.get("productive_use", {})
    if not pu_cfg.get("enabled", False):
        return 1.0

    factor = pu_cfg.get("uplift_factor", 1.30)

    # Check qualifying criteria
    if row.get("has_health_facility", False):
        return factor
    if row.get("has_education_facility", False):
        return factor
    # Land cover: IGBP classes 12 (cropland) and 14 (cropland/natural mosaic)
    lc = int(row.get("LandCover", 0) or 0)
    if lc in (12, 14):
        return factor

    return 1.0



# ── SHS price decline ──────────────────────────────────────────────────────────

def shs_price_at_year(base_price: float, target_year: int) -> float:
    """
    Adjusted SHS kit price at a given calendar year, applying the
    5%/yr price decline curve configured in assumptions.yaml.

    Parameters
    ----------
    base_price  : 2025 kit price (USD)
    target_year : calendar year of the purchase (e.g. 2025, 2030)

    Returns
    -------
    float : adjusted price (never below floor)
    """
    cfg = SHS_CONFIG.get("price_decline", {})
    if not cfg.get("enabled", True):
        return base_price

    base_year   = GENERAL["base_year"]          # 2025
    annual_rate = cfg.get("annual_rate", 0.05)  # 5%/yr
    floor_year  = cfg.get("floor_year", 2030)
    floor_pct   = cfg.get("floor_pct", 0.70)

    years_elapsed = max(0, min(target_year, floor_year) - base_year)
    adjusted = base_price * (1 - annual_rate) ** years_elapsed
    floor_price = base_price * floor_pct
    return max(adjusted, floor_price)


# ── VIDA base demand ───────────────────────────────────────────────────────────

def _vida_base_demand_kwh_year(row: pd.Series) -> float:
    """
    VIDA settlement-level demand at t=0 in kWh/year.
    Primary  : energy_demand (kWh/day) × 365
    Secondary: avg_connection_energy_demand × num_connections × 365
    """
    ed = row.get("energy_demand", None)
    if pd.notna(ed) and ed > 0:
        return float(ed) * 365.0

    avg_conn = row.get("avg_connection_energy_demand", None)
    num_conn = row.get("num_connections", None)
    if (pd.notna(avg_conn) and pd.notna(num_conn)
            and avg_conn > 0 and num_conn > 0):
        return float(avg_conn) * float(num_conn) * 365.0

    return 0.0


# ── Core demand functions ──────────────────────────────────────────────────────

def vida_demand_year(row: pd.Series, t: int) -> float:
    """
    Total kWh/year demand for a settlement at planning year t.

    Formula:
      D[t] = base_demand_kwh
             × intensity_factor(t)        ← per-HH income-driven growth
             × hh_growth_factor(t)        ← new households forming
             × tier_progression_factor(t) ← ramp-up for newly connected HH

    For electrified settlements:
      - intensity_factor(t) applies (existing HH use more over time)
      - hh_growth_factor does NOT apply (densification handled separately)
      - tier_progression = 1.0 (already have appliances)
    """
    base = _vida_base_demand_kwh_year(row)
    if base <= 0:
        return 0.0

    is_unelec = (row.get("elec_status", "unelectrified") == "unelectrified")

    # Intensity growth always applies
    demand_t = base * _intensity_factor(t)

    # HH count growth only for unelectrified settlements
    if is_unelec:
        demand_t *= _hh_growth_factor(t)

    # Tier progression ramp-up only for unelectrified settlements
    if is_unelec:
        tier_kwh = DEMAND["mtf_tiers"].get(
            f"tier_{row.get('mtf_tier', DEMAND.get('default_tier', 2))}",
            DEMAND["mtf_tiers"]["tier_2"]
        )
        demand_t *= _tier_progression_factor(t, tier_kwh, is_unelectrified=True)

    return demand_t


def vida_demand_timeseries(row: pd.Series) -> list:
    """
    Annual demand values (kWh/year) over the full planning horizon.
    Returns list of length (horizon + 1): [D_0, D_1, ..., D_T]
    """
    T = GENERAL["planning_horizon_years"]
    return [vida_demand_year(row, t) for t in range(T + 1)]


def num_households_at_year(row: pd.Series, t: int) -> float:
    """
    Projected number of households in a settlement at year t.

    For unelectrified settlements: grows at population_growth_rate × rural_unelec_share
    For electrified settlements:   stays fixed (densification tracked separately)
    """
    base_hh = float(row.get("num_connections", 1) or 1)
    is_unelec = (row.get("elec_status", "unelectrified") == "unelectrified")
    if is_unelec:
        return base_hh * _hh_growth_factor(t)
    return base_hh


# ── Public API ─────────────────────────────────────────────────────────────────

def add_demand_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich settlements GeoDataFrame with realistic demand estimates.

    Changes vs previous version:
      - demand_timeseries now uses dual growth rates (intensity + HH count)
      - demand includes tier progression ramp-up for newly connected settlements
      - num_households_y0 and num_households_yT show HH count evolution
      - shs_price_y0 and shs_price_yT reflect price decline curve

    New columns added:
      vida_demand_year0_kwh     — raw VIDA base demand at t=0 (kWh/yr)
      demand_year0_kwh          — modelled demand at t=0 (= vida for electrified)
      demand_yearT_kwh          — demand at end of horizon (t=15)
      demand_timeseries         — list [D_0 … D_T] for LCOE denominator
      demand_per_connection_kwh — demand_year0 / num_connections (reporting)
      num_households            — num_connections at t=0 (for CAPEX sizing)
      num_households_yT         — projected HH count at t=T (for investment sizing)
      hh_growth_total           — num_households_yT - num_households (new HH to connect)
      shs_price_t2_y0           — Tier-2 SHS price at base year (price decline applied)
      shs_price_t2_yT           — Tier-2 SHS price at end of horizon
    """
    df = df.copy()
    T = GENERAL["planning_horizon_years"]
    base_year = GENERAL["base_year"]

    # ── 1. Household counts ────────────────────────────────────────────────────
    df["num_households"] = df.get("num_connections", 1)

    df["num_households_yT"] = df.apply(
        lambda r: num_households_at_year(r, T), axis=1
    )
    df["hh_growth_total"] = (
        df["num_households_yT"] - df["num_households"]
    ).clip(lower=0).round(0)

    # ── 1b. Productive use uplift factor ──────────────────────────────────────
    # Apply to base demand before growth — anchor loads shift settlement above MG threshold
    if DEMAND.get('productive_use', {}).get('enabled', False):
        df['_pu_factor'] = df.apply(_productive_use_factor, axis=1)
    elif 'ProductiveUseFactor' in df.columns:
        df['_pu_factor'] = df['ProductiveUseFactor'].fillna(1.0)
    else:
        df['_pu_factor'] = 1.0

    # ── 2. VIDA base demand at t=0 ─────────────────────────────────────────────
    df["vida_demand_year0_kwh"] = df.apply(
        lambda r: _vida_base_demand_kwh_year(r) * r['_pu_factor'], axis=1
    )

    # ── 3. Modelled demand at t=0 and t=T ──────────────────────────────────────
    df["demand_year0_kwh"] = df.apply(lambda r: vida_demand_year(r, 0), axis=1)
    df["demand_yearT_kwh"] = df.apply(lambda r: vida_demand_year(r, T), axis=1)

    # ── 4. Full demand timeseries for LCOE NPV ─────────────────────────────────
    df["demand_timeseries"] = df.apply(vida_demand_timeseries, axis=1)

    # ── 5. Per-connection demand (for reporting) ───────────────────────────────
    df["demand_per_connection_kwh"] = df.apply(
        lambda r: r["demand_year0_kwh"] / max(
            float(r.get("num_connections", 1) or 1), 1
        ),
        axis=1
    )

    # ── 6. SHS price decline ───────────────────────────────────────────────────
    t2_base = SHS_CONFIG["tier_2"]["capex_per_unit"]
    df["shs_price_t2_y0"] = shs_price_at_year(t2_base, base_year)
    df["shs_price_t2_yT"] = shs_price_at_year(t2_base, base_year + T)

    # ── Summary ────────────────────────────────────────────────────────────────
    unelec = df[df.get("elec_status", pd.Series("unelectrified",
                        index=df.index)) == "unelectrified"]
    print("=== DEMAND MODEL SUMMARY (dual-rate) ===")
    print(f"  Intensity growth rate     : {DEMAND['demand_intensity_growth_rate']*100:.1f}%/yr (per-HH income-driven)")
    print(f"  Population growth rate    : {DEMAND['population_growth_rate']*100:.1f}%/yr × {DEMAND['rural_unelec_share']*100:.0f}% rural share")
    eff = (DEMAND["population_growth_rate"] * DEMAND["rural_unelec_share"]) * 100
    combined = ((1 + DEMAND["demand_intensity_growth_rate"]) *
                (1 + DEMAND["population_growth_rate"] * DEMAND["rural_unelec_share"]) - 1) * 100
    print(f"  Effective HH growth rate  : {eff:.2f}%/yr")
    print(f"  Combined demand growth    : {combined:.2f}%/yr (vs old single 4%)")
    print(f"  Tier progression          : ramp-up over {DEMAND.get('tier_progression',{}).get('years_to_target_tier',5)} years ✓")
    if len(unelec) > 0:
        print(f"  Unelec demand year 0      : {unelec['demand_year0_kwh'].sum()/1e6:.1f} GWh/yr")
        print(f"  Unelec demand year {T}     : {unelec['demand_yearT_kwh'].sum()/1e6:.1f} GWh/yr")
        print(f"  New HH from pop growth    : {unelec['hh_growth_total'].sum():,.0f} households by {GENERAL['base_year']+T}")
    print(f"  SHS Tier-2 price 2025     : ${df['shs_price_t2_y0'].iloc[0]:.0f}")
    print(f"  SHS Tier-2 price {base_year+T}    : ${df['shs_price_t2_yT'].iloc[0]:.0f}")

    return df


def shs_price_at_year(
    base_price: float,
    year: int,
    base_year: int = 2025,
    annual_decline: float = 0.05,
    floor_pct: float = 0.70,
) -> float:
    """
    SHS kit price at a given year with annual price decline.

    Parameters
    ----------
    base_price     : kit price at base_year (USD)
    year           : target year
    base_year      : reference year (default 2025)
    annual_decline : annual price decline rate (default 5%/yr)
    floor_pct      : minimum price as fraction of base_price (default 70%)

    Returns
    -------
    float — kit price at target year (USD)
    """
    years_elapsed = max(0, year - base_year)
    floor_price   = base_price * floor_pct
    price         = base_price * ((1 - annual_decline) ** years_elapsed)
    return max(price, floor_price)


# ── SHS price decline helper ───────────────────────────────────────────────────

def shs_price_at_year(
    base_price: float,
    year: int,
    base_year: int = 2025,
    annual_decline: float = 0.05,
    floor_pct: float = 0.70,
) -> float:
    """
    SHS kit price at a given calendar year with annual price decline.

    Parameters
    ----------
    base_price     : kit price at base_year (USD)
    year           : target calendar year
    base_year      : reference year (default 2025)
    annual_decline : annual price decline rate (default 5%/yr)
    floor_pct      : price floor as fraction of base_price (default 70%)

    Returns
    -------
    float : kit price at target year (USD)

    Source: GOGLA Global Off-Grid Solar Market Report H2 2023;
            BloombergNEF 2024 — 5%/yr conservative forward projection.
    """
    years_elapsed = max(0, year - base_year)
    floor_price   = base_price * floor_pct
    price         = base_price * ((1 - annual_decline) ** years_elapsed)
    return max(price, floor_price)
