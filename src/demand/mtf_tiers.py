"""
mtf_tiers.py
------------
Multi-Tier Framework (MTF) tier definitions and assignment logic.

Reference:
  Bhatia & Angelou (2015) — Beyond Connections: Energy Access Redefined.
  ESMAP/World Bank MTF for Household Energy Access.

Tier definitions (kWh/household/year):
  Tier 1 :   4.3   — very limited: basic lighting only
  Tier 2 :  73     — limited: lighting + phone charging + fan
  Tier 3 : 365     — medium: lighting + productive appliances
  Tier 4 : 1095    — high: medium appliances, fridge possible
  Tier 5 : 2737    — very high: all appliances

Assignment logic uses three proxy signals available in the settlement data:
  1. mean_rwi                       — Relative Wealth Index (Meta/Chi et al.)
  2. Medium_and_large_buildings_pc  — share of medium + large buildings
  3. has_health_facility /
     has_education_facility          — presence of social services (anchor loads)
"""

import pandas as pd
from src.config import DEMAND


# ── Tier demand lookup (kWh/HH/year) ──────────────────────────────────────────

TIER_KWH: dict[int, float] = {
    tier: DEMAND["mtf_tiers"][f"tier_{tier}"]
    for tier in range(1, 6)
}


def assign_tier(row: pd.Series) -> int:
    """
    Assign an MTF demand tier (1–4) to a single settlement row.

    Scoring system (0–5 points → mapped to tier 1–4):
      Wealth signal      (mean_rwi):                    0–2 pts
      Building mix       (Medium_and_large_buildings_pc): 0–2 pts
      Social services    (has_health/education):          0–1 pt
    ─────────────────────────────────────────────────────────────
    Total score  →  Tier
        0–1      →    1
        2        →    2
        3        →    3
        4–5      →    4
    """
    score = 0

    # 1. Wealth signal ─────────────────────────────────────────────────────────
    rwi = row.get("mean_rwi", None)
    if pd.notna(rwi):
        high_rwi  = DEMAND["rwi_thresholds"]["high"]
        med_rwi   = DEMAND["rwi_thresholds"]["medium"]
        if rwi >= high_rwi:
            score += 2
        elif rwi >= med_rwi:
            score += 1

    # 2. Building size mix ─────────────────────────────────────────────────────
    bld_mix = row.get("Medium_and_large_buildings_pc", None)
    if pd.notna(bld_mix):
        high_bld = DEMAND["building_mix_thresholds"]["high"]
        med_bld  = DEMAND["building_mix_thresholds"]["medium"]
        if bld_mix >= high_bld:
            score += 2
        elif bld_mix >= med_bld:
            score += 1

    # 3. Social services ───────────────────────────────────────────────────────
    has_health = bool(row.get("has_health_facility", False))
    has_edu    = bool(row.get("has_education_facility", False))
    if has_health or has_edu:
        score += 1

    # Map score to tier
    if score <= 1:
        return 1
    elif score == 2:
        return 2
    elif score == 3:
        return 3
    else:
        return 4


def assign_tiers_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply assign_tier to all rows and add 'mtf_tier' column."""
    df = df.copy()
    df["mtf_tier"] = df.apply(assign_tier, axis=1)
    return df


def get_tier_demand(tier: int) -> float:
    """Return kWh/household/year for a given tier."""
    return TIER_KWH.get(tier, TIER_KWH[2])


def tier_summary() -> pd.DataFrame:
    """Return a human-readable summary of tier definitions."""
    rows = [
        {"Tier": 1, "kWh/HH/year":    4.3, "Description": "Basic lighting only"},
        {"Tier": 2, "kWh/HH/year":   73.0, "Description": "Lighting + phone charging + fan"},
        {"Tier": 3, "kWh/HH/year":  365.0, "Description": "Lighting + productive appliances"},
        {"Tier": 4, "kWh/HH/year": 1095.0, "Description": "Medium appliances, possible fridge"},
        {"Tier": 5, "kWh/HH/year": 2737.0, "Description": "Full appliance use"},
    ]
    return pd.DataFrame(rows)
