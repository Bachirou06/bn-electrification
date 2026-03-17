"""
grid_extension.py
-----------------
CAPEX, OPEX, LCOE and grid penalty for grid extension.

Grid Penalty (from GEP-OnSSET)
================================
OnSSET applies a penalty multiplier to grid CAPEX based on five factors
that increase construction difficulty and cost:

  Factor            Weight   Scoring (1=worst, 5=best)
  ─────────────────────────────────────────────────────
  Slope             0.30     ≤10° → 5, ≤20° → 4, ≤30° → 3, ≤40° → 2, >40° → 1
  Road distance     0.15     ≤5km → 5, ≤10 → 4, ≤25 → 3, ≤50 → 2, >50 → 1
  Substation dist   0.20     ≤0.5km → 5, ≤1 → 4, ≤5 → 3, ≤10 → 2, >10 → 1
  Land cover        0.20     IGBP-based (water/urban low, cropland/forest high)
  Elevation         0.15     ≤500m → 5, ≤1000 → 4, ≤2000 → 3, ≤3000 → 2, >3000 → 1

Combined score (weighted average) → penalty multiplier:
  penalty = 1 + (exp(0.85 × |1 - score|) - 1) / 100

Slope adjustment for MV line cost (Benin-specific)
====================================================
  Slope < 2°  → $40,000/km  (flat coastal south)
  Slope 2-5°  → $45,000/km  (central plateaux)
  Slope > 5°  → $55,000/km  (Atacora mountains, north)
Source: ABERME (2022). Couts Unitaires des Infrastructures Electriques Rurales.
"""

import numpy as np
import pandas as pd
from math import exp

from src.config import GRID, GENERAL, DISCOUNT_RATE, HORIZON
from src.costs.lcoe_calculator import compute_lcoe, compute_npc


# ── Grid penalty — exact OnSSET implementation ────────────────────────────────

def _classify_road_dist(road_dist_km: float) -> float:
    if road_dist_km <= 5:   return 5
    elif road_dist_km <= 10: return 4
    elif road_dist_km <= 25: return 3
    elif road_dist_km <= 50: return 2
    else:                    return 1

def _classify_substation_dist(sub_dist_km: float) -> float:
    if sub_dist_km <= 0.5:  return 5
    elif sub_dist_km <= 1:  return 4
    elif sub_dist_km <= 5:  return 3
    elif sub_dist_km <= 10: return 2
    else:                   return 1

def _classify_land_cover(igbp_class: int) -> float:
    """IGBP land cover class → construction difficulty score."""
    lc_map = {
        0: 1,   # no data
        1: 3,   # evergreen needleleaf forest
        2: 4,   # evergreen broadleaf forest
        3: 3,   # deciduous needleleaf
        4: 4,   # deciduous broadleaf
        5: 3,   # mixed forest
        6: 2,   # closed shrubland
        7: 5,   # open shrubland
        8: 2,   # woody savanna
        9: 5,   # savanna
        10: 5,  # grassland
        11: 1,  # permanent wetland (difficult)
        12: 3,  # cropland
        13: 3,  # urban
        14: 5,  # cropland/natural mosaic
        15: 3,  # snow/ice
        16: 5,  # barren
    }
    return lc_map.get(int(igbp_class), 3)

def _classify_elevation(elevation_m: float) -> float:
    if elevation_m <= 500:   return 5
    elif elevation_m <= 1000: return 4
    elif elevation_m <= 2000: return 3
    elif elevation_m <= 3000: return 2
    else:                     return 1

def _classify_slope(slope_deg: float) -> float:
    if slope_deg <= 10:  return 5
    elif slope_deg <= 20: return 4
    elif slope_deg <= 30: return 3
    elif slope_deg <= 40: return 2
    else:                 return 1

def compute_grid_penalty(
    slope_deg: float,
    road_dist_km: float,
    substation_dist_km: float,
    land_cover: int,
    elevation_m: float,
) -> float:
    """
    Grid penalty multiplier — exact OnSSET formula.

    Returns a value >= 1.0. Higher penalty = more expensive grid construction.
    Typical range: 1.00 (flat, near road) to ~1.07 (steep, remote, difficult terrain).
    """
    score = (
        0.30 * _classify_slope(slope_deg) +
        0.15 * _classify_road_dist(road_dist_km) +
        0.20 * _classify_substation_dist(substation_dist_km) +
        0.20 * _classify_land_cover(land_cover) +
        0.15 * _classify_elevation(elevation_m)
    )
    return 1 + (exp(0.85 * abs(1 - score)) - 1) / 100


# ── Slope-adjusted MV line cost (Benin-specific) ──────────────────────────────

def _mv_cost_per_km(slope_deg: float) -> float:
    """
    Terrain-adjusted MV line cost based on SRTM slope.
    Source: ABERME (2022). Couts Unitaires des Infrastructures Electriques Rurales.
    """
    if slope_deg < 2:   return 40_000   # flat coastal south
    elif slope_deg < 5: return 45_000   # central plateaux
    else:               return 55_000   # Atacora mountains, northern escarpments


# ── Core CAPEX and LCOE functions ─────────────────────────────────────────────

def grid_capex(
    distance_km: float,
    num_households: float,
    slope_deg: float = 2.0,
    road_dist_km: float = 10.0,
    substation_dist_km: float = 5.0,
    land_cover: int = 14,
    elevation_m: float = 300.0,
) -> tuple:
    """
    Grid extension CAPEX with slope-adjusted line cost and OnSSET penalty.

    Returns
    -------
    (capex_usd, penalty, mv_cost_per_km)
    """
    mv_cost   = _mv_cost_per_km(slope_deg)
    penalty   = compute_grid_penalty(
        slope_deg, road_dist_km, substation_dist_km, land_cover, elevation_m
    )
    line_cost = mv_cost * distance_km * penalty
    conn_cost = GRID["capex_per_connection"] * num_households
    capex     = line_cost + conn_cost
    return capex, penalty, mv_cost


def grid_lcoe(
    distance_km: float,
    num_households: float,
    demand_timeseries: list,
    slope_deg: float = 2.0,
    road_dist_km: float = 10.0,
    substation_dist_km: float = 5.0,
    land_cover: int = 14,
    elevation_m: float = 300.0,
) -> float:
    """
    LCOE for grid extension [USD/kWh] with terrain penalty.

    Returns np.inf if infeasible (distance > max_viable_distance_km).
    """
    if distance_km > GRID["max_viable_distance_km"] or distance_km <= 0:
        return np.inf

    capex, _, _ = grid_capex(
        distance_km, num_households,
        slope_deg, road_dist_km, substation_dist_km, land_cover, elevation_m
    )
    annual_opex = capex * GRID["opex_rate"]
    loss_factor = 1 - GRID["loss_rate"]
    energy_net  = [e * loss_factor for e in demand_timeseries]

    return compute_lcoe(
        capex=capex,
        annual_opex=annual_opex,
        energy_series=energy_net,
        discount_rate=DISCOUNT_RATE,
        lifetime_years=GRID["lifetime_years"],
    )


# ── DataFrame-level function ──────────────────────────────────────────────────

def add_grid_lcoe(
    df: pd.DataFrame,
    dist_col: str = "GridDistKm",
) -> pd.DataFrame:
    """
    Add grid penalty, slope-adjusted MV cost, and LCOE columns to settlements.

    New columns added:
      grid_penalty        — OnSSET penalty multiplier (>= 1.0)
      mv_cost_per_km      — slope-adjusted $/km
      grid_capex_usd      — total grid CAPEX including penalty
      lcoe_grid           — LCOE [USD/kWh]
      grid_feasible       — True if distance <= max_viable_distance_km
    """
    df = df.copy()

    def _row_penalty(r):
        return compute_grid_penalty(
            slope_deg          = r.get('Slope', 2.0) or 2.0,
            road_dist_km       = r.get('dist_road_km', 10.0) or 10.0,
            substation_dist_km = r.get('DistSubstation', 5.0) or 5.0,
            land_cover         = int(r.get('LandCover', 14) or 14),
            elevation_m        = r.get('Elevation', 300.0) or 300.0,
        )

    def _row_mv_cost(r):
        return _mv_cost_per_km(r.get('Slope', 2.0) or 2.0)

    def _row_capex(r):
        capex, _, _ = grid_capex(
            distance_km        = r[dist_col],
            num_households     = r.get('num_households', 1) or 1,
            slope_deg          = r.get('Slope', 2.0) or 2.0,
            road_dist_km       = r.get('dist_road_km', 10.0) or 10.0,
            substation_dist_km = r.get('DistSubstation', 5.0) or 5.0,
            land_cover         = int(r.get('LandCover', 14) or 14),
            elevation_m        = r.get('Elevation', 300.0) or 300.0,
        )
        return capex

    def _row_lcoe(r):
        return grid_lcoe(
            distance_km        = r[dist_col],
            num_households     = r.get('num_households', 1) or 1,
            demand_timeseries  = r.get('demand_timeseries', [0]),
            slope_deg          = r.get('Slope', 2.0) or 2.0,
            road_dist_km       = r.get('dist_road_km', 10.0) or 10.0,
            substation_dist_km = r.get('DistSubstation', 5.0) or 5.0,
            land_cover         = int(r.get('LandCover', 14) or 14),
            elevation_m        = r.get('Elevation', 300.0) or 300.0,
        )

    print('Computing grid penalty...')
    df['grid_penalty']   = df.apply(_row_penalty, axis=1)
    df['mv_cost_per_km'] = df.apply(_row_mv_cost, axis=1)
    df['grid_capex_usd'] = df.apply(_row_capex, axis=1)

    print('Computing grid LCOE...')
    df['lcoe_grid']      = df.apply(_row_lcoe, axis=1)
    df['grid_feasible']  = df[dist_col] <= GRID['max_viable_distance_km']

    print(f'Grid penalty stats:')
    print(f'  min={df["grid_penalty"].min():.4f}  '
          f'mean={df["grid_penalty"].mean():.4f}  '
          f'max={df["grid_penalty"].max():.4f}')
    print(f'MV cost/km distribution:')
    print(df['mv_cost_per_km'].value_counts().sort_index().to_string())
    print(f'Grid feasible: {df["grid_feasible"].sum():,} / {len(df):,} settlements')

    return df


# ── Grid Densification ────────────────────────────────────────────────────────

# Parameters (ABERME sources)
TARGET_CONNECTION_RATE = 0.90    # ABERME Plan Directeur 2022 target
LV_COST_PER_KM        = 8_000   # USD/km — ABERME LV 400V unit cost
LV_LENGTH_PER_HH      = 0.030   # km — 30m LV per new connection (OnSSET cluster model)
DENSIF_COST_PER_HH    = LV_COST_PER_KM * LV_LENGTH_PER_HH + 400  # $240 LV + $400 conn = $640


def densification_capex(hh_to_connect: float) -> float:
    """
    Total densification CAPEX [USD].

    = (LV infrastructure + connection cost) × HH to connect
    = ($240 + $400) × HH_to_connect
    = $640 × HH_to_connect

    Source: ABERME (2022) LV unit costs + OnSSET cluster model (30m LV/HH)
    """
    return DENSIF_COST_PER_HH * max(hh_to_connect, 0)


def densification_lcoe(
    hh_to_connect: float,
    demand_per_connection_kwh: float,
) -> float:
    """
    LCOE for grid densification [USD/kWh].

    Parameters
    ----------
    hh_to_connect             : households to connect to reach 90% target
    demand_per_connection_kwh : VIDA demand per connection at year 0 (kWh/year)

    Returns np.nan if no households to connect or no demand.
    """
    if hh_to_connect <= 0 or demand_per_connection_kwh <= 0:
        return np.nan

    capex       = densification_capex(hh_to_connect)
    annual_opex = capex * GRID["opex_rate"]
    annual_energy = (
        hh_to_connect * demand_per_connection_kwh * (1 - GRID["loss_rate"])
    )

    return compute_lcoe(
        capex          = capex,
        annual_opex    = annual_opex,
        energy_series  = [annual_energy] * GRID["lifetime_years"],
        discount_rate  = DISCOUNT_RATE,
        lifetime_years = GRID["lifetime_years"],
    )


def add_densification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute grid densification gap and cost for electrified settlements.

    Only applied to settlements where elec_status == 'electrified'
    and connection_rate < TARGET_CONNECTION_RATE (90%).

    New columns added:
      connection_rate        — num_connections / num_buildings
      HH_to_connect          — households needed to reach 90% target
      densification_needed   — True if HH_to_connect > 0
      densification_capex    — total CAPEX [USD]
      lcoe_densification     — LCOE [USD/kWh]
    """
    df = df.copy()

    # Connection rate
    if "num_buildings" in df.columns and df["num_buildings"].sum() > 0:
        df["connection_rate"] = (
            df["num_connections"] / df["num_buildings"].replace(0, np.nan)
        ).clip(0, 1).fillna(0)
    else:
        df["connection_rate"] = 0.0
        print("⚠ num_buildings not found — connection_rate set to 0")

    # HH to connect
    buildings = df.get("num_buildings", df["num_connections"])
    df["HH_to_connect"] = (
        (TARGET_CONNECTION_RATE - df["connection_rate"])
        .clip(lower=0) * buildings.fillna(df["num_connections"])
    ).round(0)

    # Only applies to electrified settlements
    df["densification_needed"] = (
        (df["elec_status"] == "electrified") & (df["HH_to_connect"] > 0)
    )

    # CAPEX
    df["densification_capex"] = np.where(
        df["densification_needed"],
        df["HH_to_connect"].apply(densification_capex),
        0.0
    )

    # LCOE
    def _lcoe(r):
        if not r["densification_needed"]:
            return np.nan
        demand_per_conn = r.get("demand_per_connection_kwh") or (
            r["demand_year0_kwh"] / max(r.get("num_connections", 1) or 1, 1)
        )
        return densification_lcoe(r["HH_to_connect"], demand_per_conn)

    df["lcoe_densification"] = df.apply(_lcoe, axis=1)

    # Summary
    dens = df[df["densification_needed"]]
    print(f"=== GRID DENSIFICATION ===")
    print(f"  Electrified settlements     : {(df['elec_status']=='electrified').sum():,}")
    print(f"  Densification needed        : {dens.shape[0]:,}")
    print(f"  Total HH to connect         : {dens['HH_to_connect'].sum():,.0f}")
    print(f"  Total CAPEX                 : ${dens['densification_capex'].sum()/1e6:.1f}M")
    if len(dens) > 0:
        med = df["lcoe_densification"].dropna().median()
        print(f"  Median LCOE                 : ${med:.3f}/kWh")

    return df
