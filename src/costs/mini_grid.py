"""
mini_grid.py — Mini-grid LCOE for Benin electrification model.

Three sub-types:
  mg_solar  : Solar PV + battery. Min 20 connections.
  mg_hybrid : Solar + diesel. Min 50 connections, road <= 50km.
              Diesel price spatially varying from DieselPrice column (NB00).
  mg_hydro  : Run-of-river hydro. Min 30 connections, river <= 5km, head >= 2m.

All LCOEs use VIDA demand_timeseries as energy denominator.
"""

import ast
import numpy as np
import pandas as pd

from src.config import DISCOUNT_RATE
from src.costs.lcoe_calculator import compute_lcoe


# ── Sub-type parameters ───────────────────────────────────────────────────────

MG_SUBTYPES = {
    "mg_solar": {
        "label":               "Mini-Grid: Solar PV Only",
        "capex_per_kw":        2500,
        "battery_cost_kwh":    270,
        "battery_hours":       6,
        "battery_replace_yr":  10,
        "capex_per_conn":      280,
        "opex_rate":           0.025,
        "loss_rate":           0.08,
        "lifetime_years":      20,
        # Demand threshold — ESMAP (2019): min 3-5 kW = ~15,000 kWh/yr
        "min_demand_kwh":      15_000,
        # Operational floor — absolute minimum customers to run a mini-grid
        "min_connections":     10,
        "max_road_dist_km":    9999,
        "max_river_dist_km":   9999,
        "peak_to_base_ratio":  3.5,
        "diesel_fraction":     0.0,
        "diesel_capex_per_kw": 0,
        "diesel_ltr_per_kwh":  0.0,
        "diesel_replace_yr":   999,
    },
    "mg_hybrid": {
        "label":               "Mini-Grid: Solar-Diesel Hybrid",
        "capex_per_kw":        2500,
        "battery_cost_kwh":    250,
        "battery_hours":       4,
        "battery_replace_yr":  10,
        "capex_per_conn":      300,
        "opex_rate":           0.03,
        "loss_rate":           0.08,
        "lifetime_years":      20,
        # Demand threshold — ESMAP (2019): min 5-10 kW = ~25,000 kWh/yr
        # Higher than solar because diesel logistics requires larger revenue base
        "min_demand_kwh":      25_000,
        "min_connections":     10,
        "max_road_dist_km":    50,
        "max_river_dist_km":   9999,
        "peak_to_base_ratio":  2.5,
        "diesel_fraction":     0.35,
        "diesel_capex_per_kw": 500,
        "diesel_ltr_per_kwh":  0.30,
        "diesel_replace_yr":   10,
    },
    "mg_hydro": {
        "label":               "Mini-Grid: Mini-Hydro",
        "capex_per_kw":        4500,
        "battery_cost_kwh":    0,
        "battery_hours":       0,
        "battery_replace_yr":  999,
        "capex_per_conn":      350,
        "opex_rate":           0.02,
        "loss_rate":           0.05,
        "lifetime_years":      40,
        # Demand threshold — IRENA (2016): small run-of-river min 2-5 kW = ~10,000 kWh/yr
        # Lower than solar because hydro provides firm 24/7 baseload power
        "min_demand_kwh":      10_000,
        "min_connections":     10,
        "max_road_dist_km":    9999,
        "max_river_dist_km":   5,
        "peak_to_base_ratio":  2.0,
        "diesel_fraction":     0.0,
        "diesel_capex_per_kw": 0,
        "diesel_ltr_per_kwh":  0.0,
        "diesel_replace_yr":   999,
    },
}


# ── Safe demand_timeseries conversion ─────────────────────────────────────────

def _to_list(ts):
    """Convert any demand_timeseries type to plain Python float list."""
    if ts is None:
        return []
    if isinstance(ts, str):
        try:
            ts = ast.literal_eval(ts)
        except Exception:
            return []
    try:
        return [float(x) for x in list(ts)]
    except Exception:
        return []


# ── Hydro power ───────────────────────────────────────────────────────────────

def hydro_available_power_kw(head_m, discharge_m3s, efficiency=0.85):
    """P = η × ρ × g × H × Q / 1000"""
    if head_m <= 0 or discharge_m3s <= 0:
        return 0.0
    return efficiency * 1000 * 9.81 * head_m * discharge_m3s / 1000


# ── Peak kW sizing ────────────────────────────────────────────────────────────

def peak_kw_from_demand(annual_kwh, subtype):
    avg_load_kw = annual_kwh / 8760.0
    return avg_load_kw * MG_SUBTYPES[subtype]["peak_to_base_ratio"]


# ── CAPEX ─────────────────────────────────────────────────────────────────────

def mg_capex(num_households, subtype, annual_kwh=None):
    p = MG_SUBTYPES[subtype]
    peak_kw = peak_kw_from_demand(annual_kwh, subtype) if annual_kwh and annual_kwh > 0 \
              else num_households * 0.25

    battery      = peak_kw * p["battery_hours"] * p["battery_cost_kwh"]
    solar        = peak_kw * p["capex_per_kw"]
    conn         = num_households * p["capex_per_conn"]
    diesel_kw    = peak_kw * p["diesel_fraction"]
    diesel_capex = diesel_kw * p["diesel_capex_per_kw"]
    total        = solar + battery + conn + diesel_capex

    return {
        "solar_capex":     solar,
        "battery_capex":   battery,
        "diesel_capex":    diesel_capex,
        "connection_capex":conn,
        "total_capex":     total,
        "peak_kw":         peak_kw,
        "diesel_kw":       diesel_kw,
        "sizing_method":   "vida" if (annual_kwh and annual_kwh > 0) else "fallback",
    }


# ── Feasibility ───────────────────────────────────────────────────────────────

def mg_is_feasible(num_connections, dist_road_km, dist_river_km, subtype,
                   head_m=0, discharge_m3s=0, peak_kw=0, demand_kwh=0):
    """
    Feasibility check combining demand threshold (primary) and
    connection count (secondary operational floor).

    Demand threshold source:
      Solar MG : >= 15,000 kWh/yr  (ESMAP 2019: min 3-5 kW system)
      Hybrid   : >= 25,000 kWh/yr  (ESMAP 2019: min 5-10 kW + diesel logistics)
      Hydro    : >= 10,000 kWh/yr  (IRENA 2016: small run-of-river min 2-5 kW)
    """
    p = MG_SUBTYPES[subtype]
    # Primary: minimum annual demand (economic viability — ESMAP/IRENA)
    if demand_kwh < p["min_demand_kwh"]:              return False
    # Secondary: minimum connections (operational floor)
    if num_connections < p["min_connections"]:        return False
    # Road and river constraints
    if dist_road_km  > p["max_road_dist_km"]:         return False
    if dist_river_km > p["max_river_dist_km"]:        return False
    if subtype == "mg_hydro":
        # Head >= 5m: minimum for economic micro-hydro in West Africa
        # (2m gives near-zero head pressure, not viable for run-of-river turbines)
        # Source: IRENA (2016); typical Benin project sites: 5-20m head
        if head_m < 5.0:                              return False
        # Discharge 0.1-50 m3/s: village-scale rivers only
        # > 50 m3/s = major river (Oueme, Mono, Pendjari) — not village micro-hydro
        # < 0.1 m3/s = too small to drive a turbine
        if discharge_m3s < 0.10:                      return False
        if discharge_m3s > 50.0:                      return False
        avail = hydro_available_power_kw(head_m, discharge_m3s)
        if peak_kw > 0 and avail < peak_kw * 0.5:    return False
    return True


# ── LCOE ─────────────────────────────────────────────────────────────────────

def mg_lcoe(num_households, num_connections, dist_road_km, dist_river_km,
            demand_timeseries, subtype, diesel_price_usd_l=0.85,
            head_m=0.0, discharge_m3s=0.0):
    """LCOE for a mini-grid sub-type [USD/kWh]. Returns np.inf if infeasible."""
    p  = MG_SUBTYPES[subtype]
    ts = _to_list(demand_timeseries)
    annual_kwh_y0 = ts[0] if ts else 0.0
    peak_kw = peak_kw_from_demand(annual_kwh_y0, subtype) if annual_kwh_y0 > 0 else 0.0

    if not mg_is_feasible(num_connections, dist_road_km, dist_river_km,
                          subtype, head_m, discharge_m3s, peak_kw,
                          demand_kwh=annual_kwh_y0):
        return np.inf

    cap = mg_capex(num_households, subtype, annual_kwh=annual_kwh_y0)
    annual_fixed_opex = cap["total_capex"] * p["opex_rate"]

    # Diesel fuel cost (hybrid only, spatially varying price)
    if p["diesel_fraction"] > 0 and annual_kwh_y0 > 0:
        annual_diesel_kwh  = annual_kwh_y0 * p["diesel_fraction"]
        annual_fuel_cost   = annual_diesel_kwh * p["diesel_ltr_per_kwh"] * diesel_price_usd_l
    else:
        annual_fuel_cost = 0.0

    annual_opex = annual_fixed_opex + annual_fuel_cost

    # Replacement costs
    replacements = {}
    if p["battery_replace_yr"] < p["lifetime_years"] and p["battery_cost_kwh"] > 0:
        replacements[p["battery_replace_yr"]] = cap["battery_capex"]
    if p.get("diesel_replace_yr", 999) < p["lifetime_years"] and cap["diesel_capex"] > 0:
        replacements[p["diesel_replace_yr"]] = cap["diesel_capex"]

    # Energy net of losses — extend to lifetime
    energy_net = [e * (1 - p["loss_rate"]) for e in ts]
    lifetime   = p["lifetime_years"]
    if lifetime > len(energy_net):
        last = energy_net[-1] if energy_net else 0.0
        energy_net += [last] * (lifetime - len(energy_net))

    return compute_lcoe(
        capex=cap["total_capex"],
        annual_opex=annual_opex,
        energy_series=energy_net,
        discount_rate=DISCOUNT_RATE,
        lifetime_years=lifetime,
        replacement_costs=replacements,
    )


# ── DataFrame-level ───────────────────────────────────────────────────────────

def add_minigrid_lcoe(df: pd.DataFrame) -> pd.DataFrame:
    """Compute LCOE for all three mini-grid sub-types per settlement."""
    df = df.copy()

    df["_road"]    = df["dist_road_km"].fillna(0)    if "dist_road_km"    in df.columns else 0.0
    df["_river"]   = df["dist_river_km"].fillna(999) if "dist_river_km"   in df.columns else 999.0
    df["_head"]    = df["HydroHead"].fillna(0)       if "HydroHead"       in df.columns else 0.0
    df["_discharge"]= df["HydroDischarge"].fillna(0) if "HydroDischarge"  in df.columns else 0.0
    df["_diesel"]  = df["DieselPrice"].fillna(0.85)  if "DieselPrice"     in df.columns else 0.85

    if "DieselPrice" not in df.columns:
        import warnings
        warnings.warn("DieselPrice not found — using flat $0.85/L", UserWarning, stacklevel=2)

    for st in MG_SUBTYPES:
        df[f"lcoe_{st}"] = df.apply(
            lambda r, s=st: mg_lcoe(
                num_households     = float(r.get("num_households", 1) or 1),
                num_connections    = float(r.get("num_connections", 1) or 1),
                dist_road_km       = float(r["_road"]),
                dist_river_km      = float(r["_river"]),
                demand_timeseries  = r.get("demand_timeseries"),
                subtype            = s,
                diesel_price_usd_l = float(r["_diesel"]),
                head_m             = float(r["_head"]),
                discharge_m3s      = float(r["_discharge"]),
            ),
            axis=1,
        )

    mg_cols  = [f"lcoe_{st}" for st in MG_SUBTYPES]
    mat      = df[mg_cols].values
    best_idx = np.argmin(mat, axis=1)

    df["lcoe_minigrid"]          = mat[np.arange(len(df)), best_idx]
    df["minigrid_subtype"]       = [list(MG_SUBTYPES.keys())[i] for i in best_idx]
    df["minigrid_subtype_label"] = df["minigrid_subtype"].map(
        {st: MG_SUBTYPES[st]["label"] for st in MG_SUBTYPES}
    )

    def _cap(row):
        ts  = _to_list(row.get("demand_timeseries"))
        kwh = ts[0] if ts else None
        c   = mg_capex(float(row.get("num_households", 1) or 1),
                       row["minigrid_subtype"], annual_kwh=kwh)
        return c["total_capex"], c["peak_kw"]

    cp = df.apply(_cap, axis=1, result_type="expand")
    df["minigrid_capex_usd"] = cp[0]
    df["minigrid_peak_kw"]   = cp[1]
    df["minigrid_feasible"]  = df["lcoe_minigrid"] < np.inf
    df["mg_hydro_feasible"]  = df["lcoe_mg_hydro"] < np.inf

    df = df.drop(columns=["_road","_river","_head","_discharge","_diesel"], errors="ignore")

    print("=== MINI-GRID LCOE SUMMARY ===")
    for st in MG_SUBTYPES:
        col  = f"lcoe_{st}"
        fin  = df[col].replace(np.inf, np.nan).dropna()
        feas = (df[col] < np.inf).sum()
        med  = f"{fin.median():.3f}" if len(fin) > 0 else "n/a"
        print(f"  {MG_SUBTYPES[st]['label']:<35}: median=${med}  feasible={feas:,} ({feas/len(df)*100:.0f}%)")

    return df
