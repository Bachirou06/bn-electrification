"""
mini_grid.py
------------
LCOE for THREE mini-grid sub-types modelled for Benin.

Sub-type 1 — Solar PV Only (mg_solar)
  Fully off-grid PV + battery. No fuel dependency.
  Peak-to-base ratio: 3.5 (concentrated evening residential peak)
  CAPEX: $2,500/kWp | Battery: 6h | Min connections: 20

Sub-type 2 — Solar-Diesel Hybrid (mg_hybrid)
  PV + diesel genset for peak shaving and backup.
  Diesel price is settlement-specific from OnSSET logistics model:
    DieselPrice = 0.85 + 0.00382 × TravelHours  (USD/litre)
  Fuel consumption: 0.3 L/kWh (standard diesel generator efficiency)
  Peak-to-base ratio: 2.5 (diesel smooths evening peak)
  Solar fraction: 65% of annual energy
  CAPEX: $2,500/kWp solar + $500/kW diesel | Min connections: 50

Sub-type 3 — Mini-Hydro (mg_hydro)
  Run-of-river small hydro. Zero fuel cost, 40-year lifetime.
  Feasibility: dist_river_km ≤ 5 AND HydroHead ≥ 2m
  Available power: P = η × ρ × g × Head × Discharge / 1000
  Peak-to-base ratio: 2.0 (continuous baseload)
  CAPEX: $4,500/kW | Min connections: 30

All LCOEs use VIDA demand_timeseries as energy denominator.
"""

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
        "min_connections":     20,
        "max_road_dist_km":    9999,
        "max_river_dist_km":   9999,
        "peak_to_base_ratio":  3.5,
        # Diesel parameters — not used for solar
        "diesel_fraction":     0.0,
        "diesel_capex_per_kw": 0,
        "diesel_ltr_per_kwh":  0.0,
        "diesel_replace_yr":   999,
    },
    "mg_hybrid": {
        "label":               "Mini-Grid: Solar-Diesel Hybrid",
        # Solar component
        "capex_per_kw":        2500,     # USD/kWp solar PV
        "battery_cost_kwh":    250,
        "battery_hours":       4,
        "battery_replace_yr":  10,
        "capex_per_conn":      300,
        "opex_rate":           0.03,
        "loss_rate":           0.08,
        "lifetime_years":      20,
        "min_connections":     50,
        "max_road_dist_km":    50,       # fuel delivery constraint
        "max_river_dist_km":   9999,
        "peak_to_base_ratio":  2.5,
        # Diesel component
        "diesel_fraction":     0.35,     # 35% of annual energy from diesel
        "diesel_capex_per_kw": 500,      # USD/kW generator CAPEX
        "diesel_ltr_per_kwh":  0.30,     # litres per kWh (standard gen efficiency)
        "diesel_replace_yr":   10,       # generator replacement at year 10
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
        "min_connections":     30,
        "max_road_dist_km":    9999,
        "max_river_dist_km":   5,
        "peak_to_base_ratio":  2.0,
        # Diesel parameters — not used for hydro
        "diesel_fraction":     0.0,
        "diesel_capex_per_kw": 0,
        "diesel_ltr_per_kwh":  0.0,
        "diesel_replace_yr":   999,
    },
}


# ── Mini-hydro feasibility and power ─────────────────────────────────────────

def hydro_available_power_kw(head_m: float, discharge_m3s: float,
                              efficiency: float = 0.85) -> float:
    """
    Available hydraulic power in kW.
    P = η × ρ × g × Head × Discharge / 1000
    """
    if head_m <= 0 or discharge_m3s <= 0:
        return 0.0
    return efficiency * 1000 * 9.81 * head_m * discharge_m3s / 1000


# ── Peak kW sizing from VIDA demand ──────────────────────────────────────────

def peak_kw_from_demand(annual_kwh: float, subtype: str) -> float:
    """
    Peak system capacity [kW] from VIDA annual demand.
    avg_load_kW = annual_kWh / 8760
    peak_kW     = avg_load_kW × peak_to_base_ratio
    """
    avg_load_kw = annual_kwh / 8760.0
    return avg_load_kw * MG_SUBTYPES[subtype]["peak_to_base_ratio"]


# ── CAPEX breakdown ───────────────────────────────────────────────────────────

def mg_capex(num_households: float, subtype: str,
             annual_kwh: float = None) -> dict:
    """
    CAPEX breakdown for a mini-grid sub-type [USD].

    For hybrid: includes both solar PV and diesel generator CAPEX.
    Diesel generator sized to cover peak fraction (35% of peak load).
    """
    p = MG_SUBTYPES[subtype]

    if annual_kwh is not None and annual_kwh > 0:
        peak_kw = peak_kw_from_demand(annual_kwh, subtype)
    else:
        peak_kw = num_households * 0.25  # fallback

    battery = peak_kw * p["battery_hours"] * p["battery_cost_kwh"]
    solar   = peak_kw * p["capex_per_kw"]
    conn    = num_households * p["capex_per_conn"]

    # Diesel generator CAPEX (hybrid only)
    diesel_kw    = peak_kw * p["diesel_fraction"]
    diesel_capex = diesel_kw * p["diesel_capex_per_kw"]

    total = solar + battery + conn + diesel_capex

    return {
        "solar_capex":    solar,
        "battery_capex":  battery,
        "diesel_capex":   diesel_capex,
        "connection_capex": conn,
        "total_capex":    total,
        "peak_kw":        peak_kw,
        "diesel_kw":      diesel_kw,
        "sizing_method":  "vida_peak_to_base" if (annual_kwh and annual_kwh > 0) else "fallback",
    }


# ── Feasibility check ─────────────────────────────────────────────────────────

def mg_is_feasible(num_connections: float, dist_road_km: float,
                   dist_river_km: float, subtype: str,
                   head_m: float = 0, discharge_m3s: float = 0,
                   peak_kw: float = 0) -> bool:
    """
    Feasibility check for a mini-grid sub-type.

    mg_solar : connections >= 20
    mg_hybrid: connections >= 50 AND road <= 50 km
    mg_hydro : connections >= 30 AND river <= 5 km
               AND HydroHead >= 2m AND available_power >= 50% of peak
    """
    p = MG_SUBTYPES[subtype]
    if num_connections < p["min_connections"]:
        return False
    if dist_road_km > p["max_road_dist_km"]:
        return False
    if dist_river_km > p["max_river_dist_km"]:
        return False
    # Extra hydro checks
    if subtype == "mg_hydro":
        if head_m < 2.0:
            return False
        if discharge_m3s < 0.05:
            return False
        avail = hydro_available_power_kw(head_m, discharge_m3s)
        if peak_kw > 0 and avail < peak_kw * 0.5:
            return False
    return True


# ── LCOE calculation ──────────────────────────────────────────────────────────

def mg_lcoe(
    num_households: float,
    num_connections: float,
    dist_road_km: float,
    dist_river_km: float,
    demand_timeseries: list,
    subtype: str,
    diesel_price_usd_l: float = 0.85,
    head_m: float = 0.0,
    discharge_m3s: float = 0.0,
) -> float:
    """
    LCOE for a mini-grid sub-type [USD/kWh].

    Key change from standard OnSSET:
    - mg_hybrid uses settlement-specific DieselPrice (USD/litre) from NB00
      rather than a flat national fuel cost per kWh.
    - Fuel cost = diesel_fraction × annual_demand × diesel_ltr_per_kwh × price

    Returns np.inf if infeasible.
    """
    p = MG_SUBTYPES[subtype]
    # Safely convert demand_timeseries — may be list, numpy array, or string after GeoJSON
    if isinstance(demand_timeseries, str):
        import ast as _ast
        demand_timeseries = _ast.literal_eval(demand_timeseries)
    demand_timeseries = [float(x) for x in demand_timeseries] if demand_timeseries is not None else []
    annual_kwh_y0 = demand_timeseries[0] if demand_timeseries else 0.0
    peak_kw       = peak_kw_from_demand(annual_kwh_y0, subtype) if annual_kwh_y0 > 0 else 0

    if not mg_is_feasible(num_connections, dist_road_km, dist_river_km,
                          subtype, head_m, discharge_m3s, peak_kw):
        return np.inf

    cap = mg_capex(num_households, subtype, annual_kwh=annual_kwh_y0)

    # Fixed OPEX
    annual_fixed_opex = cap["total_capex"] * p["opex_rate"]

    # Variable fuel OPEX (hybrid only — spatially varying diesel price)
    if p["diesel_fraction"] > 0 and annual_kwh_y0 > 0:
        annual_diesel_kwh  = annual_kwh_y0 * p["diesel_fraction"]
        annual_diesel_ltrs = annual_diesel_kwh * p["diesel_ltr_per_kwh"]
        annual_fuel_cost   = annual_diesel_ltrs * diesel_price_usd_l
    else:
        annual_fuel_cost = 0.0

    annual_opex = annual_fixed_opex + annual_fuel_cost

    # Replacement costs
    replacements = {}
    if p["battery_replace_yr"] < p["lifetime_years"] and p["battery_cost_kwh"] > 0:
        replacements[p["battery_replace_yr"]] = cap["battery_capex"]
    if p.get("diesel_replace_yr", 999) < p["lifetime_years"] and cap["diesel_capex"] > 0:
        replacements[p["diesel_replace_yr"]] = cap["diesel_capex"]

    # Energy net of losses
    energy_net = [float(e) * (1 - p["loss_rate"]) for e in demand_timeseries]
    lifetime   = p["lifetime_years"]
    if lifetime > len(energy_net):
        last = energy_net[-1] if energy_net else 0
        energy_net = energy_net + [last] * (lifetime - len(energy_net))

    return compute_lcoe(
        capex          = cap["total_capex"],
        annual_opex    = annual_opex,
        energy_series  = energy_net,
        discount_rate  = DISCOUNT_RATE,
        lifetime_years = lifetime,
        replacement_costs = replacements,
    )


# ── DataFrame-level function ──────────────────────────────────────────────────

def add_minigrid_lcoe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute LCOE for all three mini-grid sub-types per settlement.

    Key: mg_hybrid uses DieselPrice column (settlement-specific, from NB00)
    rather than a flat fuel cost. This is one of the three methodological
    modifications to standard OnSSET.

    New columns:
      lcoe_mg_solar, lcoe_mg_hybrid, lcoe_mg_hydro
      lcoe_minigrid          — best (lowest feasible) LCOE
      minigrid_subtype       — winning sub-type key
      minigrid_subtype_label
      minigrid_capex_usd
      minigrid_peak_kw
      minigrid_feasible
      mg_hydro_feasible
    """
    df = df.copy()

    # Road distance
    road_col  = "dist_road_km"
    df["_road"] = df[road_col].fillna(0) if road_col in df.columns else 0.0

    # River distance
    river_col = "dist_river_km"
    df["_river"] = df[river_col].fillna(999) if river_col in df.columns else 999.0
    if river_col not in df.columns:
        import warnings
        warnings.warn("dist_river_km not found — mini-hydro infeasible everywhere", UserWarning)

    # Hydro head and discharge
    df["_head"]      = df["HydroHead"].fillna(0)      if "HydroHead"      in df.columns else 0.0
    df["_discharge"] = df["HydroDischarge"].fillna(0) if "HydroDischarge" in df.columns else 0.0

    # Settlement-specific diesel price (OnSSET logistics model from NB00)
    # Fallback to SBEE regulated base price if column missing
    DIESEL_BASE = 0.85
    df["_diesel_price"] = df["DieselPrice"].fillna(DIESEL_BASE) \
        if "DieselPrice" in df.columns else DIESEL_BASE
    if "DieselPrice" not in df.columns:
        import warnings
        warnings.warn(
            "DieselPrice column not found — using flat $0.85/L for all settlements. "
            "Run Notebook 00 to compute settlement-specific diesel prices.",
            UserWarning
        )

    # Compute LCOE per sub-type
    for st in MG_SUBTYPES:
        df[f"lcoe_{st}"] = df.apply(
            lambda r, s=st: mg_lcoe(
                num_households      = r["num_households"],
                num_connections     = r.get("num_connections", r["num_households"]),
                dist_road_km        = r["_road"],
                dist_river_km       = r["_river"],
                demand_timeseries   = r.get("demand_timeseries", []),
                subtype             = s,
                diesel_price_usd_l  = r["_diesel_price"],
                head_m              = r["_head"],
                discharge_m3s       = r["_discharge"],
            ),
            axis=1,
        )

    # Best mini-grid = lowest feasible LCOE
    mg_cols  = [f"lcoe_{st}" for st in MG_SUBTYPES]
    mat      = df[mg_cols].values
    best_idx = np.argmin(mat, axis=1)

    df["lcoe_minigrid"]          = mat[np.arange(len(df)), best_idx]
    df["minigrid_subtype"]       = [list(MG_SUBTYPES.keys())[i] for i in best_idx]
    df["minigrid_subtype_label"] = df["minigrid_subtype"].map(
        {st: MG_SUBTYPES[st]["label"] for st in MG_SUBTYPES}
    )

    def _cap(row):
        ts  = row.get("demand_timeseries", [])
        if isinstance(ts, str):
            import ast as _ast; ts = _ast.literal_eval(ts)
        ts  = [float(x) for x in ts] if ts else []
        kwh = ts[0] if ts else None
        c   = mg_capex(row["num_households"], row["minigrid_subtype"], annual_kwh=kwh)
        return c["total_capex"], c["peak_kw"]

    cp = df.apply(_cap, axis=1, result_type="expand")
    df["minigrid_capex_usd"] = cp[0]
    df["minigrid_peak_kw"]   = cp[1]
    df["minigrid_feasible"]  = df["lcoe_minigrid"] < np.inf
    df["mg_hydro_feasible"]  = df["lcoe_mg_hydro"] < np.inf

    df = df.drop(columns=["_road","_river","_head","_discharge","_diesel_price"],
                 errors="ignore")

    # Summary
    print("=== MINI-GRID LCOE SUMMARY ===")
    for st in MG_SUBTYPES:
        col = f"lcoe_{st}"
        fin = df[col].replace(np.inf, np.nan).dropna()
        feas = (df[col] < np.inf).sum()
        print(f"  {MG_SUBTYPES[st]['label']:<35}: "
              f"median=${fin.median():.3f}  feasible={feas:,} ({feas/len(df)*100:.0f}%)")

    return df
