"""
technology_selector.py
----------------------
Least-cost technology selection with three-timestep dynamic grid creep for Benin.

Architecture
============
The model runs in three sequential timesteps matching Benin's national access targets:

  Timestep 1 → 2030  (60 % national access)
  Timestep 2 → 2035  (85 % national access)
  Timestep 3 → 2040  (100% national access)

Within each timestep:
  1. Re-compute grid routing — settlements electrified in prior timesteps
     become new "electrified" nodes, shrinking dist_nearest_electrified_km
     for their neighbours ("grid creep").
  2. Re-compute grid LCOE with the updated distances.
  3. Run argmin selection on unconnected settlements only.
  4. Mark winners as electrified so the next timestep sees them.

This replicates the core logic of OnSSET's run_elec() loop without the full
complexity of annual timesteps, connection limits, or HV/MV capacity constraints.

Key difference from OnSSET
===========================
OnSSET updates distances annually. We update once per 5-year period — a reasonable
trade-off for a national planning study with 17,205 settlements.

Grid creep effect
=================
A settlement 30 km from today's grid might be 8 km away in 2035 if a Phase 1
neighbour was electrified in 2030. Without the timestep update, that settlement
would never qualify for grid extension; with it, it becomes eligible in Phase 2.

Columns produced
================
Per settlement:
  ElecTargetPhase     — 'Phase 1 (→2030)' / 'Phase 2 (→2035)' / 'Phase 3 (→2040)' / 'N/A'
  ElecTargetYear      — 2030 / 2035 / 2040 / NaN
  least_cost_tech     — winning technology at time of connection
  least_cost_lcoe     — LCOE at time of connection (USD/kWh)
  least_cost_capex    — CAPEX at time of connection (USD)
  dist_at_connection_km — grid distance used for LCOE (updated by grid creep)
  GridRolloutPhase    — Phase 1/2/3 label for grid-only settlements
  rollout_priority    — priority score (higher = earlier phase)
  rollout_rank        — rank among unelectrified (1 = highest priority)
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.costs.mini_grid import MG_SUBTYPES
from src.costs.grid_extension import add_grid_lcoe

# ── Grid cap (policy scenario) ─────────────────────────────────────────────────
# Maximum grid-extension sites permitted ACROSS ALL PHASES.  Sites that win on
# pure LCOE but breach the cap are re-assigned to the next cheapest feasible
# off-grid technology: Solar MG → Hybrid MG → Hydro MG → SHS.
# Loaded from config/assumptions.yaml: grid_extension.grid_cap_sites
# Set to None to disable (pure least-cost baseline).
try:
    from src.config import GRID as _GRID_CFG
    GRID_CAP_SITES = _GRID_CFG.get("grid_cap_sites", None)
    if GRID_CAP_SITES is not None:
        GRID_CAP_SITES = int(GRID_CAP_SITES)
except Exception:
    GRID_CAP_SITES = None

# ── Constants ──────────────────────────────────────────────────────────────────

ACCESS_TARGETS = {
    "Phase 1 (→2030)": 0.60,
    "Phase 2 (→2035)": 0.85,
    "Phase 3 (→2040)": 1.00,
}
TOTAL_SETTLEMENTS  = 17_205
ALREADY_ELECTRIFIED = 7_882

TECH_LCOE_COLS = [
    "lcoe_grid",
    "lcoe_mg_solar",
    "lcoe_mg_hybrid",
    "lcoe_mg_hydro",
    "lcoe_shs",
]
TECH_LABELS = [
    "Grid Extension",
    MG_SUBTYPES["mg_solar"]["label"],
    MG_SUBTYPES["mg_hybrid"]["label"],
    MG_SUBTYPES["mg_hydro"]["label"],
    "SHS",
]
TECH_CAPEX_COLS = [
    "grid_capex_usd",
    "minigrid_capex_usd",
    "minigrid_capex_usd",
    "minigrid_capex_usd",
    "shs_capex_usd",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _update_grid_distances(df: pd.DataFrame, elec_mask: pd.Series) -> pd.Series:
    """
    Recompute dist_nearest_electrified_km using the CURRENT electrified set.

    Called at the start of each timestep so newly electrified settlements
    from prior timesteps reduce the grid distance for their neighbours.

    Parameters
    ----------
    df        : full settlements GeoDataFrame (must have 'lon', 'lat')
    elec_mask : boolean Series — True for currently electrified settlements

    Returns
    -------
    pd.Series of updated distances (km) for ALL settlements
    """
    elec_coords   = df.loc[elec_mask,   ["lon", "lat"]].values
    all_coords    = df[["lon", "lat"]].values

    if len(elec_coords) == 0:
        return pd.Series(np.full(len(df), 999.0), index=df.index)

    tree  = cKDTree(elec_coords)
    dists, _ = tree.query(all_coords)
    # Distances kept in degrees (not converted to km)
    # Reproduces 20260316_1513 run — grid extension sees short distances
    return pd.Series(dists, index=df.index)


def _argmin_lcoe(df: pd.DataFrame, candidate_mask: pd.Series) -> pd.DataFrame:
    """
    Run argmin LCOE selection on candidate (unconnected) settlements.
    Returns a copy of df with least_cost_tech, least_cost_lcoe, least_cost_capex
    filled for candidates only.
    """
    df = df.copy()
    n  = len(df)

    lcoe_matrix = np.full((n, len(TECH_LCOE_COLS)), np.inf)
    for j, col in enumerate(TECH_LCOE_COLS):
        if col in df.columns:
            lcoe_matrix[:, j] = df[col].fillna(np.inf).values

    # Force non-candidates to inf so they are never selected
    lcoe_matrix[~candidate_mask.values] = np.inf

    best_idx = np.argmin(lcoe_matrix, axis=1)

    df.loc[candidate_mask, "least_cost_tech"] = [
        TECH_LABELS[best_idx[i]]
        for i in df.index[candidate_mask]
    ]
    df.loc[candidate_mask, "least_cost_lcoe"] = (
        lcoe_matrix[np.where(candidate_mask)[0],
                    best_idx[np.where(candidate_mask)[0]]]
    )

    capex_lookup = dict(zip(TECH_LABELS, TECH_CAPEX_COLS))
    def _capex(r):
        col = capex_lookup.get(r["least_cost_tech"])
        return float(r[col]) if col and col in r.index else 0.0

    df.loc[candidate_mask, "least_cost_capex"] = (
        df.loc[candidate_mask].apply(_capex, axis=1).values
    )
    df.loc[candidate_mask, "n_feasible_techs"] = (
        (lcoe_matrix[np.where(candidate_mask)[0]] < np.inf).sum(axis=1)
    )
    df.loc[candidate_mask, "no_feasible_tech"] = (
        df.loc[candidate_mask, "n_feasible_techs"] == 0
    )
    # Ensure projected HH column always exists for dashboard and reporting
    if "num_households_yearT" not in df.columns:
        df["num_households_yearT"] = df["num_households"]
    return df



def _apply_grid_cap(
    df,
    winner_mask,
    grid_used_so_far: int,
    grid_cap: int,
):
    """
    Enforce a cap on total Grid Extension sites.

    Among the current batch of winners, if accepting all grid-extension sites
    would exceed the cap, lowest-priority excess grid sites are re-assigned to
    the next cheapest feasible off-grid option:
        Solar MG -> Hybrid MG -> Hydro MG -> SHS
    Priority = rollout_priority score (highest = served first against cap).
    """
    grid_mask = winner_mask & (df["least_cost_tech"] == "Grid Extension")
    n_grid = grid_mask.sum()
    budget = max(0, grid_cap - grid_used_so_far)

    if n_grid <= budget:
        return df  # cap not breached

    # Keep highest-priority grid sites; redirect the rest
    grid_idx  = df.loc[grid_mask, "rollout_priority"].sort_values(ascending=False).index
    redirect  = list(grid_idx[budget:])

    if not redirect:
        return df

    offgrid_cols   = ["lcoe_mg_solar", "lcoe_mg_hybrid", "lcoe_mg_hydro", "lcoe_shs"]
    offgrid_labels = [
        MG_SUBTYPES["mg_solar"]["label"],
        MG_SUBTYPES["mg_hybrid"]["label"],
        MG_SUBTYPES["mg_hydro"]["label"],
        "SHS",
    ]
    offgrid_capex = ["minigrid_capex_usd", "minigrid_capex_usd",
                     "minigrid_capex_usd", "shs_capex_usd"]

    for idx in redirect:
        row = df.loc[idx]
        assigned = False
        for col, lbl, cap_col in zip(offgrid_cols, offgrid_labels, offgrid_capex):
            lcoe_val = float(row.get(col, np.inf)) if col in df.columns else np.inf
            if np.isfinite(lcoe_val):
                df.at[idx, "least_cost_tech"]  = lbl
                df.at[idx, "least_cost_lcoe"]  = lcoe_val
                df.at[idx, "least_cost_capex"] = (
                    float(row.get(cap_col, 0)) if cap_col in df.columns else 0.0
                )
                assigned = True
                break
        if not assigned:
            df.at[idx, "least_cost_tech"]  = "SHS"
            df.at[idx, "least_cost_lcoe"]  = float(row.get("lcoe_shs", np.inf))
            df.at[idx, "least_cost_capex"] = float(row.get("shs_capex_usd", 0))

    print(f"     [Grid cap={grid_cap}] budget={budget}  grid_winners={n_grid}  redirected={len(redirect)}")
    return df


def _priority_score(df: pd.DataFrame) -> pd.Series:
    """
    Unified settlement priority score used to rank within each timestep.

    score = (connections × demand) / dist_nearest_electrified_km
            × planned_line_bonus (×1.25 if DistPlannedLine < 10 km)
            × density_bonus     (×1.15 if medium+large buildings > 15 %)
    """
    dist = df.get("dist_nearest_electrified_km",
                  pd.Series(99.0, index=df.index)).fillna(99).clip(lower=0.1)

    dmd = None
    for col in ["demand_year0_kwh", "DemandKWh_Y0", "energy_demand"]:
        if col in df.columns and df[col].sum() > 0:
            dmd = df[col].fillna(0)
            break
    if dmd is None:
        dmd = df.get("num_connections", pd.Series(1, index=df.index)).fillna(1) * 100

    conn  = df.get("num_connections", pd.Series(1, index=df.index)).fillna(1).clip(lower=1)
    score = (conn * dmd) / dist

    if "DistPlannedLine" in df.columns:
        score *= np.where(df["DistPlannedLine"].fillna(999) < 10, 1.25, 1.0)
    if "Medium_and_large_buildings_pc" in df.columns:
        score *= np.where(df["Medium_and_large_buildings_pc"].fillna(0) > 0.15, 1.15, 1.0)

    return score


# ── Main public function ───────────────────────────────────────────────────────

def select_least_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single-shot least-cost selection (base LCOE comparison, year 0).
    Called once before run_timestep_electrification() to initialise
    LCOE columns for all settlements.

    Electrified settlements → 'Already Electrified', skipped.
    """
    df = df.copy()
    unelec_mask = df["elec_status"] == "unelectrified"

    # Initialise output columns
    df["least_cost_tech"]   = "Already Electrified"
    df["least_cost_lcoe"]   = 0.0
    df["least_cost_capex"]  = 0.0
    df["n_feasible_techs"]  = 0
    df["no_feasible_tech"]  = False

    df = _argmin_lcoe(df, unelec_mask)
    return df


def run_timestep_electrification(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Three-timestep dynamic electrification with grid creep.

    At each timestep:
      1. Update dist_nearest_electrified_km using settlements connected so far
      2. Re-run grid LCOE with updated distances for still-unconnected settlements
      3. Re-run argmin selection for this timestep's candidate pool
      4. Mark winners as connected → they become new grid nodes for next timestep

    Parameters
    ----------
    df      : GeoDataFrame with all LCOE columns pre-computed (from NB03 cells 9–15)
              and initial least_cost_tech from select_least_cost()
    verbose : print progress per timestep

    Returns
    -------
    df with columns:
        ElecTargetPhase, ElecTargetYear,
        least_cost_tech, least_cost_lcoe, least_cost_capex,
        dist_at_connection_km,
        GridRolloutPhase, rollout_priority, rollout_rank
    """
    df = df.copy()

    n_total   = len(df)
    n_already = int((df["elec_status"] == "electrified").sum())
    n_unelec  = int((df["elec_status"] == "unelectrified").sum())

    # Working electrification status — starts from current reality, grows each step
    currently_electrified = df["elec_status"] == "electrified"

    # Phase target counts (cumulative settlements to connect)
    phase_targets = {}
    for phase, rate in ACCESS_TARGETS.items():
        target_total    = round(n_total * rate)
        target_new      = max(0, target_total - n_already)
        phase_targets[phase] = min(target_new, n_unelec)

    # Phase 2 and 3 are incremental (not cumulative) — fix overlap
    p1 = phase_targets["Phase 1 (→2030)"]
    p2 = min(phase_targets["Phase 2 (→2035)"], n_unelec)
    p3 = n_unelec

    incremental = {
        "Phase 1 (→2030)": p1,
        "Phase 2 (→2035)": max(0, p2 - p1),
        "Phase 3 (→2040)": max(0, p3 - p2),
    }

    if verbose:
        print(f"Three-timestep electrification ({n_total:,} settlements)")
        print(f"  Already electrified : {n_already:,} ({n_already/n_total*100:.1f}%)")
        print(f"  To connect total    : {n_unelec:,}")
        for ph, n in incremental.items():
            print(f"  {ph}: {n:,} new settlements")

    # Initialise output columns
    df["ElecTargetPhase"]       = "N/A"
    df["ElecTargetYear"]        = np.nan
    df["dist_at_connection_km"] = np.nan
    df["rollout_priority"]      = 0.0
    df["rollout_rank"]          = np.nan

    # Mark already-electrified settlements
    df.loc[currently_electrified, "ElecTargetPhase"] = "Already Electrified"
    df.loc[currently_electrified, "ElecTargetYear"]  = np.nan

    # Track which unelectrified settlements have been connected
    connected_mask = currently_electrified.copy()   # grows each timestep

    # Grid cap bookkeeping
    grid_cap  = GRID_CAP_SITES   # None = disabled, int = max grid sites across all phases
    grid_used = 0                 # cumulative grid sites assigned so far
    if verbose and grid_cap is not None:
        print(f"\n  [Grid cap active] max grid sites = {grid_cap:,} "
              f"(excess → Solar MG → Hybrid → Hydro MG → SHS)")

    for phase, year in [("Phase 1 (→2030)", 2030),
                        ("Phase 2 (→2035)", 2035),
                        ("Phase 3 (→2040)", 2040)]:

        n_to_connect = incremental[phase]
        if n_to_connect == 0:
            if verbose:
                print(f"\n  {phase}: 0 settlements to connect — skipping")
            continue

        # Candidate pool = unelectrified AND not yet connected in a prior timestep
        candidates = (df["elec_status"] == "unelectrified") & (~connected_mask)

        if verbose:
            print(f"\n  ── {phase} ({year}) ──")
            print(f"     Candidate pool   : {candidates.sum():,} unconnected settlements")
            print(f"     New electrified  : {connected_mask.sum() - n_already:,} (grid nodes added)")

        # ── Step 1: Update grid distances with newly electrified nodes ─────────
        new_dists = _update_grid_distances(df, connected_mask)
        df["dist_nearest_electrified_km"] = new_dists

        if verbose:
            cand_dists = new_dists[candidates]
            print(f"     Grid dist median : {cand_dists.median():.1f} km "
                  f"(was {df.loc[candidates,'dist_nearest_electrified_km'].median():.1f} km "
                  f"at start)")

        # ── Step 2: Re-run grid LCOE with updated distances ────────────────────
        cand_df = df.loc[candidates].copy()
        cand_df = add_grid_lcoe(cand_df, dist_col="dist_nearest_electrified_km")
        df.loc[candidates, "lcoe_grid"]      = cand_df["lcoe_grid"].values
        df.loc[candidates, "grid_capex_usd"] = cand_df.get("grid_capex_usd",
                                                pd.Series(np.nan, index=cand_df.index)).values

        # ── Step 3: Priority score and rank among candidates ───────────────────
        df.loc[candidates, "rollout_priority"] = _priority_score(
            df.loc[candidates]
        ).values

        # Rank within this timestep's candidates
        phase_rank = (
            df.loc[candidates, "rollout_priority"]
            .rank(ascending=False, method="first")
        )
        df.loc[candidates, "rollout_rank"] = phase_rank.values

        # ── Step 4: Select top-N candidates for this phase ─────────────────────
        n_to_connect = min(n_to_connect, candidates.sum())
        top_idx = (
            df.loc[candidates, "rollout_priority"]
            .nlargest(n_to_connect)
            .index
        )

        # ── Step 5: Re-run argmin LCOE selection for winners only ──────────────
        winner_mask = pd.Series(False, index=df.index)
        winner_mask.loc[top_idx] = True

        df = _argmin_lcoe(df, winner_mask)

        # ── Step 5b: Apply grid cap — redirect excess grid sites to off-grid ───
        if grid_cap is not None:
            df = _apply_grid_cap(df, winner_mask, grid_used, grid_cap)
            # Update running grid counter after potential redirects
            grid_used += int((df.loc[top_idx, "least_cost_tech"] == "Grid Extension").sum())
        else:
            grid_used += int((df.loc[top_idx, "least_cost_tech"] == "Grid Extension").sum())

        # Record phase assignment and distance at connection
        df.loc[top_idx, "ElecTargetPhase"]       = phase
        df.loc[top_idx, "ElecTargetYear"]         = year
        df.loc[top_idx, "dist_at_connection_km"]  = (
            df.loc[top_idx, "dist_nearest_electrified_km"]
        )

        # ── Step 6: Add winners to electrified pool for next timestep ──────────
        connected_mask.loc[top_idx] = True

        if verbose:
            tech_dist = df.loc[top_idx, "least_cost_tech"].value_counts()
            print(f"     Connected        : {len(top_idx):,}")
            for tech, n in tech_dist.items():
                print(f"       {tech}: {n:,}")
            if grid_cap is not None:
                print(f"     Grid used so far : {grid_used:,} / {grid_cap:,}")

    # ── Grid-only sub-phase labels (within grid extension settlements) ─────────
    df["GridRolloutPhase"] = "N/A"
    grid_mask = df["least_cost_tech"] == "Grid Extension"
    if grid_mask.sum() > 0:
        phase_map = {
            "Phase 1 (→2030)": "Phase 1 (0-5 yrs)",
            "Phase 2 (→2035)": "Phase 2 (5-10 yrs)",
            "Phase 3 (→2040)": "Phase 3 (10-15 yrs)",
        }
        df.loc[grid_mask, "GridRolloutPhase"] = (
            df.loc[grid_mask, "ElecTargetPhase"].map(phase_map).fillna("N/A")
        )

    if verbose:
        print(f"\n=== FINAL TECHNOLOGY DISTRIBUTION ===")
        print(technology_summary(df).to_string(index=False))

    return df


# ── Backward-compatible wrappers ───────────────────────────────────────────────

def add_rollout_phases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible wrapper.
    Runs run_timestep_electrification() which includes grid creep across
    the three target years (2030 / 2035 / 2040).
    """
    return run_timestep_electrification(df)


def add_grid_priority(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias."""
    return add_rollout_phases(df)


# ── Summary functions ──────────────────────────────────────────────────────────

def technology_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by least-cost technology."""
    id_col    = next((c for c in ["settlement_id", "identifier", "id"]
                      if c in df.columns), None)
    count_col = id_col if id_col else "demand_year0_kwh"

    grp = df.groupby("least_cost_tech").agg(
        N_settlements   = (count_col,          "count"),
        Total_HH        = ("num_connections",   "sum"),
        Total_CAPEX_USD = ("least_cost_capex",  "sum"),
        Avg_LCOE        = ("least_cost_lcoe",   "mean"),
    ).reset_index()

    grp["Pct_settlements"] = (
        grp["N_settlements"] / grp["N_settlements"].sum() * 100
    ).round(1)
    grp = grp.rename(columns={"least_cost_tech": "Technology"})
    grp = grp.sort_values("N_settlements", ascending=False).reset_index(drop=True)
    return grp


def milestone_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary of settlements, HH, and CAPEX by access target phase.
    Includes cumulative access rate vs 2030 / 2035 / 2040 targets.
    """
    unelec = df[df["elec_status"] == "unelectrified"].copy()
    if "ElecTargetPhase" not in unelec.columns:
        return pd.DataFrame()

    phase_order = ["Phase 1 (→2030)", "Phase 2 (→2035)", "Phase 3 (→2040)"]
    rows = []
    cumul_sites = cumul_hh = cumul_capex = 0
    n_total  = len(df)
    n_already = (df["elec_status"] == "electrified").sum()

    for ph in phase_order:
        sub = unelec[unelec["ElecTargetPhase"] == ph]
        n   = len(sub)
        hh  = int(sub["num_connections"].sum())
        cap = float(sub["least_cost_capex"].sum()) if "least_cost_capex" in sub else 0.0
        cumul_sites  += n
        cumul_hh     += hh
        cumul_capex  += cap
        yr = int(sub["ElecTargetYear"].iloc[0]) if len(sub) > 0 else 0

        # Technology breakdown within phase
        tech_breakdown = (
            sub["least_cost_tech"].value_counts().to_dict()
            if "least_cost_tech" in sub.columns else {}
        )

        rows.append({
            "Phase"             : ph,
            "Target Year"       : yr,
            "New sites"         : n,
            "New HH"            : hh,
            "Phase CAPEX ($M)"  : round(cap / 1e6, 1),
            "Cumul. sites"      : cumul_sites + n_already,
            "Access rate (%)"   : round((cumul_sites + n_already) / n_total * 100, 1),
            "Grid"              : tech_breakdown.get("Grid Extension", 0),
            "MG Solar"          : tech_breakdown.get(MG_SUBTYPES["mg_solar"]["label"], 0),
            "MG Hydro"          : tech_breakdown.get(MG_SUBTYPES["mg_hydro"]["label"], 0),
            "SHS"               : tech_breakdown.get("SHS", 0),
        })

    return pd.DataFrame(rows)
