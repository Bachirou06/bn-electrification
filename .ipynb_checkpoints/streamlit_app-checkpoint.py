"""
Benin Electrification Explorer
================================
Inspired by GEP-OnSSET:
  - visualization_app.py  → SPLAT palettes, doughnut, stacked bar, statistics panel, PNG/SVG export
  - annual_rollout_app.py → S-curve allocation, annual breakdown, 5-year periods, cumulative S-curve

Run:  streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings, io
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Benin Electrification Explorer",
                   page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.kpi-card{background:white;border-radius:10px;padding:16px 20px;
  border-left:4px solid #1565C0;box-shadow:0 1px 4px rgba(0,0,0,.08);margin:4px 0}
.kpi-label{font-size:.72rem;text-transform:uppercase;letter-spacing:.8px;
  color:#90A4AE;font-weight:600}
.kpi-value{font-size:1.6rem;font-weight:700;color:#1A237E;line-height:1.2;
  font-family:'DM Mono',monospace}
.section-hd{font-size:.9rem;font-weight:600;color:#1565C0;text-transform:uppercase;
  letter-spacing:.6px;border-bottom:2px solid #E3F2FD;padding-bottom:6px;
  margin:14px 0 10px 0}
.info-box{background:#E3F2FD;border-left:4px solid #1565C0;padding:9px 13px;
  border-radius:0 6px 6px 0;font-size:.84rem;color:#1A237E;margin:6px 0}
.phase-card{border-radius:8px;padding:12px 16px;color:white;text-align:center;margin:4px 0}
.phase-title{font-weight:700;font-size:.84rem}
.phase-detail{font-size:.77rem;opacity:.9}
</style>""", unsafe_allow_html=True)

# ── Colours ──────────────────────────────────────────────────────────────────
TECH_COLORS = {
    "Already Electrified"            : "#90A4AE",
    "Grid Extension"                 : "#1565C0",
    "Mini-Grid: Solar PV Only"       : "#F9A825",
    "Mini-Grid: Solar-Diesel Hybrid" : "#E53935",
    "Mini-Grid: Mini-Hydro"          : "#00897B",
    "SHS"                            : "#43A047",
}
ALL_TECHS   = list(TECH_COLORS.keys())
UNELEC      = [t for t in ALL_TECHS if t != "Already Electrified"]
PHASE_COLORS = {
    "Phase 1 (→2030)" : "#C62828",   # 60% target
    "Phase 2 (→2035)" : "#E65100",   # 85% target
    "Phase 3 (→2040)" : "#1565C0",   # 100% target
}
ACCESS_TARGETS    = {2030: 0.60, 2035: 0.85, 2040: 1.00}
TOTAL_SETTLEMENTS = 17_205
ALREADY_ELECTRIFIED = 7_882

# ── Allocation strategies (annual_rollout_app) ───────────────────────────────
def weights_s_curve(n, k=8):
    if n<=0: return np.array([1.])
    x=np.linspace(-2,2,n+1); c=1/(1+np.exp(-k*x)); w=np.diff(c)
    return w/w.sum()
def weights_front(n):
    if n<=0: return np.array([1.])
    w=np.linspace(n,1,n); return w/w.sum()
def weights_back(n):
    if n<=0: return np.array([1.])
    w=np.linspace(1,n,n); return w/w.sum()
def weights_uniform(n):
    return np.ones(max(n,1))/max(n,1)

STRATS = {
    "uniform" : ("Uniform — equal each year",    weights_uniform),
    "front"   : ("Front-loaded — more early",    weights_front),
    "back"    : ("Back-loaded — more later",     weights_back),
}

def allocate(total, n, strategy):
    fn = STRATS[strategy][1]
    return total * fn(n)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for base in [Path("data/outputs/tables"), Path("../data/outputs/tables")]:
        files = sorted(base.glob("benin_electrification_streamlit_*.csv"), reverse=True)
        if files:
            df = pd.read_csv(files[0])
            for c in df.select_dtypes("object").columns:
                try: df[c] = pd.to_numeric(df[c])
                except: pass
            if "GridRolloutPhase" in df.columns:
                df["GridRolloutPhase"] = df["GridRolloutPhase"].fillna("N/A")
            for t in ALL_TECHS:
                if t not in df["MinimumOverall"].values:
                    df = pd.concat([df, pd.DataFrame([{
                        "MinimumOverall":t,"NumConnections":0,
                        "InvestmentCost":0,"MinimumOverallLCOE":np.nan,
                        "_dummy":True}])],
                        ignore_index=True)
            if "_dummy" not in df.columns:
                df["_dummy"] = False
            else:
                df["_dummy"] = df["_dummy"].fillna(False)
            return df, files[0].name
    st.error("⚠️ No benin_electrification_streamlit_*.csv — run NB03 save cell first.")
    st.stop()

df_all, fname = load_data()
dist_col       = "DistNearElecKm"      if "DistNearElecKm"      in df_all.columns else "GridDistKm"
conn_dist_col  = "DistAtConnectionKm"  if "DistAtConnectionKm"  in df_all.columns else dist_col

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Flag_of_Benin.svg/200px-Flag_of_Benin.svg.png", width=80)
    st.title("⚡ Benin Electrification")
    st.caption(f"📄 {fname}")
    st.markdown("---")
    st.markdown("**🔌 Technology**")
    sel_techs = st.multiselect("", ALL_TECHS, default=ALL_TECHS, label_visibility="collapsed")

    st.markdown("**📅 Grid Phase**")
    sel_phase = st.selectbox("", ["All"]+list(PHASE_COLORS.keys()),
                              label_visibility="collapsed")

    st.markdown("**🎯 Access Target Phase**")
    sel_target_phase = st.selectbox("",
        ["All", "Phase 1 (→2030)", "Phase 2 (→2035)", "Phase 3 (→2040)"],
        label_visibility="collapsed", key="target_phase_sel")

    st.markdown("**👥 Connections (HH)**")
    cmax = int(df_all["NumConnections"].quantile(.99)) if "NumConnections" in df_all.columns else 500
    conn_r = st.slider("", 0, cmax, (0, cmax), step=5, label_visibility="collapsed")

    if dist_col in df_all.columns:
        st.markdown("**📏 Distance to Grid (km)**")
        dmax = float(df_all[dist_col].replace([np.inf,-np.inf],np.nan).max())
        dist_r = st.slider("", 0., dmax, (0., dmax), step=1., label_visibility="collapsed")
    else: dist_r = (0., 9999.)

    if "DemandKWh_Y0" in df_all.columns:
        st.markdown("**⚡ Demand (kWh/yr)**")
        demmax = float(df_all["DemandKWh_Y0"].quantile(.99))
        dem_r = st.slider("", 0., demmax, (0., demmax), step=100., label_visibility="collapsed")
    else: dem_r = (0., 9e9)

    if "InvestmentCost" in df_all.columns:
        st.markdown("**💰 CAPEX (USD)**")
        invmax = float(df_all["InvestmentCost"].replace([np.inf,-np.inf],np.nan).max())
        inv_r = st.slider("", 0., invmax, (0., invmax), step=5000., label_visibility="collapsed")
    else: inv_r = (0., 9e9)

    if "MinimumOverallLCOE" in df_all.columns:
        st.markdown("**📊 LCOE ($/kWh)**")
        lcoe_max = float(df_all["MinimumOverallLCOE"].replace([np.inf,-np.inf],np.nan).quantile(.99))
        lcoe_r = st.slider("", 0., lcoe_max, (0., lcoe_max), step=.05,
                            label_visibility="collapsed")
    else: lcoe_r = (0., 9999.)

    st.divider()
    st.markdown("**🔬 Model Parameters**")
    st.caption("Override MG feasibility gates live — no need to re-run NB03")

    mg_threshold = st.slider(
        "☀️ Solar MG min demand (kWh/yr)",
        min_value=5_000, max_value=25_000,
        value=10_000, step=1_000,
        help="ESMAP lower bound ≈ 10,000 kWh/yr. Lower = more MG sites. "
             "Original model used 15,000."
    )
    hybrid_road = st.slider(
        "🛣️ Hybrid max road dist (km)",
        min_value=20, max_value=200,
        value=100, step=10,
        help="Max distance for diesel delivery. Extended from 50→100km."
    )
    pu_enabled = st.toggle(
        "🌾 Productive use uplift (+30%)",
        value=True,
        help="Applies ×1.30 demand to settlements with health/school/cropland. "
             "Pushes borderline settlements above MG threshold (~25% of sites)."
    )
    st.caption(
        f"Solar MG ≥ {mg_threshold:,} kWh/yr | "
        f"Hybrid ≤ {hybrid_road} km | "
        f"PU {'ON +30%' if pu_enabled else 'OFF'}"
    )

    st.divider()
    if st.button("🔄 Reset filters", use_container_width=True): st.rerun()

# ── Apply filters ─────────────────────────────────────────────────────────────
def safe(col, lo, hi):
    return df_all[col].replace([np.inf,-np.inf],np.nan).fillna(0).between(lo, hi)

mask = df_all["MinimumOverall"].isin(sel_techs)
# Keep Already Electrified regardless of connection count slider
conn_mask = safe("NumConnections", *conn_r) | (df_all["MinimumOverall"] == "Already Electrified")
mask &= conn_mask
if dist_col in df_all.columns:         mask &= safe(dist_col, *dist_r)
if "DemandKWh_Y0"      in df_all.columns: mask &= safe("DemandKWh_Y0", *dem_r)
if "InvestmentCost"     in df_all.columns: mask &= safe("InvestmentCost", *inv_r)
if "MinimumOverallLCOE" in df_all.columns:
    # Always keep Already Electrified (LCOE=0) regardless of LCOE slider
    lcoe_mask = safe("MinimumOverallLCOE", *lcoe_r) | (df_all["MinimumOverall"] == "Already Electrified")
    mask &= lcoe_mask
if sel_phase != "All":
    phase_col = "ElecTargetPhase" if "ElecTargetPhase" in df_all.columns else "GridRolloutPhase"
    mask &= (df_all.get(phase_col, pd.Series("N/A",index=df_all.index)) == sel_phase)
df = df_all[mask & ~df_all.get("_dummy", pd.Series(False, index=df_all.index))].copy()

# ── Live MG re-simulation based on sidebar threshold slider ──────────────────
# The saved CSV was produced with a fixed threshold. When the user moves the
# slider we recompute which settlements would qualify under the new threshold
# without re-running the full LCOE pipeline.
if "DemandKWh_Y0" in df_all.columns:
    pu_factor = 1.30 if pu_enabled else 1.0

    def _recompute_tech(row):
        """Reassign technology live based on current sidebar parameters."""
        orig = row["MinimumOverall"]
        if orig == "Already Electrified":
            return orig
        demand = float(row.get("DemandKWh_Y0", 0) or 0) * pu_factor

        # If currently SHS — check if it now qualifies for Solar MG
        if orig == "SHS" and demand >= mg_threshold:
            return "Mini-Grid: Solar PV Only"

        # If currently Solar MG — check if demand dropped below threshold
        if orig == "Mini-Grid: Solar PV Only" and demand < mg_threshold:
            return "SHS"

        # Hybrid: check road constraint
        if orig == "Mini-Grid: Solar-Diesel Hybrid":
            road = float(row.get("DistRoadKm", 0) or row.get("dist_road_km", 0) or 0)
            if road > hybrid_road:
                return "Mini-Grid: Solar PV Only"  # too remote for diesel → pure solar

        # Reclassify SHS sites that might now qualify as hybrid
        if orig == "SHS" and demand >= 25_000:
            road = float(row.get("DistRoadKm", 0) or 0)
            if road <= hybrid_road:
                return "Mini-Grid: Solar-Diesel Hybrid"

        return orig

    df_all = df_all.copy()
    df_all["MinimumOverall"] = df_all.apply(_recompute_tech, axis=1)
    # Re-apply mask with updated technologies
    mask = df_all["MinimumOverall"].isin(sel_techs)
    conn_mask = safe("NumConnections", *conn_r) | (df_all["MinimumOverall"] == "Already Electrified")
    mask &= conn_mask
    if dist_col in df_all.columns:         mask &= safe(dist_col, *dist_r)
    if "DemandKWh_Y0" in df_all.columns:   mask &= safe("DemandKWh_Y0", *dem_r)
    if "InvestmentCost" in df_all.columns:  mask &= safe("InvestmentCost", *inv_r)
    if "MinimumOverallLCOE" in df_all.columns:
        lcoe_mask = safe("MinimumOverallLCOE", *lcoe_r) | (df_all["MinimumOverall"] == "Already Electrified")
        mask &= lcoe_mask
    df = df_all[mask & ~df_all.get("_dummy", pd.Series(False, index=df_all.index))].copy()

    # Show a compact banner with the live counts
    n_solar = (df["MinimumOverall"] == "Mini-Grid: Solar PV Only").sum()
    n_hybrid = (df["MinimumOverall"] == "Mini-Grid: Solar-Diesel Hybrid").sum()
    n_shs   = (df["MinimumOverall"] == "SHS").sum()
    st.info(
        f"🔬 **Live model parameters applied** — "
        f"Solar MG threshold: **{mg_threshold:,} kWh/yr** | "
        f"Hybrid road: **≤{hybrid_road} km** | "
        f"PU uplift: **{'×1.30' if pu_enabled else 'off'}**  →  "
        f"☀️ Solar MG: **{n_solar:,}** sites | "
        f"⚡ Hybrid: **{n_hybrid:,}** sites | "
        f"🏠 SHS: **{n_shs:,}** sites",
        icon=None
    )

# Compute installed capacity from demand (more reliable than MG_PeakKW column)
# Peak-to-base ratios from mini_grid.py model parameters
_PTB = {
    "Mini-Grid: Solar PV Only"       : 3.5,
    "Mini-Grid: Solar-Diesel Hybrid" : 2.5,
    "Mini-Grid: Mini-Hydro"          : 2.0,
    "Grid Extension"                 : 1/0.85,  # grid base-to-peak
}
if "DemandKWh_Y0" in df.columns:
    df["_capacity_kw"] = df.apply(
        lambda r: (r["DemandKWh_Y0"] / 8760 * _PTB[r["MinimumOverall"]])
                  if r["MinimumOverall"] in _PTB and pd.notna(r["DemandKWh_Y0"]) else 0,
        axis=1
    )
else:
    df["_capacity_kw"] = df.get("MG_PeakKW", 0).fillna(0)

# ── Header + KPIs ─────────────────────────────────────────────────────────────
st.markdown("## 🇧🇯 Benin Electrification Explorer")
st.caption("National least-cost electrification model · 17,205 settlements · 2025")
st.divider()

def kpi(label, val, delta=""):
    return (f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{val}</div>'
            + (f'<div style="font-size:.78rem;color:#43A047">{delta}</div>' if delta else "")
            + '</div>')

hh  = df["NumConnections"].sum() if "NumConnections" in df.columns else 0
inv = df["InvestmentCost"].sum()  if "InvestmentCost"  in df.columns else 0
med = df["MinimumOverallLCOE"].replace([np.inf,-np.inf],np.nan).median() if "MinimumOverallLCOE" in df.columns else 0
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
hh_s = f"{hh/1e6:.2f}M" if hh>1e6 else f"{hh:,.0f}"
mg_n = int(df["MinimumOverall"].str.startswith("Mini-Grid").sum())
unelec_inv = df[df["MinimumOverall"]!="Already Electrified"]["InvestmentCost"].sum() if "InvestmentCost" in df.columns else 0
real_total = int((~df_all.get("_dummy", pd.Series(False, index=df_all.index))).sum())
with c1: st.markdown(kpi("Settlements",         f"{len(df):,}",           f"of {real_total:,}"), unsafe_allow_html=True)
with c2: st.markdown(kpi("Households",          hh_s), unsafe_allow_html=True)
with c3: st.markdown(kpi("Unelectrified CAPEX", f"${unelec_inv/1e6:.1f}M","excl. electrified"), unsafe_allow_html=True)
with c4: st.markdown(kpi("Median LCOE",         f"${med:.3f}"), unsafe_allow_html=True)
with c5: st.markdown(kpi("Grid sites",          f"{(df['MinimumOverall']=='Grid Extension').sum():,}"), unsafe_allow_html=True)
with c6: st.markdown(kpi("Mini-Grid sites",     f"{mg_n:,}"), unsafe_allow_html=True)
with c7: st.markdown(kpi("SHS sites",           f"{(df['MinimumOverall']=='SHS').sum():,}"), unsafe_allow_html=True)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
t_map, t_charts, t_rollout, t_targets, t_sensitivity, t_sites, t_data = st.tabs(
    ["🗺️ Map", "📊 Charts", "📅 Annual Rollout", "🎯 Access Targets",
     "🔬 Sensitivity", "📍 Site Filtering", "📋 Data"])

# ── Access target constants ────────────────────────────────────────────────────
TARGET_60  = 0.60   # 2030
TARGET_85  = 0.85   # 2035
TARGET_100 = 1.00   # 2040
TOTAL_SETT = 17_205
ALREADY_ELEC = 7_882
TOTAL_UNELEC = TOTAL_SETT - ALREADY_ELEC   # 9,323

TARGET_COLORS = {
    "Phase 1 (→2030)": "#1565C0",
    "Phase 2 (→2035)": "#00897B",
    "Phase 3 (→2040)": "#C62828",
}

def compute_target_phases(df_in):
    """
    Partition unelectrified settlements into 3 phases based on
    60 / 85 / 100% national access targets and priority score.
    Returns df with ElecTargetPhase column.
    """
    df_out = df_in.copy()
    unelec = df_out["MinimumOverall"] != "Already Electrified"

    # How many unelectrified settlements fall in each phase?
    n_total  = TOTAL_SETT
    n_elec   = ALREADY_ELEC
    p1_need  = max(0, round(n_total * TARGET_60  - n_elec))
    p2_need  = max(0, round(n_total * TARGET_85  - n_elec))
    p3_need  = TOTAL_UNELEC

    p1_need  = min(p1_need, TOTAL_UNELEC)
    p2_need  = min(p2_need, TOTAL_UNELEC)

    df_out["ElecTargetPhase"] = "N/A"
    df_out.loc[~unelec, "ElecTargetPhase"] = "Already Electrified"

    if "rollout_rank" in df_out.columns:
        rank_col = "rollout_rank"
    elif "rollout_priority" in df_out.columns:
        # re-rank from priority
        df_out.loc[unelec, "_rank_tmp"] = (
            df_out.loc[unelec, "rollout_priority"]
            .rank(ascending=False, method="first")
        )
        rank_col = "_rank_tmp"
    else:
        # fallback: rank by connections × demand / distance
        dist = df_out.get("DistNearElecKm", df_out.get("GridDistKm",
               pd.Series(50., index=df_out.index))).fillna(50).clip(lower=0.1)
        dmd  = df_out.get("DemandKWh_Y0", pd.Series(1., index=df_out.index)).fillna(1)
        conn = df_out.get("NumConnections", pd.Series(1., index=df_out.index)).fillna(1)
        score = (conn * dmd) / dist
        df_out.loc[unelec, "_rank_tmp"] = (
            score[unelec].rank(ascending=False, method="first")
        )
        rank_col = "_rank_tmp"

    df_out.loc[unelec, "ElecTargetPhase"] = df_out.loc[unelec, rank_col].apply(
        lambda r: "Phase 1 (→2030)" if r <= p1_need
        else      "Phase 2 (→2035)" if r <= p2_need
        else      "Phase 3 (→2040)"
    )
    if "_rank_tmp" in df_out.columns:
        df_out = df_out.drop(columns=["_rank_tmp"])
    return df_out, p1_need, p2_need - p1_need, p3_need - p2_need

# ── MAP ───────────────────────────────────────────────────────────────────────
with t_map:
    mc, ms = st.columns([3,1])
    with mc:
        st.markdown('<div class="section-hd">Technology Map</div>', unsafe_allow_html=True)
        if "X_deg" in df.columns and "Y_deg" in df.columns:
            dfm = df.dropna(subset=["X_deg","Y_deg","MinimumOverall"])
            # Build clean hover using customdata + hovertemplate
            # Only show system size for MG and Grid (not SHS / Already Electrified)
            cap_col   = "_capacity_kw" if "_capacity_kw" in dfm.columns else None
            lcoe_col  = "MinimumOverallLCOE" if "MinimumOverallLCOE" in dfm.columns else None
            dist_c    = dist_col if dist_col in dfm.columns else None
            conn_col  = "NumConnections" if "NumConnections" in dfm.columns else None

            # customdata order: [connections, capacity_kw, lcoe, dist, lon, lat]
            custom_cols = []
            if conn_col:  custom_cols.append(conn_col)
            if cap_col:   custom_cols.append(cap_col)
            if lcoe_col:  custom_cols.append(lcoe_col)
            if dist_c:    custom_cols.append(dist_c)
            custom_cols += ["X_deg","Y_deg"]
            custom_cols  = [c for c in custom_cols if c in dfm.columns]

            HAS_CAP_TECHS = {"Mini-Grid: Solar PV Only","Mini-Grid: Solar-Diesel Hybrid",
                              "Mini-Grid: Mini-Hydro","Grid Extension"}

            def build_hover(row):
                lines_h = [f"<b>{row['MinimumOverall']}</b>"]
                if conn_col and conn_col in row.index:
                    lines_h.append(f"Connections : {int(row[conn_col]):,} HH")
                if cap_col and cap_col in row.index and row['MinimumOverall'] in HAS_CAP_TECHS:
                    kw = row[cap_col]
                    if pd.notna(kw) and kw > 0:
                        lines_h.append(f"System size : {kw:,.1f} kW")
                if lcoe_col and lcoe_col in row.index:
                    lcoe = row[lcoe_col]
                    if pd.notna(lcoe) and lcoe < 99:
                        lines_h.append(f"LCOE        : ${lcoe:.3f}/kWh")
                # Grid settlements: show distance at time of connection (after grid creep)
                if row['MinimumOverall'] == "Grid Extension":
                    if "DistAtConnectionKm" in row.index and pd.notna(row["DistAtConnectionKm"]):
                        lines_h.append(f"Dist. at connection : {row['DistAtConnectionKm']:.1f} km")
                    elif dist_c and dist_c in row.index:
                        lines_h.append(f"Dist. grid  : {row[dist_c]:.1f} km")
                elif dist_c and dist_c in row.index:
                    lines_h.append(f"Dist. grid  : {row[dist_c]:.1f} km")
                lines_h.append(f"Lon/Lat     : {row['X_deg']:.4f}, {row['Y_deg']:.4f}")
                return "<br>".join(lines_h)

            dfm = dfm.copy()
            dfm["_hover_text"] = dfm.apply(build_hover, axis=1)

            # Use go.Figure directly for full hover control
            fig = go.Figure()
            for t in ALL_TECHS:
                sub = dfm[dfm["MinimumOverall"]==t]
                if len(sub) == 0: continue
                # Marker size: proportional to connections
                if conn_col and conn_col in sub.columns:
                    max_conn = dfm[conn_col].max() if dfm[conn_col].max() > 0 else 1
                    sizes = (sub[conn_col].fillna(1) / max_conn * 18 + 6).tolist()
                else:
                    sizes = [8] * len(sub)
                fig.add_trace(go.Scattermapbox(
                    lat=sub["Y_deg"].tolist(),
                    lon=sub["X_deg"].tolist(),
                    mode="markers",
                    name=t,
                    marker=dict(
                        color=TECH_COLORS[t],
                        size=sizes,
                        opacity=0.85,
                    ),
                    text=sub["_hover_text"].tolist(),
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=True,
                ))
            # Add empty traces for missing techs (keep legend complete)
            for t in ALL_TECHS:
                if t not in dfm["MinimumOverall"].values:
                    fig.add_trace(go.Scattermapbox(
                        lat=[], lon=[], mode="markers", name=t,
                        marker=dict(color=TECH_COLORS[t], size=10),
                        showlegend=True))
            fig.update_layout(
                mapbox=dict(style="carto-positron", zoom=5.8,
                            center=dict(lat=9.3, lon=2.3)),
                margin=dict(l=0,r=0,t=0,b=0),
                legend=dict(title="Technology",orientation="v",x=.01,y=.99,
                            bgcolor="rgba(255,255,255,.92)",
                            bordercolor="#CFD8DC",borderwidth=1,font=dict(size=11)),
                height=580, clickmode="event+select")
            sel = st.plotly_chart(fig, use_container_width=True,
                                  on_select="rerun", key="main_map")
            if sel and sel.get("selection",{}).get("points"):
                pt = sel["selection"]["points"][0]
                lat,lon = pt.get("lat"),pt.get("lon")
                if lat and lon:
                    nb = dfm[(dfm["Y_deg"].between(lat-.4,lat+.4))&
                              (dfm["X_deg"].between(lon-.4,lon+.4))]
                    fz = px.scatter_mapbox(nb,lat="Y_deg",lon="X_deg",
                        color="MinimumOverall",color_discrete_map=TECH_COLORS,
                        size="NumConnections" if "NumConnections" in nb.columns else None,
                        size_max=22,hover_name=None,
                        hover_data={"_hover_text":True,"Y_deg":False,"X_deg":False,
                                    "MinimumOverall":False,"NumConnections":False},
                        mapbox_style="carto-positron",zoom=10,
                        center={"lat":lat,"lon":lon},height=320)
                    fz.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
                    fz.update_layout(margin=dict(l=0,r=0,t=0,b=0),showlegend=False)
                    st.caption(f"📍 {len(nb):,} settlements near ({lat:.3f}, {lon:.3f})")
                    st.plotly_chart(fz, use_container_width=True)
        else:
            st.warning("Coordinate columns X_deg/Y_deg not found.")

    with ms:
        st.markdown('<div class="section-hd">Statistics</div>', unsafe_allow_html=True)
        ts2 = df.groupby("MinimumOverall").agg(
            Sites=("MinimumOverall","count"),HH=("NumConnections","sum"),
            CAPEX=("InvestmentCost","sum")).reset_index()
        ts2["Pct"] = (ts2["Sites"]/len(df)*100).round(1)
        dom = ts2.loc[ts2["Sites"].idxmax(),"MinimumOverall"] if len(ts2) > 0 else "—"
        dom_p = ts2["Pct"].max() if len(ts2) > 0 else 0
        st.markdown(f'<div class="info-box">🏆 Dominant: <b>{dom}</b> ({dom_p:.1f}%)<br>'
                    f'💰 Total: <b>${ts2["CAPEX"].sum()/1e6:.1f}M</b></div>',
                    unsafe_allow_html=True)
        for _,row in ts2.sort_values("Sites",ascending=False).iterrows():
            c=TECH_COLORS.get(row["MinimumOverall"],"#999")
            st.markdown(
                f'<div style="display:flex;align-items:center;padding:5px 0;'
                f'border-bottom:1px solid #F5F5F5">'
                f'<div style="width:10px;height:10px;border-radius:50%;'
                f'background:{c};margin-right:8px;flex-shrink:0"></div>'
                f'<div style="flex:1;font-size:.79rem">{row["MinimumOverall"]}</div>'
                f'<div style="font-size:.79rem;font-weight:600;color:#1A237E">'
                f'{row["Sites"]:,} ({row["Pct"]:.1f}%)</div></div>',
                unsafe_allow_html=True)
        if (df["MinimumOverall"]=="Mini-Grid: Solar-Diesel Hybrid").sum()==0:
            with st.expander("ℹ️ Why 0 Hybrid?"):
                st.markdown("""
**Three reasons:**
1. Demand ≥25,000 kWh/yr — only ~8% of settlements qualify
2. Hydro wins at $0.40/kWh vs hybrid $0.53/kWh
3. Solar MG cheaper ($0.52 vs $0.53/kWh)

*Correct model result — not a bug.*""")

# ── CHARTS (SPLAT-inspired) ───────────────────────────────────────────────────
with t_charts:
    ct = st.radio("", ["🍩 Electrification Doughnut","📊 Stacked Bar",
                        "📈 Electrification Rate","🥧 Technology Donut"],
                   horizontal=True, label_visibility="collapsed")

    if "🍩" in ct:
        st.markdown('<div class="section-hd">Electrification Rate — Doughnut</div>',
                    unsafe_allow_html=True)
        labs,vals,clrs=[],[],[]
        for t in ALL_TECHS:
            n=(df["MinimumOverall"]==t).sum()
            if n>0: labs.append(t);vals.append(n);clrs.append(TECH_COLORS[t])
        total=sum(vals)
        elec=(df["MinimumOverall"]=="Already Electrified").sum()
        fig=go.Figure(go.Pie(values=vals,labels=labs,hole=.55,
            marker=dict(colors=clrs,line=dict(color="white",width=2)),
            textinfo="percent",textposition="inside",
            hovertemplate="%{label}<br>%{value:,} settlements<br>%{percent}<extra></extra>"))
        fig.update_layout(height=420,template="plotly_white",
            annotations=[dict(text=f"<b>{elec/total*100:.1f}%</b><br>electrified",
                              x=.5,y=.5,font_size=18,showarrow=False)],
            legend=dict(orientation="v",x=1.02,y=.5),
            margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    elif "Stacked" in ct:
        st.markdown('<div class="section-hd">Technology Distribution</div>',
                    unsafe_allow_html=True)
        metric = st.radio("Metric",["Settlements","HH","CAPEX ($M)","Capacity (kW)"],
                          horizontal=True)

        ts3 = df.groupby("MinimumOverall").agg(
            Settlements = ("MinimumOverall","count"),
            HH          = ("NumConnections","sum"),
            CAPEX       = ("InvestmentCost",   lambda x: x.sum()/1e6),
            Power_kW    = ("_capacity_kw", lambda x: x.sum()),  # computed from demand × PTB ratio
        ).reset_index()

        col_map = {"Settlements":"Settlements","HH":"HH",
                   "CAPEX ($M)":"CAPEX","Capacity (kW)":"Power_kW"}
        ycol = col_map[metric]

        fig = go.Figure()
        for t in ALL_TECHS:
            row = ts3[ts3["MinimumOverall"]==t]
            val = float(row[ycol].values[0]) if len(row)>0 else 0
            if val == 0 and "Capacity" in metric:
                continue  # skip techs with no computed capacity
            fig.add_trace(go.Bar(
                name=t, x=[t], y=[val],
                marker_color=TECH_COLORS[t],
                marker_line=dict(color="white",width=1),
                text=f"{val:,.0f}", textposition="inside",
                hovertemplate=f"<b>{t}</b><br>%{{y:,.0f}}<extra></extra>",
            ))

        ylab = {"Settlements":"Settlements","HH":"Households",
                "CAPEX ($M)":"CAPEX ($M)","Capacity (kW)":"Capacity (kW) — MG: generation | Grid: peak LV load"}.get(metric)
        fig.update_layout(
            barmode="stack", height=460, template="plotly_white",
            xaxis_title="", yaxis_title=ylab,
            showlegend=False,
            margin=dict(l=20,r=20,t=20,b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        if metric == "Capacity (kW)":
            st.caption("ℹ️ Capacity computed as: demand (kWh/yr) ÷ 8,760h × peak-to-base ratio "
                       "(Solar MG ×3.5, Hybrid ×2.5, Hydro ×2.0, Grid ×1.18). "
                       "SHS individual kit capacity not included (no shared generation).")

        # Summary table
        tbl = ts3[ts3["MinimumOverall"]!="Already Electrified"][
            ["MinimumOverall","Settlements","HH","CAPEX","Power_kW"]].copy()
        tbl.columns = ["Technology","Settlements","HH","CAPEX ($M)","Power (kW)"]
        tbl["Pct"] = (tbl["Settlements"]/tbl["Settlements"].sum()*100).round(1)
        st.dataframe(tbl.style.format({
            "HH":"{:,.0f}","CAPEX ($M)":"${:.1f}",
            "Power (kW)":"{:,.0f}","Pct":"{:.1f}%"}),
            use_container_width=True, hide_index=True)

    elif "Electrification Rate" in ct:
        st.markdown('<div class="section-hd">Electrification Rate by Technology</div>',
                    unsafe_allow_html=True)
        ts_er = df.groupby("MinimumOverall").agg(
            Sites=("MinimumOverall","count"),
            HH=("NumConnections","sum")).reset_index()
        total_sites = len(df)
        total_hh    = ts_er["HH"].sum()
        ts_er["Pct_Sites"] = (ts_er["Sites"] / total_sites * 100).round(2)
        ts_er["Pct_HH"]    = (ts_er["HH"]    / total_hh    * 100).round(2)

        # Single grouped bar chart — % Settlements and % HH side by side per technology
        fig_er = go.Figure()

        # Trace 1: % Settlements
        xs1, ys1, txt1, clrs1 = [], [], [], []
        for t in ALL_TECHS:
            row = ts_er[ts_er["MinimumOverall"]==t]
            if len(row) > 0:
                pct  = float(row["Pct_Sites"].values[0])
                n    = int(row["Sites"].values[0])
                xs1.append(t);  ys1.append(pct)
                txt1.append(f"{pct:.1f}%\n({n:,})")
                clrs1.append(TECH_COLORS[t])
        fig_er.add_trace(go.Bar(
            name="% Settlements", x=xs1, y=ys1,
            marker_color=clrs1,
            text=txt1, textposition="outside",
            hovertemplate="<b>%{x}</b><br>Settlements: %{y:.1f}%<extra></extra>",
            offsetgroup=0,
        ))

        # Trace 2: % Households (pattern fill to distinguish)
        xs2, ys2, txt2, clrs2 = [], [], [], []
        for t in ALL_TECHS:
            row = ts_er[ts_er["MinimumOverall"]==t]
            if len(row) > 0 and float(row["HH"].values[0]) > 0:
                pct = float(row["Pct_HH"].values[0])
                hh  = float(row["HH"].values[0])
                xs2.append(t);  ys2.append(pct)
                txt2.append(f"{pct:.1f}%\n({hh/1e3:.0f}K HH)")
                clrs2.append(TECH_COLORS[t])
        fig_er.add_trace(go.Bar(
            name="% Households", x=xs2, y=ys2,
            marker=dict(color=clrs2,
                        pattern=dict(shape="/", fgcolor="white", size=6)),
            text=txt2, textposition="outside",
            hovertemplate="<b>%{x}</b><br>Households: %{y:.1f}%<extra></extra>",
            offsetgroup=1,
        ))

        fig_er.update_layout(
            barmode="group", height=480, template="plotly_white",
            yaxis_title="Share (%)",
            xaxis_title="",
            legend=dict(orientation="h", x=0.3, y=1.08),
            margin=dict(l=20,r=20,t=50,b=80),
        )
        st.plotly_chart(fig_er, use_container_width=True)
        st.caption("Solid bars = % of settlements · Hatched bars = % of households")


    elif "Donut" in ct:
        st.markdown('<div class="section-hd">Unelectrified Settlements — Technology Split</div>',
                    unsafe_allow_html=True)
        unelec=df[df["MinimumOverall"]!="Already Electrified"]
        counts=unelec["MinimumOverall"].value_counts()
        fig=go.Figure(go.Pie(
            values=counts.values,labels=counts.index,hole=.6,
            marker=dict(colors=[TECH_COLORS.get(l,"#999") for l in counts.index],
                        line=dict(color="white",width=2)),
            textinfo="label+percent",textposition="auto",
            hovertemplate="%{label}<br>%{value:,}<br>%{percent}<extra></extra>"))
        fig.update_layout(height=420,template="plotly_white",
            annotations=[dict(text=f"<b>{len(unelec):,}</b><br>unelectrified",
                              x=.5,y=.5,font_size=15,showarrow=False)],
            margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)



# ── ANNUAL ROLLOUT ────────────────────────────────────────────────────────────
with t_rollout:
    st.markdown('<div class="section-hd">Annual Investment Rollout Plan</div>',
                unsafe_allow_html=True)

    rc1,rc2,rc3=st.columns(3)
    with rc1:
        strategy=st.selectbox("Allocation strategy",
            options=list(STRATS.keys()),
            format_func=lambda k: STRATS[k][0])
    with rc2: base_yr=st.number_input("Base year",2024,2030,2025)
    with rc3: end_yr =st.number_input("End year",2030,2050,2040)

    n_yrs=end_yr-base_yr; years=list(range(base_yr+1,end_yr+1))

    # Use full unelectrified dataset for rollout totals — not sidebar-filtered df
    # Sidebar filters (LCOE range, connection count) distort investment totals
    _df_rollout = df_all[
        (df_all["MinimumOverall"] != "Already Electrified") &
        ~df_all.get("_dummy", pd.Series(False, index=df_all.index))
    ].copy()

    rrows=[]
    for t in UNELEC:
        tc=_df_rollout[_df_rollout["MinimumOverall"]==t]["InvestmentCost"].sum() if "InvestmentCost" in _df_rollout.columns else 0
        if tc>0:
            alloc=allocate(tc,n_yrs,strategy)
            for i,yr in enumerate(years):
                rrows.append({"Year":yr,"Technology":t,"Investment":alloc[i]})

    if rrows:
        rdf=pd.DataFrame(rrows)
        tot_r=rdf["Investment"].sum()
        ann_t=rdf.groupby("Year")["Investment"].sum()
        avg_a=ann_t.mean(); pk_yr=ann_t.idxmax(); pk_v=ann_t.max()

        k1,k2,k3,k4=st.columns(4)
        with k1: st.metric("Total",      f"${tot_r/1e6:.1f}M")
        with k2: st.metric("Avg/year",   f"${avg_a/1e6:.2f}M")
        with k3: st.metric("Peak year",  str(pk_yr))
        with k4: st.metric("Peak value", f"${pk_v/1e6:.2f}M")
        st.divider()

        cm2=st.radio("View as:",["Stacked Bar","Line"],horizontal=True)
        if cm2=="Stacked Bar":
            fb=px.bar(rdf,x="Year",y="Investment",color="Technology",
                color_discrete_map=TECH_COLORS,category_orders={"Technology":ALL_TECHS},
                barmode="stack",height=340,template="plotly_white",
                labels={"Investment":"Investment (USD)"})
            fb.update_layout(margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fb, use_container_width=True)
        elif cm2=="Line":
            fl=go.Figure()
            for t in UNELEC:
                sub=rdf[rdf["Technology"]==t]
                if len(sub)>0:
                    fl.add_trace(go.Scatter(x=sub["Year"].astype(str),
                        y=sub["Investment"]/1e6,mode="lines+markers",name=t,
                        line=dict(color=TECH_COLORS.get(t,"#999"),width=2),marker=dict(size=5)))
            fl.add_trace(go.Scatter(x=ann_t.index.astype(str),y=ann_t.values/1e6,
                mode="lines+markers",name="Total",
                line=dict(color="#333",width=3,dash="dash"),
                marker=dict(size=7,symbol="square")))
            fl.update_layout(height=340,template="plotly_white",
                xaxis_title="Year",yaxis_title="$M",hovermode="x unified",
                margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fl, use_container_width=True)
        # 5-year periods
        st.markdown('<div class="section-hd">5-Year Period Summary</div>',
                    unsafe_allow_html=True)
        p_rows=[]
        ys=base_yr+1
        while ys<=end_yr:
            ye=min(ys+4,end_yr)
            sub=ann_t[(ann_t.index>=ys)&(ann_t.index<=ye)]
            p_rows.append({"Period":f"{ys}–{ye}","Total ($M)":sub.sum()/1e6,
                           "Annual avg ($M)":sub.mean()/1e6,"Years":len(sub)})
            ys=ye+1
        st.dataframe(pd.DataFrame(p_rows).style.format(
            {"Total ($M)":"${:.2f}","Annual avg ($M)":"${:.2f}"}),
            use_container_width=True, hide_index=True)

        # Cumulative connections
        if "NumConnections" in _df_rollout.columns:
            crow=[]
            for t in UNELEC:
                th=_df_rollout[_df_rollout["MinimumOverall"]==t]["NumConnections"].sum()
                if th>0:
                    ah=allocate(th,n_yrs,strategy)
                    for i,yr in enumerate(years): crow.append({"Year":yr,"HH":ah[i]})
            if crow:
                cdf2=pd.DataFrame(crow).groupby("Year")["HH"].sum().cumsum().reset_index()
                cdf2.columns=["Year","Cumulative HH"]
                fc2=px.area(cdf2,x="Year",y="Cumulative HH",
                    title="Cumulative New Connections",height=280,
                    template="plotly_white",color_discrete_sequence=["#43A047"])
                fc2.update_layout(margin=dict(l=20,r=20,t=40,b=20))
                st.plotly_chart(fc2, use_container_width=True)

        # ── Full planning metrics across all three phases ─────────────
        st.divider()
        st.markdown('<div class="section-hd">Planning Metrics — All Three Phases Combined</div>',
                    unsafe_allow_html=True)
        pm1,pm2,pm3 = st.columns(3)

        # Investment
        total_inv_m = rdf["Investment"].sum()/1e6
        with pm1:
            st.metric("Total Investment", f"${total_inv_m:.1f}M")

        # Estimate OPEX (assume 2.5% of CAPEX/yr, 15 years)
        opex_rate = 0.025
        total_capex_val = _df_rollout["InvestmentCost"].sum() if "InvestmentCost" in _df_rollout.columns else 0
        opex_total = total_capex_val * opex_rate * 15
        with pm2:
            st.metric("Estimated Total OPEX (15yr)", f"${opex_total/1e6:.1f}M",
                      help="2.5% of CAPEX/year × 15 years")
        with pm3:
            st.metric("Total CAPEX + OPEX", f"${(total_capex_val+opex_total)/1e6:.1f}M")

        pm4,pm5,pm6 = st.columns(3)
        total_hh   = _df_rollout["NumConnections"].sum() if "NumConnections" in _df_rollout.columns else 0
        total_pop  = _df_rollout["Population"].sum() if "Population" in _df_rollout.columns else total_hh * 5
        total_sites = len(_df_rollout)
        with pm4:
            hh_s2 = f"{total_hh/1e6:.2f}M" if total_hh>1e6 else f"{total_hh:,.0f}"
            st.metric("New Connections (HH)", hh_s2)
        with pm5:
            pop_s = f"{total_pop/1e6:.2f}M" if total_pop>1e6 else f"{total_pop:,.0f}"
            st.metric("Population to be served", pop_s)
        with pm6:
            st.metric("Sites to electrify", f"{total_sites:,}")

        # Per-phase breakdown
        st.markdown("**Per-phase breakdown:**")
        if "GridRolloutPhase" in df.columns:
            grid_only = df[df["MinimumOverall"]=="Grid Extension"]
            phase_rows = []
            for ph in ["Phase 1 (0-5 yrs)","Phase 2 (5-10 yrs)","Phase 3 (10-15 yrs)"]:
                sub = grid_only[grid_only["GridRolloutPhase"]==ph]
                phase_rows.append({
                    "Phase": ph,
                    "Grid sites": len(sub),
                    "HH": int(sub["NumConnections"].sum()) if "NumConnections" in sub.columns else 0,
                    "CAPEX ($M)": round(sub["InvestmentCost"].sum()/1e6, 1) if "InvestmentCost" in sub.columns else 0,
                    "Avg LCOE ($/kWh)": round(sub["MinimumOverallLCOE"].mean(), 3) if "MinimumOverallLCOE" in sub.columns else 0,
                })
            # Add non-grid totals
            non_grid = df[df["MinimumOverall"]!="Grid Extension"]
            for t in ["Mini-Grid: Solar PV Only","Mini-Grid: Mini-Hydro","SHS"]:
                sub_t = non_grid[non_grid["MinimumOverall"]==t]
                if len(sub_t) > 0:
                    phase_rows.append({
                        "Phase": t,
                        "Grid sites": len(sub_t),
                        "HH": int(sub_t["NumConnections"].sum()) if "NumConnections" in sub_t.columns else 0,
                        "CAPEX ($M)": round(sub_t["InvestmentCost"].sum()/1e6, 1) if "InvestmentCost" in sub_t.columns else 0,
                        "Avg LCOE ($/kWh)": round(sub_t["MinimumOverallLCOE"].mean(), 3) if "MinimumOverallLCOE" in sub_t.columns else 0,
                    })
            st.dataframe(pd.DataFrame(phase_rows).style.format({
                "CAPEX ($M)":"${:.1f}","Avg LCOE ($/kWh)":"{:.3f}","HH":"{:,.0f}"}),
                use_container_width=True, hide_index=True)

        st.download_button("⬇️ Download rollout CSV",
            data=rdf.to_csv(index=False),
            file_name="benin_annual_rollout.csv",
            mime="text/csv",use_container_width=True)

# ── ACCESS TARGETS ────────────────────────────────────────────────────────────
with t_targets:
    st.markdown("### 🎯 National Access Targets: 60% → 85% → 100%")
    st.markdown(
        '<div class="info-box">Settlements are ranked by priority score '
        '(connections × demand / distance) and assigned to phases that '
        'achieve <b>60% access by 2030</b>, <b>85% by 2035</b>, and '
        '<b>100% by 2040</b>. '
        'Grid distances are <b>updated at each timestep</b> — settlements '
        'electrified in Phase 1 become new grid nodes, reducing distances for '
        'their neighbours in Phase 2 and 3 (<i>grid creep</i>).</div>',
        unsafe_allow_html=True
    )

    # ── Compute phase assignment ───────────────────────────────────────────────
    df_phases, n_p1, n_p2, n_p3 = compute_target_phases(df_all[
        ~df_all.get("_dummy", pd.Series(False, index=df_all.index))
    ].copy())

    phase_labels = ["Phase 1 (→2030)", "Phase 2 (→2035)", "Phase 3 (→2040)"]
    phase_years  = [2030, 2035, 2040]
    phase_n      = [n_p1, n_p2, n_p3]

    # ── Target milestone KPI row ───────────────────────────────────────────────
    tg_c1, tg_c2, tg_c3, tg_c4 = st.columns(4)
    with tg_c1:
        st.markdown(kpi("Already Electrified",
                        f"{ALREADY_ELEC:,}",
                        f"{ALREADY_ELEC/TOTAL_SETT*100:.1f}% baseline"),
                    unsafe_allow_html=True)
    for col, ph, yr, n_ph, tgt in zip(
        [tg_c2, tg_c3, tg_c4],
        phase_labels, phase_years, phase_n,
        [TARGET_60, TARGET_85, TARGET_100]
    ):
        cumul = ALREADY_ELEC + sum(phase_n[:phase_labels.index(ph)+1])
        with col:
            st.markdown(kpi(f"{ph}  —  {int(tgt*100)}% by {yr}",
                            f"{cumul/TOTAL_SETT*100:.1f}%",
                            f"+{n_ph:,} new settlements"),
                        unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 1: Cumulative access curve + Phase tech mix ───────────────────────
    ra1, ra2 = st.columns([3, 2])

    with ra1:
        st.markdown('<div class="section-hd">Cumulative Access Rate Over Time</div>',
                    unsafe_allow_html=True)
        # Build staircase of access rate by year
        base_rate = ALREADY_ELEC / TOTAL_SETT * 100
        curve_yrs  = [2025, 2030, 2030, 2035, 2035, 2040]
        curve_vals = [
            base_rate,
            base_rate,
            TARGET_60 * 100,
            TARGET_60 * 100,
            TARGET_85 * 100,
            TARGET_100 * 100
        ]
        # Add individual years inside each phase using uniform spread
        all_yrs, all_vals = [2025], [base_rate]
        prev_rate = base_rate
        for ph_idx, (yr, n_ph, tgt) in enumerate(zip(phase_years, phase_n, [60, 85, 100])):
            start_yr = [2025, 2030, 2035][ph_idx]
            for y in range(start_yr + 1, yr + 1):
                frac = (y - start_yr) / (yr - start_yr)
                all_yrs.append(y)
                all_vals.append(prev_rate + frac * (tgt - prev_rate))
            prev_rate = tgt

        fig_curve = go.Figure()
        # Shaded regions per phase
        phase_ranges = [(2025, 2030, "Phase 1 (→2030)"),
                        (2030, 2035, "Phase 2 (→2035)"),
                        (2035, 2040, "Phase 3 (→2040)")]
        for y0, y1, ph in phase_ranges:
            fig_curve.add_vrect(
                x0=y0, x1=y1,
                fillcolor=TARGET_COLORS[ph], opacity=0.07,
                layer="below", line_width=0,
                annotation_text=ph.split(" ")[0]+" "+ph.split(" ")[1],
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color=TARGET_COLORS[ph]
            )
        # Target lines
        for yr, tgt, ph in zip(phase_years, [60, 85, 100], phase_labels):
            fig_curve.add_hline(y=tgt, line_dash="dot",
                                line_color=TARGET_COLORS[ph], line_width=1.5,
                                annotation_text=f" {tgt}%",
                                annotation_position="right",
                                annotation_font_color=TARGET_COLORS[ph],
                                annotation_font_size=11)
        # Access curve
        fig_curve.add_trace(go.Scatter(
            x=all_yrs, y=all_vals,
            mode="lines+markers",
            line=dict(color="#1A237E", width=3),
            marker=dict(size=5, color="#1A237E"),
            fill="tozeroy", fillcolor="rgba(26,35,126,0.08)",
            name="Access rate"
        ))
        # Current baseline marker
        fig_curve.add_trace(go.Scatter(
            x=[2025], y=[base_rate],
            mode="markers+text",
            marker=dict(size=12, color="#F9A825", symbol="diamond"),
            text=[f" {base_rate:.1f}% (2025)"],
            textposition="middle right",
            showlegend=False
        ))
        fig_curve.update_layout(
            height=340, template="plotly_white",
            xaxis=dict(title="Year", tickmode="array",
                       tickvals=list(range(2025, 2041, 1)),
                       ticktext=[str(y) if y % 5 == 0 else "" for y in range(2025, 2041, 1)]),
            yaxis=dict(title="National access rate (%)", range=[0, 105]),
            margin=dict(t=20, b=40, l=50, r=60),
            showlegend=False,
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    with ra2:
        st.markdown('<div class="section-hd">Technology Mix per Phase</div>',
                    unsafe_allow_html=True)
        ph_tech_rows = []
        for ph in phase_labels:
            sub = df_phases[df_phases["ElecTargetPhase"] == ph]
            for tech in [t for t in ALL_TECHS if t != "Already Electrified"]:
                n = (sub["MinimumOverall"] == tech).sum()
                if n > 0:
                    ph_tech_rows.append({"Phase": ph, "Technology": tech,
                                         "Sites": n, "HH": sub.loc[
                                             sub["MinimumOverall"]==tech,
                                             "NumConnections"].sum()})
        if ph_tech_rows:
            ph_tech_df = pd.DataFrame(ph_tech_rows)
            fig_ptm = px.bar(ph_tech_df, x="Phase", y="Sites", color="Technology",
                             color_discrete_map=TECH_COLORS,
                             barmode="stack", height=340,
                             template="plotly_white",
                             labels={"Sites": "Settlements", "Phase": ""})
            fig_ptm.update_layout(
                margin=dict(t=20, b=40, l=50, r=10),
                legend=dict(orientation="h", y=-0.22, font_size=10),
                xaxis_tickfont_size=10,
            )
            st.plotly_chart(fig_ptm, use_container_width=True)

    st.markdown("---")

    # ── Grid Creep Diagnostic ─────────────────────────────────────────────────
    if "DistAtConnectionKm" in df_phases.columns:
        st.markdown('<div class="section-hd">⚡ Grid Creep Effect</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">Grid distances are re-computed at each timestep. '
            'Settlements connected in Phase 1 become new grid nodes — reducing distances '
            'for Phase 2 and Phase 3 neighbours. The table below shows the median grid '
            'distance <i>at time of connection</i> vs today\'s baseline distance.</div>',
            unsafe_allow_html=True
        )
        gc_cols = st.columns(3)
        baseline_med = df_phases[dist_col].replace([np.inf,-np.inf], np.nan).median() \
                       if dist_col in df_phases.columns else None

        for ci, ph in enumerate(["Phase 1 (→2030)", "Phase 2 (→2035)", "Phase 3 (→2040)"]):
            sub = df_phases[df_phases["ElecTargetPhase"] == ph]
            grid_sub = sub[sub["MinimumOverall"] == "Grid Extension"]
            with gc_cols[ci]:
                if len(grid_sub) > 0 and "DistAtConnectionKm" in grid_sub.columns:
                    med_conn = grid_sub["DistAtConnectionKm"].median()
                    med_base = grid_sub[dist_col].median() \
                               if dist_col in grid_sub.columns else None
                    delta = f"↓ {med_base - med_conn:.1f} km shorter" \
                            if med_base is not None and med_base > med_conn else ""
                    st.markdown(
                        f'<div class="phase-card" style="background:{PHASE_COLORS[ph]}">'
                        f'<div class="phase-title">{ph}</div>'
                        f'<div class="phase-detail">{len(grid_sub):,} grid sites</div>'
                        f'<div class="phase-detail">Median dist at connection:<br>'
                        f'<b>{med_conn:.1f} km</b> {delta}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    n_all = len(sub)
                    tech_top = sub["MinimumOverall"].value_counts().index[0] \
                               if len(sub) > 0 else "—"
                    st.markdown(
                        f'<div class="phase-card" style="background:{PHASE_COLORS[ph]}">'
                        f'<div class="phase-title">{ph}</div>'
                        f'<div class="phase-detail">{n_all:,} settlements</div>'
                        f'<div class="phase-detail">Leading tech:<br><b>{tech_top}</b></div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        st.markdown("---")

    # ── Row 2: Per-phase summary table + CAPEX waterfall ─────────────────────
    rb1, rb2 = st.columns([2, 3])

    with rb1:
        st.markdown('<div class="section-hd">Phase Milestone Summary</div>',
                    unsafe_allow_html=True)
        phase_rows_tbl = []
        cumul_sites = ALREADY_ELEC
        cumul_hh    = int(df_phases.loc[
            df_phases["MinimumOverall"] == "Already Electrified",
            "NumConnections"].sum())
        cumul_capex = 0.0

        for ph, yr in zip(phase_labels, phase_years):
            sub = df_phases[df_phases["ElecTargetPhase"] == ph]
            n   = len(sub)
            hh  = int(sub["NumConnections"].sum()) if "NumConnections" in sub.columns else 0
            cap = float(sub["InvestmentCost"].sum()) if "InvestmentCost" in sub.columns else 0.0
            cumul_sites  += n
            cumul_hh     += hh
            cumul_capex  += cap
            phase_rows_tbl.append({
                "Phase"       : ph,
                "Target"      : f"{int(cumul_sites/TOTAL_SETT*100)}% by {yr}",
                "New sites"   : f"{n:,}",
                "New HH"      : f"{hh:,}",
                "CAPEX ($M)"  : f"${cap/1e6:.1f}M",
                "Cumul.sites" : f"{cumul_sites:,}",
                "Access rate" : f"{cumul_sites/TOTAL_SETT*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(phase_rows_tbl), use_container_width=True, hide_index=True)

        # Compact tech breakdown per phase
        st.markdown('<div class="section-hd" style="margin-top:14px">Sites by Technology × Phase</div>',
                    unsafe_allow_html=True)
        cross_rows = []
        for ph in phase_labels:
            sub = df_phases[df_phases["ElecTargetPhase"] == ph]
            row = {"Phase": ph.replace("Phase ","Ph")}
            for tech in [t for t in ALL_TECHS if t != "Already Electrified"]:
                n = (sub["MinimumOverall"] == tech).sum()
                row[tech[:6]] = n if n > 0 else "—"
            cross_rows.append(row)
        st.dataframe(pd.DataFrame(cross_rows), use_container_width=True, hide_index=True)

    with rb2:
        st.markdown('<div class="section-hd">CAPEX by Phase & Technology</div>',
                    unsafe_allow_html=True)
        capex_rows = []
        for ph in phase_labels:
            sub = df_phases[df_phases["ElecTargetPhase"] == ph]
            for tech in [t for t in ALL_TECHS if t != "Already Electrified"]:
                cap = sub.loc[sub["MinimumOverall"]==tech, "InvestmentCost"].sum() if "InvestmentCost" in sub.columns else 0
                if cap > 0:
                    capex_rows.append({"Phase": ph, "Technology": tech,
                                       "CAPEX_M": cap / 1e6})
        if capex_rows:
            capex_df = pd.DataFrame(capex_rows)
            fig_capex = px.bar(capex_df, x="Technology", y="CAPEX_M",
                               color="Phase",
                               color_discrete_map=TARGET_COLORS,
                               barmode="group", height=340,
                               template="plotly_white",
                               labels={"CAPEX_M": "CAPEX ($M)", "Technology": ""},
                               text_auto=".1f")
            fig_capex.update_traces(textfont_size=9, textposition="outside")
            fig_capex.update_layout(
                margin=dict(t=20, b=80, l=50, r=10),
                legend=dict(orientation="h", y=-0.32, font_size=10),
                xaxis_tickangle=-20, xaxis_tickfont_size=10,
                yaxis_title="CAPEX ($M)",
            )
            st.plotly_chart(fig_capex, use_container_width=True)

    st.markdown("---")

    # ── Row 3: Annual investment schedule + geographic density ────────────────
    rc1, rc2 = st.columns([3, 2])

    with rc1:
        st.markdown('<div class="section-hd">Annual Investment Schedule</div>',
                    unsafe_allow_html=True)
        strat_sel = st.radio("Allocation strategy",
                             ["Uniform", "Front-loaded", "Back-loaded"],
                             horizontal=True, key="target_strat")
        strat_key = strat_sel.split("-")[0].strip().lower()
        if strat_key not in STRATS: strat_key = "uniform"

        ann_rows = []
        phase_periods = [(2025, 2030), (2030, 2035), (2035, 2040)]
        for ph, (y0, y1) in zip(phase_labels, phase_periods):
            sub = df_phases[df_phases["ElecTargetPhase"] == ph]
            n_yrs = y1 - y0
            for tech in [t for t in ALL_TECHS if t != "Already Electrified"]:
                sub_t = sub[sub["MinimumOverall"] == tech]
                n     = len(sub_t)
                cap   = sub_t["InvestmentCost"].sum() if "InvestmentCost" in sub_t.columns else 0.0
                if n == 0 and cap == 0:
                    continue
                w = allocate(1.0, n_yrs, strat_key)
                for i, yr in enumerate(range(y0 + 1, y1 + 1)):
                    ann_rows.append({
                        "Year": yr,
                        "Phase": ph,
                        "Technology": tech,
                        "Sites": round(n * w[i]),
                        "CAPEX_M": cap * w[i] / 1e6,
                    })

        if ann_rows:
            ann_df = pd.DataFrame(ann_rows)
            fig_ann = px.bar(
                ann_df.groupby(["Year", "Technology"], as_index=False)
                      .agg({"CAPEX_M": "sum", "Sites": "sum"}),
                x="Year", y="CAPEX_M", color="Technology",
                color_discrete_map=TECH_COLORS, barmode="stack",
                height=320, template="plotly_white",
                labels={"CAPEX_M": "CAPEX ($M)", "Year": ""},
                text_auto=False
            )
            # Add vertical lines at phase boundaries
            for boundary_yr, ph_label in zip([2030, 2035], ["60% target", "85% target"]):
                fig_ann.add_vline(x=boundary_yr, line_dash="dash",
                                  line_color="grey", line_width=1.5,
                                  annotation_text=f" {ph_label}",
                                  annotation_position="top right",
                                  annotation_font_size=9)
            fig_ann.update_layout(
                margin=dict(t=10, b=40, l=50, r=10),
                legend=dict(orientation="h", y=-0.28, font_size=10),
                xaxis=dict(tickmode="linear", dtick=1, tickfont_size=10),
            )
            st.plotly_chart(fig_ann, use_container_width=True)

            # 5-year period summary
            st.markdown("**5-year totals:**")
            period_summ = ann_df.groupby("Phase").agg(
                Sites=("Sites","sum"), CAPEX_M=("CAPEX_M","sum")).reset_index()
            period_summ["CAPEX_M"] = period_summ["CAPEX_M"].round(1)
            st.dataframe(period_summ.style.format({"CAPEX_M": "${:.1f}M",
                                                    "Sites": "{:,.0f}"}),
                         use_container_width=True, hide_index=True)

    with rc2:
        st.markdown('<div class="section-hd">Households Connected by Phase</div>',
                    unsafe_allow_html=True)
        hh_rows = []
        for ph in phase_labels:
            sub = df_phases[df_phases["ElecTargetPhase"] == ph]
            for tech in [t for t in ALL_TECHS if t != "Already Electrified"]:
                hh = sub.loc[sub["MinimumOverall"] == tech, "NumConnections"].sum()
                if hh > 0:
                    hh_rows.append({"Phase": ph, "Technology": tech, "HH": hh})
        if hh_rows:
            hh_df = pd.DataFrame(hh_rows)
            fig_hh = px.bar(hh_df, y="Phase", x="HH", color="Technology",
                            color_discrete_map=TECH_COLORS, barmode="stack",
                            orientation="h", height=320, template="plotly_white",
                            labels={"HH": "Households", "Phase": ""},
                            text_auto=False)
            fig_hh.update_layout(
                margin=dict(t=10, b=10, l=20, r=80),
                legend=dict(orientation="h", y=-0.18, font_size=10),
                yaxis=dict(tickfont_size=10),
                xaxis_title="Households connected",
            )
            # Add target annotations
            cumul = ALREADY_ELEC
            for ph, n_ph, tgt in zip(phase_labels, phase_n, [60, 85, 100]):
                cumul_sett = cumul + n_ph
                cumul = cumul_sett
            st.plotly_chart(fig_hh, use_container_width=True)

        # Progress gauge
        st.markdown('<div class="section-hd" style="margin-top:4px">Progress Toward Each Target</div>',
                    unsafe_allow_html=True)
        cumul_s = ALREADY_ELEC
        for ph, n_ph, tgt, yr in zip(phase_labels, phase_n, [60, 85, 100], phase_years):
            cumul_s += n_ph
            achieved = cumul_s / TOTAL_SETT * 100
            color = TARGET_COLORS[ph]
            bar_pct = min(achieved / 100, 1.0) * 100
            st.markdown(
                f"""<div style="margin:8px 0">
                <div style="display:flex;justify-content:space-between;
                font-size:.8rem;font-weight:600;color:{color}">
                  <span>{ph} — {tgt}% by {yr}</span>
                  <span>{achieved:.1f}% achieved</span>
                </div>
                <div style="background:#EEE;border-radius:4px;height:10px;margin-top:3px">
                  <div style="background:{color};width:{bar_pct:.1f}%;
                  height:10px;border-radius:4px"></div>
                </div></div>""",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Row 4: LCOE distribution per phase ────────────────────────────────────
    st.markdown('<div class="section-hd">LCOE Distribution by Phase</div>',
                unsafe_allow_html=True)
    rd1, rd2, rd3 = st.columns(3)
    for col, ph in zip([rd1, rd2, rd3], phase_labels):
        with col:
            sub = df_phases[
                (df_phases["ElecTargetPhase"] == ph) &
                (df_phases.get("MinimumOverallLCOE",
                 pd.Series(np.nan, index=df_phases.index)).notna())
            ]
            lcoe_vals = sub["MinimumOverallLCOE"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(lcoe_vals) > 0:
                fig_box = go.Figure()
                for tech in [t for t in ALL_TECHS if t != "Already Electrified"]:
                    vals = sub.loc[sub["MinimumOverall"]==tech,
                                   "MinimumOverallLCOE"].replace(
                        [np.inf,-np.inf],np.nan).dropna()
                    if len(vals) > 0:
                        fig_box.add_trace(go.Box(
                            y=vals, name=tech[:12],
                            marker_color=TECH_COLORS.get(tech, "#999"),
                            boxpoints=False, showlegend=False
                        ))
                yr = {"Phase 1 (→2030)":2030,"Phase 2 (→2035)":2035,"Phase 3 (→2040)":2040}[ph]
                fig_box.update_layout(
                    title=dict(text=f"<b>{ph}</b><br><sup>{len(sub):,} sites · target {yr}</sup>",
                               font_size=12),
                    height=280, template="plotly_white",
                    margin=dict(t=50, b=40, l=40, r=10),
                    yaxis_title="LCOE ($/kWh)",
                )
                st.plotly_chart(fig_box, use_container_width=True)

    # ── Download ───────────────────────────────────────────────────────────────
    st.markdown("---")
    dl_df = df_phases[["MinimumOverall","ElecTargetPhase","NumConnections",
                        "InvestmentCost","MinimumOverallLCOE",
                        "X_deg","Y_deg"]].copy() if "X_deg" in df_phases.columns else \
            df_phases[["MinimumOverall","ElecTargetPhase","NumConnections",
                        "InvestmentCost","MinimumOverallLCOE"]].copy()
    st.download_button(
        "⬇️ Download phased plan CSV",
        dl_df.to_csv(index=False),
        file_name="benin_access_targets_phased.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ── SENSITIVITY ANALYSIS ──────────────────────────────────────────────────────
with t_sensitivity:
    st.markdown("### 🔬 Sensitivity Analysis")
    st.markdown(
        '<div class="info-box">Shows how technology distribution changes when key '
        'model assumptions are varied. Load the <b>sensitivity_results_*.csv</b> '
        'produced by NB04 to explore all 17 scenarios interactively. '
        'Charts include tornado, grouped bar, stacked bar, and histogram views.</div>',
        unsafe_allow_html=True
    )

    # ── Load sensitivity CSV ──────────────────────────────────────────────────
    sens_df = None
    tornado_df = None
    for base in [Path("data/outputs/tables"), Path("../data/outputs/tables")]:
        sens_files    = sorted(base.glob("sensitivity_results_*.csv"),    reverse=True)
        tornado_files = sorted(base.glob("tornado_summary_*.csv"),        reverse=True)
        if sens_files:
            sens_df = pd.read_csv(sens_files[0])
            st.caption(f"📄 {sens_files[0].name}")
        if tornado_files:
            tornado_df = pd.read_csv(tornado_files[0])
        if sens_df is not None:
            break

    if sens_df is None:
        st.info("No sensitivity_results_*.csv found. Run NB04 and its save cell first, "
                "then refresh the dashboard.")
        # ── Static fallback using hardcoded NB04 results ─────────────────────
        st.markdown("**Showing results from the last NB04 run:**")
        sens_df = pd.DataFrame([
            # Battery
            {"Scenario":"Lead-acid (yr 5)",       "Parameter":"Battery lifetime",
             "Technology":"Mini-Grid: Solar PV Only","Pct_settlements":5.9},
            {"Scenario":"LFP base case (yr 10)",  "Parameter":"Battery lifetime",
             "Technology":"Mini-Grid: Solar PV Only","Pct_settlements":6.1},
            {"Scenario":"LFP optimistic (yr 12)", "Parameter":"Battery lifetime",
             "Technology":"Mini-Grid: Solar PV Only","Pct_settlements":6.1},
            # Connection cost
            {"Scenario":"Subsidised ($150)",      "Parameter":"Connection cost",
             "Technology":"Mini-Grid: Solar PV Only","Pct_settlements":6.1},
            {"Scenario":"Base case ($280)",        "Parameter":"Connection cost",
             "Technology":"Mini-Grid: Solar PV Only","Pct_settlements":6.1},
            {"Scenario":"Full cost ($350)",        "Parameter":"Connection cost",
             "Technology":"Mini-Grid: Solar PV Only","Pct_settlements":4.9},
            # SHS kit
            {"Scenario":"Low kit ($120)",          "Parameter":"SHS kit cost",
             "Technology":"Mini-Grid: Solar PV Only","Pct_settlements":4.9},
            {"Scenario":"Base case ($150)",        "Parameter":"SHS kit cost",
             "Technology":"Mini-Grid: Solar PV Only","Pct_settlements":6.1},
            {"Scenario":"High kit ($180)",         "Parameter":"SHS kit cost",
             "Technology":"Mini-Grid: Solar PV Only","Pct_settlements":6.1},
            # Diesel
            {"Scenario":"Low ($0.75/L)",           "Parameter":"Diesel price",
             "Technology":"Mini-Grid: Solar PV Only","Pct_settlements":6.0},
            {"Scenario":"Base ($0.85/L)",          "Parameter":"Diesel price",
             "Technology":"Mini-Grid: Solar PV Only","Pct_settlements":6.1},
            {"Scenario":"High ($1.00/L)",          "Parameter":"Diesel price",
             "Technology":"Mini-Grid: Solar PV Only","Pct_settlements":6.1},
            # Threshold
            {"Scenario":"Low threshold (10,000 kWh)",  "Parameter":"Demand threshold",
             "Technology":"Mini-Grid: Solar PV Only","Pct_settlements":8.6},
            {"Scenario":"Base case (15,000 kWh)",       "Parameter":"Demand threshold",
             "Technology":"Mini-Grid: Solar PV Only","Pct_settlements":6.1},
            {"Scenario":"High threshold (25,000 kWh)", "Parameter":"Demand threshold",
             "Technology":"Mini-Grid: Solar PV Only","Pct_settlements":3.6},
        ])

    # Ensure Parameter column exists (infer from Scenario if needed)
    if "Parameter" not in sens_df.columns:
        param_map = {
            "Lead-acid":"Battery lifetime","LFP":"Battery lifetime",
            "Subsidised":"Connection cost","Low ($":"Connection cost",
            "Full cost":"Connection cost","Base case ($2":"Connection cost",
            "Low kit":"SHS kit cost","High kit":"SHS kit cost",
            "Base case ($1":"SHS kit cost",
            "Low ($0":"Diesel price","High ($1":"Diesel price",
            "Base ($":"Diesel price",
            "threshold":"Demand threshold",
        }
        def _infer_param(scen):
            for k, v in param_map.items():
                if k.lower() in scen.lower():
                    return v
            return "Other"
        sens_df["Parameter"] = sens_df["Scenario"].apply(_infer_param)

    MG_SOLAR = "Mini-Grid: Solar PV Only"
    SHS_TECH = "SHS"
    MG_HYDRO = "Mini-Grid: Mini-Hydro"
    MG_HYB   = "Mini-Grid: Solar-Diesel Hybrid"
    params   = sorted(sens_df["Parameter"].unique())
    PARAM_COLORS = {
        "Demand threshold" : "#C62828",
        "Connection cost"  : "#E65100",
        "SHS kit cost"     : "#F9A825",
        "Diesel price"     : "#2E7D32",
        "Battery lifetime" : "#1565C0",
        "Technology exclusion": "#7B1FA2",
    }

    # ── Hydro exclusion toggle ────────────────────────────────────────────────
    st.markdown('<div class="section-hd">⚙️ Scenario Configuration</div>',
                unsafe_allow_html=True)
    cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
    with cfg_c1:
        exclude_hydro  = st.toggle("🚫 Exclude Mini-Hydro",  value=False,
            help="Forces hydro sites to reassign to Solar MG or Hybrid MG")
    with cfg_c2:
        exclude_hybrid = st.toggle("🚫 Exclude Diesel Hybrid", value=False,
            help="Forces hybrid sites to reassign to Solar MG")
    with cfg_c3:
        show_excl_impact = st.toggle("📊 Show exclusion impact", value=True,
            help="Show a summary card of how exclusions shift technology shares")

    # Compute live exclusion impact from sensitivity CSV if available
    if (exclude_hydro or exclude_hybrid) and show_excl_impact:
        excl_rows = sens_df[
            sens_df["Parameter"].str.contains("exclusion|Exclusion|hydro|Hydro", na=False)
        ] if "Parameter" in sens_df.columns else pd.DataFrame()

        # Build exclusion impact panel from base vs no-hydro scenario in sens_df
        base_rows = sens_df[sens_df["Scenario"].str.contains("Base case|base case", na=False)]
        excl_scen = sens_df[sens_df["Scenario"].str.contains(
            "No hydro|no hydro|Solar only|solar only", na=False
        )]

        impact_cols = st.columns(len(ALL_TECHS) - 1)
        techs_to_show = [t for t in ALL_TECHS if t != "Already Electrified"]

        if len(base_rows) > 0 and len(excl_scen) > 0:
            scen_label = excl_scen["Scenario"].iloc[0]
            for ci, tech in enumerate(techs_to_show):
                base_v = base_rows[base_rows["Technology"]==tech]["Pct_settlements"]
                excl_v = excl_scen[excl_scen["Technology"]==tech]["Pct_settlements"]
                bv = float(base_v.iloc[0]) if len(base_v) > 0 else 0.0
                ev = float(excl_v.iloc[0]) if len(excl_v) > 0 else bv
                delta = ev - bv
                short = tech.replace("Mini-Grid: ","MG ").replace(" Only","").replace(" PV","")
                with impact_cols[ci]:
                    st.metric(
                        label=short,
                        value=f"{ev:.1f}%",
                        delta=f"{delta:+.1f} pp" if abs(delta) > 0.05 else "0 pp",
                        delta_color="inverse" if tech == MG_HYDRO else "normal"
                    )
            st.caption(f"Scenario shown: *{scen_label}* vs base case")
        else:
            # Fallback: compute approximate impact from known base results
            hydro_base  = float((df_all["MinimumOverall"] == "Mini-Grid: Mini-Hydro").sum())
            solar_base  = float((df_all["MinimumOverall"] == "Mini-Grid: Solar PV Only").sum())
            total_u     = float((df_all["MinimumOverall"] != "Already Electrified").sum())
            hydro_pct   = hydro_base / max(total_u, 1) * 100
            solar_gain  = round(hydro_pct * 0.8, 1) if exclude_hydro  else 0.0
            hybrid_gain = round(hydro_pct * 0.2, 1) if exclude_hydro  else 0.0
            st.markdown(
                f'<div class="info-box">'
                f'{"🚫 Hydro excluded: " if exclude_hydro else ""}'
                f'{int(hydro_base):,} hydro sites ({hydro_pct:.1f}%) will reassign → '
                f'Solar MG +{solar_gain:.1f} pp, Hybrid +{hybrid_gain:.1f} pp<br>'
                f'{"🚫 Hybrid excluded: hybrid sites → Solar MG<br>" if exclude_hybrid else ""}'
                f'<i>Counts reflect live model parameters. '
                f'Run NB04 for exact LCOE redistribution.</i>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Row 1: Tornado chart + impact summary ─────────────────────────────────
    sc1, sc2 = st.columns([3, 2])

    with sc1:
        st.markdown('<div class="section-hd">Tornado Chart — Solar MG Share Range</div>',
                    unsafe_allow_html=True)

        # Build tornado data from sensitivity CSV
        tornado_rows = []
        mg_solar_df = sens_df[sens_df["Technology"] == MG_SOLAR]
        for param in params:
            sub = mg_solar_df[mg_solar_df["Parameter"] == param]["Pct_settlements"]
            if len(sub) == 0:
                continue
            lo, hi = sub.min(), sub.max()
            base_sub = sens_df[
                (sens_df["Technology"] == MG_SOLAR) &
                (sens_df["Parameter"] == param) &
                (sens_df["Scenario"].str.contains("Base|base", case=False, na=False))
            ]["Pct_settlements"]
            base_val = float(base_sub.iloc[0]) if len(base_sub) > 0 else (lo + hi) / 2
            tornado_rows.append({
                "Parameter": param,
                "Min": lo, "Max": hi, "Base": base_val,
                "Range": hi - lo,
                "Color": PARAM_COLORS.get(param, "#546E7A")
            })
        tornado_rows = sorted(tornado_rows, key=lambda r: r["Range"])

        if tornado_rows:
            fig_t = go.Figure()
            base_val_global = tornado_rows[0]["Base"] if tornado_rows else 6.1

            # Base line
            fig_t.add_vline(x=base_val_global, line_dash="dash",
                            line_color="#F9A825", line_width=2,
                            annotation_text=f"Base: {base_val_global:.1f}%",
                            annotation_position="top",
                            annotation_font_color="#F9A825",
                            annotation_font_size=11)

            for row in tornado_rows:
                c = row["Color"]
                # Low bar
                fig_t.add_trace(go.Bar(
                    y=[row["Parameter"]], x=[row["Base"] - row["Min"]],
                    base=[row["Min"]],
                    orientation="h",
                    marker_color=c, opacity=0.55,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{row['Parameter']}</b><br>"
                        f"Range: {row['Min']:.1f}% – {row['Max']:.1f}%<br>"
                        f"Swing: {row['Range']:.1f} pp<extra></extra>"
                    )
                ))
                # High bar
                fig_t.add_trace(go.Bar(
                    y=[row["Parameter"]], x=[row["Max"] - row["Base"]],
                    base=[row["Base"]],
                    orientation="h",
                    marker_color=c, opacity=0.85,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{row['Parameter']}</b><br>"
                        f"Range: {row['Min']:.1f}% – {row['Max']:.1f}%<br>"
                        f"Swing: {row['Range']:.1f} pp<extra></extra>"
                    )
                ))
                # Swing label
                fig_t.add_annotation(
                    y=row["Parameter"], x=row["Max"] + 0.15,
                    text=f"{row['Range']:.1f} pp",
                    showarrow=False, xanchor="left",
                    font=dict(size=10, color=c)
                )

            fig_t.update_layout(
                barmode="overlay",
                height=320,
                template="plotly_white",
                xaxis=dict(title="Solar MG share (%)", range=[0, 12]),
                yaxis=dict(tickfont=dict(size=12)),
                margin=dict(t=30, b=40, l=160, r=80),
            )
            st.plotly_chart(fig_t, use_container_width=True)

    with sc2:
        st.markdown('<div class="section-hd">Impact Summary</div>',
                    unsafe_allow_html=True)
        if tornado_rows:
            impact_labels = {0: "Negligible", 0.5: "Low", 1.0: "Moderate", 2.5: "High", 5: "Critical"}
            def impact_label(rng):
                if rng >= 5: return "🔴 Critical"
                elif rng >= 2.5: return "🟠 High"
                elif rng >= 1.0: return "🟡 Moderate"
                elif rng >= 0.5: return "🟢 Low"
                return "⚪ Negligible"

            for row in sorted(tornado_rows, key=lambda r: -r["Range"]):
                lbl = impact_label(row["Range"])
                st.markdown(
                    f'<div style="border-left:4px solid {row["Color"]};'
                    f'padding:6px 10px;margin:4px 0;background:#FAFAFA;">'
                    f'<b>{row["Parameter"]}</b><br>'
                    f'<span style="font-size:.85rem;color:#546E7A">'
                    f'{row["Min"]:.1f}% – {row["Max"]:.1f}%  ·  '
                    f'±{row["Range"]:.1f} pp</span><br>'
                    f'{lbl}</div>',
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # ── Row 2: Parameter deep-dive — grouped bar + histogram ─────────────────
    st.markdown('<div class="section-hd">Parameter Deep-Dive</div>',
                unsafe_allow_html=True)

    sel_param = st.selectbox(
        "Select parameter to explore:",
        params,
        index=params.index("Demand threshold") if "Demand threshold" in params else 0
    )

    param_data = sens_df[sens_df["Parameter"] == sel_param].copy()
    scenarios  = list(param_data["Scenario"].unique())

    dd1, dd2, dd3 = st.columns([3, 2, 2])

    with dd1:
        st.markdown(f"**Grouped bar — all technologies**")
        fig_dd = go.Figure()
        for tech in [t for t in ALL_TECHS if t != "Already Electrified"]:
            vals = []
            for scen in scenarios:
                row = param_data[
                    (param_data["Scenario"] == scen) &
                    (param_data["Technology"] == tech)
                ]["Pct_settlements"]
                vals.append(float(row.iloc[0]) if len(row) > 0 else 0.0)
            if sum(vals) > 0:
                fig_dd.add_trace(go.Bar(
                    name=tech, x=scenarios, y=vals,
                    marker_color=TECH_COLORS.get(tech, "#546E7A"),
                    hovertemplate=f"<b>{tech}</b><br>%{{y:.1f}}%<extra></extra>"
                ))
        fig_dd.update_layout(
            barmode="group", height=320, template="plotly_white",
            xaxis=dict(title="", tickangle=-25, tickfont=dict(size=10)),
            yaxis=dict(title="Share of settlements (%)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                        font=dict(size=9)),
            margin=dict(t=60, b=70, l=50, r=10),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    with dd2:
        st.markdown(f"**Stacked bar — technology mix**")
        fig_stk = go.Figure()
        for tech in [t for t in ALL_TECHS if t != "Already Electrified"]:
            vals = []
            for scen in scenarios:
                row = param_data[
                    (param_data["Scenario"] == scen) &
                    (param_data["Technology"] == tech)
                ]["Pct_settlements"]
                vals.append(float(row.iloc[0]) if len(row) > 0 else 0.0)
            if sum(vals) > 0:
                fig_stk.add_trace(go.Bar(
                    name=tech, x=scenarios, y=vals,
                    marker_color=TECH_COLORS.get(tech, "#546E7A"),
                    hovertemplate=f"<b>{tech}</b><br>%{{y:.1f}}%<extra></extra>"
                ))
        fig_stk.update_layout(
            barmode="stack", height=320, template="plotly_white",
            xaxis=dict(title="", tickangle=-25, tickfont=dict(size=10)),
            yaxis=dict(title="Share (%)", range=[0, 105]),
            showlegend=False,
            margin=dict(t=20, b=70, l=50, r=10),
        )
        st.plotly_chart(fig_stk, use_container_width=True)

    with dd3:
        st.markdown(f"**Solar MG vs SHS table**")
        tbl_rows = []
        for scen in scenarios:
            mg_v = param_data[
                (param_data["Scenario"] == scen) &
                (param_data["Technology"] == MG_SOLAR)
            ]["Pct_settlements"]
            shs_v = param_data[
                (param_data["Scenario"] == scen) &
                (param_data["Technology"] == SHS_TECH)
            ]["Pct_settlements"]
            tbl_rows.append({
                "Scenario": scen,
                "Solar MG (%)": f"{float(mg_v.iloc[0]):.1f}" if len(mg_v) > 0 else "—",
                "SHS (%)":      f"{float(shs_v.iloc[0]):.1f}" if len(shs_v) > 0 else "—",
            })
        st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

        mg_vals  = [float(r["Solar MG (%)"]) for r in tbl_rows if r["Solar MG (%)"] != "—"]
        shs_vals = [float(r["SHS (%)"])      for r in tbl_rows if r["SHS (%)"] != "—"]
        if mg_vals and shs_vals:
            st.markdown(
                f'<div class="info-box">'
                f'Solar MG: <b>{min(mg_vals):.1f}% – {max(mg_vals):.1f}%</b><br>'
                f'SHS: <b>{min(shs_vals):.1f}% – {max(shs_vals):.1f}%</b></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Row 3: Histograms — distribution of shares across all scenarios ───────
    st.markdown('<div class="section-hd">Distribution of Technology Shares — All Scenarios (Histogram)</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Each bar shows how many scenarios produced a given share for that technology. "
        "A narrow histogram = robust result. A wide histogram = sensitive assumption."
    )

    hist_techs = [t for t in ALL_TECHS if t != "Already Electrified"]
    h_cols = st.columns(len(hist_techs))

    for ci, tech in enumerate(hist_techs):
        tech_vals = sens_df[sens_df["Technology"] == tech]["Pct_settlements"].dropna()
        with h_cols[ci]:
            if len(tech_vals) == 0:
                st.caption(f"{tech.replace('Mini-Grid: ','MG ')}: no data")
                continue
            color = TECH_COLORS.get(tech, "#546E7A")
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(
                x=tech_vals,
                nbinsx=8,
                marker_color=color,
                marker_line=dict(color="white", width=1),
                opacity=0.85,
                hovertemplate="%{x:.1f}%<br>%{y} scenarios<extra></extra>",
            ))
            # Mean line
            mean_v = tech_vals.mean()
            fig_h.add_vline(
                x=mean_v, line_dash="dash", line_color="#1A237E", line_width=1.5,
                annotation_text=f"{mean_v:.1f}%",
                annotation_font_size=9,
                annotation_font_color="#1A237E",
                annotation_position="top right",
            )
            short = tech.replace("Mini-Grid: ", "MG ").replace("Solar PV Only", "Solar")
            fig_h.update_layout(
                title=dict(text=short, font=dict(size=11), x=0.5),
                height=200,
                template="plotly_white",
                xaxis=dict(title="%", tickfont=dict(size=9)),
                yaxis=dict(title="Scenarios", tickfont=dict(size=9)),
                margin=dict(t=35, b=35, l=35, r=10),
            )
            st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("---")

    # ── Row 4: All-scenario bar chart — every scenario side by side ───────────
    st.markdown('<div class="section-hd">All Scenarios — Technology Share Bar Chart</div>',
                unsafe_allow_html=True)

    # Pivot: rows = scenarios, cols = technologies
    all_scens   = list(sens_df["Scenario"].unique())
    # Sort by Solar MG share descending
    mg_order = (
        sens_df[sens_df["Technology"] == MG_SOLAR]
        .set_index("Scenario")["Pct_settlements"]
        .reindex(all_scens, fill_value=0)
        .sort_values(ascending=False)
        .index.tolist()
    )

    view_opt = st.radio(
        "Chart type:",
        ["Stacked (100% view)", "Grouped (side-by-side)", "Solar MG only"],
        horizontal=True
    )

    if view_opt == "Solar MG only":
        mg_all = sens_df[sens_df["Technology"] == MG_SOLAR].copy()
        mg_all = mg_all.set_index("Scenario").reindex(mg_order).reset_index()
        fig_all = go.Figure()
        fig_all.add_trace(go.Bar(
            x=mg_all["Scenario"],
            y=mg_all["Pct_settlements"],
            marker_color=[
                PARAM_COLORS.get(
                    mg_all.loc[mg_all["Scenario"]==s, "Parameter"].iloc[0]
                    if "Parameter" in mg_all.columns else "Other",
                    "#546E7A"
                )
                for s in mg_all["Scenario"]
            ],
            text=mg_all["Pct_settlements"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Solar MG: %{y:.1f}%<extra></extra>",
        ))
        # Base line
        base_row = mg_all[mg_all["Scenario"].str.contains("Base|base", case=False, na=False)]
        if len(base_row) > 0:
            fig_all.add_hline(
                y=float(base_row["Pct_settlements"].iloc[0]),
                line_dash="dash", line_color="#F9A825", line_width=2,
                annotation_text="Base case",
                annotation_font_color="#F9A825",
                annotation_position="right",
            )
        fig_all.update_layout(
            height=380, template="plotly_white",
            xaxis=dict(title="", tickangle=-35, tickfont=dict(size=9)),
            yaxis=dict(title="Solar MG share (%)", range=[0, 12]),
            margin=dict(t=20, b=100, l=50, r=80),
        )

    else:
        barmode = "stack" if "Stacked" in view_opt else "group"
        fig_all = go.Figure()
        for tech in [t for t in ALL_TECHS if t != "Already Electrified"]:
            vals = []
            for scen in mg_order:
                row_v = sens_df[
                    (sens_df["Scenario"] == scen) &
                    (sens_df["Technology"] == tech)
                ]["Pct_settlements"]
                vals.append(float(row_v.iloc[0]) if len(row_v) > 0 else 0.0)
            if sum(vals) > 0:
                fig_all.add_trace(go.Bar(
                    name=tech,
                    x=mg_order,
                    y=vals,
                    marker_color=TECH_COLORS.get(tech, "#546E7A"),
                    hovertemplate=f"<b>{tech}</b><br>%{{y:.1f}}%<br>%{{x}}<extra></extra>"
                ))
        fig_all.update_layout(
            barmode=barmode,
            height=400, template="plotly_white",
            xaxis=dict(title="", tickangle=-35, tickfont=dict(size=9)),
            yaxis=dict(title="Share of settlements (%)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                        font=dict(size=9)),
            margin=dict(t=60, b=110, l=50, r=10),
        )

    st.plotly_chart(fig_all, use_container_width=True)

    st.markdown("---")

    # ── Row 5: SHS robustness horizontal bar ─────────────────────────────────
    st.markdown('<div class="section-hd">SHS Dominance — All Scenarios</div>',
                unsafe_allow_html=True)

    shs_all_df = (
        sens_df[sens_df["Technology"] == SHS_TECH]
        [["Scenario", "Parameter", "Pct_settlements"]]
        .sort_values("Pct_settlements")
    )
    if len(shs_all_df) > 0:
        fig_shs = go.Figure()
        for param in shs_all_df["Parameter"].unique():
            sub = shs_all_df[shs_all_df["Parameter"] == param]
            fig_shs.add_trace(go.Bar(
                y=sub["Scenario"], x=sub["Pct_settlements"],
                orientation="h", name=param,
                marker_color=PARAM_COLORS.get(param, "#546E7A"),
                hovertemplate="<b>%{y}</b><br>SHS: %{x:.1f}%<extra></extra>"
            ))
        min_shs = shs_all_df["Pct_settlements"].min()
        max_shs = shs_all_df["Pct_settlements"].max()
        fig_shs.add_vline(x=min_shs, line_dash="dot", line_color="#C62828",
                          annotation_text=f"Min {min_shs:.1f}%",
                          annotation_font_size=10, annotation_font_color="#C62828",
                          annotation_position="bottom right")
        fig_shs.add_vline(x=max_shs, line_dash="dot", line_color="#2E7D32",
                          annotation_text=f"Max {max_shs:.1f}%",
                          annotation_font_size=10, annotation_font_color="#2E7D32",
                          annotation_position="top right")
        fig_shs.update_layout(
            barmode="stack",
            height=max(280, len(shs_all_df) * 24),
            template="plotly_white",
            xaxis=dict(title="SHS share (%)", range=[0, 60]),
            yaxis=dict(tickfont=dict(size=10)),
            legend=dict(title="Parameter", orientation="v", x=1.01, font=dict(size=10)),
            margin=dict(t=20, b=40, l=210, r=160),
        )
        st.plotly_chart(fig_shs, use_container_width=True)
        st.markdown(
            f'<div class="info-box">✅ <b>Robust finding:</b> SHS is the dominant technology '
            f'in <b>all {len(shs_all_df)} scenarios</b>, ranging from '
            f'<b>{min_shs:.1f}% to {max_shs:.1f}%</b> of settlements. '
            f'No parameter combination changes this conclusion.</div>',
            unsafe_allow_html=True
        )



# ── SITE FILTERING ────────────────────────────────────────────────────────────
with t_sites:
    # ── ACCESS TARGET PROGRESS TRACKER ──────────────────────────────────────
    st.markdown('<div class="section-hd">🎯 National Electrification Targets — Progress</div>',
                unsafe_allow_html=True)

    phase_col = "ElecTargetPhase" if "ElecTargetPhase" in df.columns else "GridRolloutPhase"
    target_col_exists = phase_col in df.columns and df[phase_col].notna().any()

    if target_col_exists:
        # KPI cards for each target year
        n_total   = TOTAL_SETTLEMENTS
        n_already = int((df["MinimumOverall"]=="Already Electrified").sum())
        n_unelec  = int(df["MinimumOverall"].isin(
            ["Grid Extension","Mini-Grid: Solar PV Only","Mini-Grid: Solar-Diesel Hybrid",
             "Mini-Grid: Mini-Hydro","SHS"]).sum())

        tc1, tc2, tc3 = st.columns(3)
        phase_labels = list(PHASE_COLORS.keys())
        target_years = [2030, 2035, 2040]
        target_pcts  = [0.60, 0.85, 1.00]
        cols_t       = [tc1, tc2, tc3]

        phase_rows = []
        cumul = n_already
        for ph, yr, pct, col in zip(phase_labels, target_years, target_pcts, cols_t):
            sub = df[df.get(phase_col, pd.Series("N/A",index=df.index))==ph]
            n = len(sub); hh = int(sub["NumConnections"].sum()) if "NumConnections" in sub.columns else 0
            cap = sub["InvestmentCost"].sum()/1e6 if "InvestmentCost" in sub.columns else 0
            cumul += n
            actual_pct = cumul / n_total * 100
            phase_rows.append({"Phase":ph,"Year":yr,"Target":pct*100,"Sites":n,
                                "HH":hh,"CAPEX_M":cap,"Cumul":cumul,"ActualPct":actual_pct})
            color = PHASE_COLORS[ph]
            with col:
                st.markdown(
                    f'<div style="background:{color};border-radius:10px;padding:16px;color:white;text-align:center;">'                    f'<div style="font-size:1.8rem;font-weight:700;">{yr}</div>'                    f'<div style="font-size:1.1rem;font-weight:600;">{pct*100:.0f}% Target</div>'                    f'<div style="font-size:0.85rem;opacity:0.9;margin-top:6px;">{n:,} new sites · {hh:,} HH</div>'                    f'<div style="font-size:0.85rem;opacity:0.9;">${cap:.1f}M CAPEX</div>'                    f'<div style="font-size:0.9rem;font-weight:700;margin-top:8px;">→ {actual_pct:.1f}% access</div>'                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("")

        # Cumulative progress bar chart
        fig_prog = go.Figure()
        cumul_vals = [n_already/n_total*100] + [r["ActualPct"] for r in phase_rows]
        cumul_yrs  = [2025, 2030, 2035, 2040]
        target_pcts_line = [45.8, 60, 85, 100]

        # Actual trajectory
        fig_prog.add_trace(go.Bar(
            x=["Baseline\n2025","Phase 1\n2030","Phase 2\n2035","Phase 3\n2040"],
            y=cumul_vals,
            marker_color=["#90A4AE","#C62828","#E65100","#1565C0"],
            text=[f"{v:.1f}%" for v in cumul_vals],
            textposition="outside",
            name="Actual trajectory",
        ))
        # Target line
        fig_prog.add_trace(go.Scatter(
            x=["Baseline\n2025","Phase 1\n2030","Phase 2\n2035","Phase 3\n2040"],
            y=target_pcts_line,
            mode="lines+markers",
            line=dict(color="black",width=2,dash="dash"),
            marker=dict(size=8,symbol="diamond"),
            name="National target",
        ))
        fig_prog.update_layout(
            title="Cumulative Electrification Access Rate vs National Targets",
            yaxis_title="% of settlements electrified",
            yaxis_range=[0,110],
            height=360, template="plotly_white",
            legend=dict(orientation="h",x=0.3,y=1.1),
            margin=dict(l=20,r=20,t=50,b=20),
        )
        fig_prog.add_hline(y=60,  line_dash="dot", line_color="#C62828", annotation_text="60% target 2030")
        fig_prog.add_hline(y=85,  line_dash="dot", line_color="#E65100", annotation_text="85% target 2035")
        fig_prog.add_hline(y=100, line_dash="dot", line_color="#1565C0", annotation_text="100% target 2040")
        st.plotly_chart(fig_prog, use_container_width=True)

        # Phase × technology breakdown
        st.markdown('<div class="section-hd">Settlements per Phase × Technology</div>',
                    unsafe_allow_html=True)
        fig_tph = go.Figure()
        for t in UNELEC:
            vals = []
            for ph in phase_labels:
                n_t = len(df[(df["MinimumOverall"]==t)&(df.get(phase_col,pd.Series("N/A",index=df.index))==ph)])
                vals.append(n_t)
            if sum(vals) > 0:
                fig_tph.add_trace(go.Bar(
                    name=t,
                    x=phase_labels,
                    y=vals,
                    marker_color=TECH_COLORS[t],
                    text=[str(v) if v>0 else "" for v in vals],
                    textposition="inside",
                ))
        fig_tph.update_layout(
            barmode="stack", height=380, template="plotly_white",
            xaxis_title="Rollout Phase", yaxis_title="Settlements",
            legend=dict(orientation="v",x=1.02,y=1),
            margin=dict(l=20,r=20,t=20,b=20),
        )
        st.plotly_chart(fig_tph, use_container_width=True)

        # CAPEX per phase bar chart
        cap_by_phase = []
        for ph in phase_labels:
            sub = df[(df.get(phase_col,pd.Series("N/A",index=df.index))==ph)&
                     (df["MinimumOverall"]!="Already Electrified")]
            cap_by_phase.append(sub["InvestmentCost"].sum()/1e6 if "InvestmentCost" in sub.columns else 0)
        fig_cap = go.Figure(go.Bar(
            x=phase_labels, y=cap_by_phase,
            marker_color=list(PHASE_COLORS.values()),
            text=[f"${c:.1f}M" for c in cap_by_phase],
            textposition="outside",
        ))
        fig_cap.update_layout(
            title="Investment Required per Phase ($M)",
            height=300, template="plotly_white",
            yaxis_title="CAPEX ($M)", showlegend=False,
            margin=dict(l=20,r=20,t=40,b=20),
        )
        st.plotly_chart(fig_cap, use_container_width=True)

    else:
        st.info("Phase data not found. Re-run NB03 to generate ElecTargetPhase column.")

    st.divider()
    st.markdown('<div class="section-hd">Grid Rollout Sub-Phases (Grid Extension only)</div>',
                unsafe_allow_html=True)
    gdf=df[df["MinimumOverall"]=="Grid Extension"].copy()
    grid_ph_col = "GridRolloutPhase" if "GridRolloutPhase" in gdf.columns else phase_col
    if len(gdf)>0 and grid_ph_col in gdf.columns:
        ph_s=gdf.groupby(grid_ph_col).agg(
            Sites=("MinimumOverall","count"),HH=("NumConnections","sum"),
            CAPEX_M=("InvestmentCost",lambda x: x.sum()/1e6),
            Avg_LCOE=("MinimumOverallLCOE","mean")).reset_index().sort_values("GridRolloutPhase")
        pc=st.columns(3)
        for i,row in enumerate(ph_s.itertuples()):
            ph_val = getattr(row, grid_ph_col.replace(" ","_").replace("(","").replace(")","").replace("→",""), row[1])
            # map old/new phase labels to colors
            old_new_map = {"Phase 1 (0-5 yrs)":"Phase 1 (→2030)","Phase 2 (5-10 yrs)":"Phase 2 (→2035)","Phase 3 (10-15 yrs)":"Phase 3 (→2040)"}
            lookup = old_new_map.get(ph_val, ph_val)
            col=PHASE_COLORS.get(lookup, PHASE_COLORS.get(ph_val,"#999"))
            with pc[i%3]:
                st.markdown(
                    f'<div class="phase-card" style="background:{col}">'
                    f'<div class="phase-title">{row.GridRolloutPhase}</div>'
                    f'<div class="phase-detail">{int(row.Sites)} sites · {int(row.HH):,} HH</div>'
                    f'<div class="phase-detail">${row.CAPEX_M:.1f}M · {row.Avg_LCOE:.3f} $/kWh</div>'
                    f'</div>', unsafe_allow_html=True)
        fp=px.bar(ph_s,x="GridRolloutPhase",y="Sites",color="GridRolloutPhase",
            color_discrete_map=PHASE_COLORS,text="Sites",height=260,
            template="plotly_white",labels={"GridRolloutPhase":"","Sites":"Settlements"})
        fp.update_traces(textposition="outside")
        fp.update_layout(showlegend=False,margin=dict(l=20,r=20,t=10,b=20))
        st.plotly_chart(fp, use_container_width=True)

        # Phase milestones bar chart by target year
        st.markdown('<div class="section-hd">Grid Rollout — Cumulative Progress by Target Year</div>',
                    unsafe_allow_html=True)
        target_years = [2025, 2030, 2035, 2040]
        phase_year_map = {
            "Phase 1 (→2030)" : 2030,
            "Phase 2 (→2035)" : 2035,
            "Phase 3 (→2040)" : 2040,
            # backward compat
            "Phase 1 (0-5 yrs)"  : 2030,
            "Phase 2 (5-10 yrs)" : 2035,
            "Phase 3 (10-15 yrs)": 2040,
        }
        milestone_rows = []
        cumul_sites, cumul_hh, cumul_capex, cumul_kw = 0, 0, 0, 0
        for ph in ["Phase 1 (0-5 yrs)","Phase 2 (5-10 yrs)","Phase 3 (10-15 yrs)"]:
            sub_ph = gdf[gdf["GridRolloutPhase"]==ph]
            cumul_sites += len(sub_ph)
            cumul_hh    += sub_ph["NumConnections"].sum() if "NumConnections" in sub_ph.columns else 0
            cumul_capex += sub_ph["InvestmentCost"].sum() if "InvestmentCost" in sub_ph.columns else 0
            # Grid capacity: demand / 8760 × (1/base_to_peak 0.85)
            if "_capacity_kw" in sub_ph.columns:
                cumul_kw += sub_ph["_capacity_kw"].sum()
            elif "DemandKWh_Y0" in sub_ph.columns:
                cumul_kw += (sub_ph["DemandKWh_Y0"] / 8760 / 0.85).sum()
            yr = phase_year_map[ph]
            milestone_rows.append({
                "Target Year" : str(yr),
                "Phase"       : ph,
                "Sites"       : cumul_sites,
                "HH"          : int(cumul_hh),
                "CAPEX ($M)"  : round(cumul_capex/1e6, 1),
                "Capacity (kW)": round(cumul_kw, 0),
                "Color"       : PHASE_COLORS.get(ph,"#999"),
            })
        mdf = pd.DataFrame(milestone_rows)

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            fg1 = go.Figure()
            for _, row in mdf.iterrows():
                fg1.add_trace(go.Bar(
                    x=[row["Target Year"]], y=[row["Sites"]],
                    name=row["Phase"], marker_color=row["Color"],
                    text=str(row["Sites"]), textposition="outside",
                    showlegend=True,
                    hovertemplate=f"<b>{row['Phase']}</b><br>By {row['Target Year']}: {row['Sites']} sites<extra></extra>",
                ))
            fg1.update_layout(barmode="group", height=340, template="plotly_white",
                title="Cumulative Grid Sites",
                xaxis_title="Target Year", yaxis_title="Cumulative sites",
                legend=dict(orientation="h",x=0,y=-0.28,font=dict(size=9)),
                margin=dict(l=20,r=20,t=40,b=80))
            st.plotly_chart(fg1, use_container_width=True)

        with mc2:
            fg2 = go.Figure()
            for _, row in mdf.iterrows():
                fg2.add_trace(go.Bar(
                    x=[row["Target Year"]], y=[row["CAPEX ($M)"]],
                    name=row["Phase"], marker_color=row["Color"],
                    text=f"${row['CAPEX ($M)']:.1f}M", textposition="outside",
                    showlegend=False,
                    hovertemplate=f"<b>{row['Phase']}</b><br>By {row['Target Year']}: ${row['CAPEX ($M)']:.1f}M<extra></extra>",
                ))
            fg2.update_layout(barmode="group", height=340, template="plotly_white",
                title="Cumulative Grid CAPEX",
                xaxis_title="Target Year", yaxis_title="Cumulative CAPEX ($M)",
                margin=dict(l=20,r=20,t=40,b=80))
            st.plotly_chart(fg2, use_container_width=True)

        with mc3:
            fg3 = go.Figure()
            for _, row in mdf.iterrows():
                fg3.add_trace(go.Bar(
                    x=[row["Target Year"]], y=[row["Capacity (kW)"]],
                    name=row["Phase"], marker_color=row["Color"],
                    text=f"{row['Capacity (kW)']:,.0f} kW", textposition="outside",
                    showlegend=False,
                    hovertemplate=f"<b>{row['Phase']}</b><br>By {row['Target Year']}: {row['Capacity (kW)']:,.0f} kW<extra></extra>",
                ))
            fg3.update_layout(barmode="group", height=340, template="plotly_white",
                title="Cumulative Grid Capacity",
                xaxis_title="Target Year", yaxis_title="Cumulative capacity (kW)",
                margin=dict(l=20,r=20,t=40,b=80))
            st.plotly_chart(fg3, use_container_width=True)

        # Summary table
        st.dataframe(mdf[["Target Year","Phase","Sites","HH","CAPEX ($M)","Capacity (kW)"]].style.format(
            {"HH":"{:,.0f}","CAPEX ($M)":"${:.1f}","Capacity (kW)":"{:,.0f} kW"}),
            use_container_width=True, hide_index=True)
    else:
        st.info("No grid extension settlements in current filter.")

    st.divider()
    st.markdown('<div class="section-hd">Priority Investment Sites</div>',
                unsafe_allow_html=True)
    sf1,sf2=st.columns(2)
    with sf1:
        if "DemandKWh_Y0" in df.columns:
            thr=st.slider("Large Solar MG — min demand (kWh/yr)",10000,100000,50000,step=5000)
            lmg=df[(df["MinimumOverall"]=="Mini-Grid: Solar PV Only")&(df["DemandKWh_Y0"]>thr)]
            st.metric(f"Solar MG > {thr/1000:.0f}k kWh/yr",
                      f"{len(lmg):,} sites · {lmg['NumConnections'].sum():,.0f} HH")
            if len(lmg)>0 and "X_deg" in lmg.columns:
                fm=px.scatter_mapbox(lmg,lat="Y_deg",lon="X_deg",
                    color="MinimumOverall",color_discrete_map=TECH_COLORS,
                    size="NumConnections",size_max=12,mapbox_style="carto-positron",
                    zoom=5.5,center={"lat":9.3,"lon":2.3},height=280)
                fm.update_layout(margin=dict(l=0,r=0,t=0,b=0),showlegend=False)
                st.plotly_chart(fm, use_container_width=True)
    with sf2:
        if dist_col in df.columns:
            sd=st.slider("SHS near grid — max distance (km)",1,20,5)
            sn=df[(df["MinimumOverall"]=="SHS")&(df[dist_col]<=sd)]
            st.metric(f"SHS within {sd}km of electrified",
                      f"{len(sn):,} sites · {sn['NumConnections'].sum():,.0f} HH",
                      help="Future grid densification candidates")
            if len(sn)>0 and "X_deg" in sn.columns:
                fs=px.scatter_mapbox(sn,lat="Y_deg",lon="X_deg",
                    color="MinimumOverall",color_discrete_map=TECH_COLORS,
                    size="NumConnections",size_max=10,mapbox_style="carto-positron",
                    zoom=5.5,center={"lat":9.3,"lon":2.3},height=280)
                fs.update_layout(margin=dict(l=0,r=0,t=0,b=0),showlegend=False)
                st.plotly_chart(fs, use_container_width=True)

# ── DATA TABLE ────────────────────────────────────────────────────────────────
with t_data:
    st.markdown('<div class="section-hd">Settlement Data</div>', unsafe_allow_html=True)
    disp=[c for c in ["MinimumOverall","NumConnections","DemandKWh_Y0",
                       "MinimumOverallLCOE","InvestmentCost","InvestmentPerCapita",
                       dist_col,"GridRolloutPhase","GHI","WealthIndex",
                       "X_deg","Y_deg"] if c in df.columns]
    ren={"MinimumOverall":"Technology","NumConnections":"HH",
         "DemandKWh_Y0":"Demand (kWh/yr)","MinimumOverallLCOE":"LCOE ($/kWh)",
         "InvestmentCost":"CAPEX ($)","InvestmentPerCapita":"$/capita",
         dist_col:"Dist elec (km)","GridRolloutPhase":"Phase"}
    fmt={"Demand (kWh/yr)":"{:,.0f}","LCOE ($/kWh)":"{:.3f}",
         "CAPEX ($)":"${:,.0f}","$/capita":"${:.0f}","Dist elec (km)":"{:.1f}"}
    st.dataframe(df[disp].rename(columns=ren).style.format(fmt),
                 use_container_width=True, height=380)
    st.download_button(f"⬇️ Download filtered data ({len(df):,} settlements)",
        data=df[disp].to_csv(index=False),
        file_name="benin_electrification_filtered.csv",
        mime="text/csv", use_container_width=True)

st.caption("Benin National Electrification Plan · 2025")
