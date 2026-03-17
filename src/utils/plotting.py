"""
plotting.py
-----------
Reusable visualization functions for the Benin electrification analysis.
"""

import folium
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas as gpd

# ── Color palettes ─────────────────────────────────────────────────────────────

TECH_COLORS = {
    "Grid Extension": "#2196F3",   # Blue
    "Mini-Grid":      "#FF9800",   # Orange
    "SHS":            "#4CAF50",   # Green
    "No Option":      "#9E9E9E",   # Grey
}

TIER_COLORS = {
    1: "#FFF9C4",
    2: "#FFE082",
    3: "#FFB300",
    4: "#E65100",
}


# ── Folium interactive maps ────────────────────────────────────────────────────

def make_technology_map(
    gdf: gpd.GeoDataFrame,
    lines_gdf: gpd.GeoDataFrame = None,
    output_path: str = None,
) -> folium.Map:
    """
    Interactive folium map showing least-cost technology per settlement.
    """
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")

    # Plot settlements
    for _, row in gdf.iterrows():
        tech   = row.get("least_cost_tech", "No Option")
        color  = TECH_COLORS.get(tech, "#9E9E9E")
        pop    = row.get("population", 0)
        lcoe   = row.get("least_cost_lcoe", np.nan)

        folium.CircleMarker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            radius=max(3, min(10, pop / 500)),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{row.get('village_name', row.get('identifier', 'N/A'))}</b><br>"
                f"Population: {int(pop):,}<br>"
                f"Technology: {tech}<br>"
                f"LCOE: {lcoe:.3f} USD/kWh" if not np.isnan(lcoe) else "",
                max_width=200,
            ),
        ).add_to(m)

    # Plot transmission lines
    if lines_gdf is not None:
        for _, row in lines_gdf.iterrows():
            try:
                coords = [(y, x) for x, y in row.geometry.coords]
                folium.PolyLine(
                    locations=coords,
                    color="#B71C1C",
                    weight=2,
                    opacity=0.7,
                    tooltip=f"{row.get('Name', '')} — {row.get('Voltage_KV', '')} kV",
                ).add_to(m)
            except Exception:
                continue

    # Legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px; border-radius:5px;
                border:1px solid #ccc; font-size:13px;">
    <b>Least-Cost Technology</b><br>
    """
    for tech, color in TECH_COLORS.items():
        legend_html += f'<span style="color:{color}">&#9679;</span> {tech}<br>'
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    if output_path:
        m.save(output_path)
        print(f"Map saved to {output_path}")

    return m


def make_demand_map(gdf: gpd.GeoDataFrame, output_path: str = None) -> folium.Map:
    """Interactive map showing demand (kWh/year) and MTF tiers."""
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")

    max_demand = gdf["demand_year0_kwh"].quantile(0.95)

    for _, row in gdf.iterrows():
        tier   = int(row.get("mtf_tier", 2))
        color  = TIER_COLORS.get(tier, "#FFE082")
        demand = row.get("demand_year0_kwh", 0)
        radius = max(3, min(12, demand / max_demand * 10))

        folium.CircleMarker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>{row.get('village_name', 'N/A')}</b><br>"
                f"MTF Tier: {tier}<br>"
                f"Demand (Y0): {demand:,.0f} kWh/yr<br>"
                f"Population: {int(row.get('population', 0)):,}",
                max_width=200,
            ),
        ).add_to(m)

    if output_path:
        m.save(output_path)
    return m


# ── Static matplotlib charts ──────────────────────────────────────────────────

def plot_tech_breakdown(df: pd.DataFrame, ax=None):
    """Bar chart of settlement count by technology."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    counts = df["least_cost_tech"].value_counts()
    colors = [TECH_COLORS.get(t, "#9E9E9E") for t in counts.index]
    counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white")

    ax.set_title("Settlements by Least-Cost Technology", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Settlements")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    return ax


def plot_lcoe_distribution(df: pd.DataFrame, ax=None):
    """Box plot of LCOE distribution by technology."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    data = []
    for tech, col in zip(TECH_LABELS := ["Grid Extension", "Mini-Grid", "SHS"],
                         ["lcoe_grid", "lcoe_minigrid", "lcoe_shs"]):
        valid = df[col].replace(np.inf, np.nan).dropna()
        for v in valid:
            data.append({"Technology": tech, "LCOE (USD/kWh)": v})

    plot_df = pd.DataFrame(data)
    palette = {t: TECH_COLORS[t] for t in TECH_LABELS}
    sns.boxplot(data=plot_df, x="Technology", y="LCOE (USD/kWh)", palette=palette, ax=ax)

    ax.set_title("LCOE Distribution by Technology", fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    return ax


def plot_demand_by_tier(df: pd.DataFrame, ax=None):
    """Bar chart of average demand by MTF tier."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    tier_demand = df.groupby("mtf_tier")["demand_year0_kwh"].mean()
    colors = [TIER_COLORS.get(t, "#FFE082") for t in tier_demand.index]
    tier_demand.plot(kind="bar", ax=ax, color=colors, edgecolor="white")

    ax.set_title("Average Annual Demand by MTF Tier (Year 0)", fontsize=13, fontweight="bold")
    ax.set_xlabel("MTF Tier")
    ax.set_ylabel("Avg Demand (kWh/year)")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    return ax


def plot_priority_top_n(df: pd.DataFrame, n: int = 20, ax=None):
    """Horizontal bar chart of top-N priority settlements."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))

    top = df.nsmallest(n, "priority_rank")[["village_name", "priority_score", "least_cost_tech"]].copy()
    colors = [TECH_COLORS.get(t, "#9E9E9E") for t in top["least_cost_tech"]]

    ax.barh(top["village_name"], top["priority_score"], color=colors)
    ax.set_title(f"Top {n} Priority Settlements", fontsize=13, fontweight="bold")
    ax.set_xlabel("Priority Score")
    ax.invert_yaxis()

    patches = [mpatches.Patch(color=v, label=k) for k, v in TECH_COLORS.items() if k != "No Option"]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    return ax
