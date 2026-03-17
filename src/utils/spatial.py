"""
spatial.py
----------
Spatial utility functions for the Benin electrification analysis.
"""

import geopandas as gpd
import pandas as pd
import numpy as np


def load_settlements(path: str) -> gpd.GeoDataFrame:
    """Load and validate the settlements GeoJSON."""
    gdf = gpd.read_file(path)
    print(f"Loaded {len(gdf):,} settlements")
    print(f"CRS: {gdf.crs}")
    print(f"Columns: {list(gdf.columns)}")

    # Basic cleaning
    gdf = gdf[gdf["population"] > 0].copy()
    gdf["population"] = pd.to_numeric(gdf["population"], errors="coerce").fillna(0)

    # Ensure distance columns are numeric
    dist_cols = [c for c in gdf.columns if "dist" in c.lower() or "distance" in c.lower()]
    for col in dist_cols:
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    print(f"After filtering zero-population: {len(gdf):,} settlements")
    return gdf


def load_transmission_lines(path: str) -> gpd.GeoDataFrame:
    """Load and validate the transmission lines GeoJSON."""
    gdf = gpd.read_file(path)
    print(f"Loaded {len(gdf):,} transmission line segments")

    if "Situation" in gdf.columns:
        print(f"Situation values: {gdf['Situation'].unique()}")
    if "Voltage_KV" in gdf.columns:
        print(f"Voltage levels (kV): {sorted(gdf['Voltage_KV'].dropna().unique())}")

    return gdf


def split_existing_planned(lines_gdf: gpd.GeoDataFrame) -> tuple:
    """
    Split transmission lines into existing and planned.
    Returns (existing_gdf, planned_gdf).
    """
    situation = lines_gdf["Situation"].str.lower()
    existing  = lines_gdf[situation.str.contains("exist|operat|active", na=False)]
    planned   = lines_gdf[situation.str.contains("plan|construct|propos", na=False)]
    return existing, planned
