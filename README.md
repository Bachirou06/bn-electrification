# 🇧🇯 Benin Least-Cost Electrification Analysis

A Python electrification planning model comparing Grid, Mini-Grid, and Solar Home Systems across **17,205 settlements** in Benin over a 15-year horizon (2025–2040), built on VIDA satellite settlement data and OnSSET methodology.

---
SHS is the dominant technology in **all 17 sensitivity scenarios** (42–49%). The mini-grid demand threshold is the most influential assumption (±5 pp swing on Solar MG share).

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/benin-electrification.git
cd benin-electrification
```

### 2. Create and activate a virtual environment

**Option A — using `venv` (recommended):**

```bash
# Create the environment
python -m venv bn-electrification

# Activate — Windows
bn-electrification\Scripts\activate

# Activate — macOS / Linux
source bn-electrification/bin/activate
```

**Option B — using `conda`:**

```bash
conda create -n bn-electrification python=3.11
conda activate bn-electrification
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, install manually:
> ```bash
> pip install geopandas pandas numpy scipy rasterio pyyaml
> pip install streamlit plotly jupyter notebook seaborn
> ```

### 4. Verify the install

```bash
python -c "import geopandas, streamlit, plotly; print('All good ✓')"
```

---

## Running the Analysis

Run the notebooks in order — each one depends on the output of the previous:

```bash
jupyter notebook
```

| Notebook | What it does |
|---|---|
| `00_gis_layer_extraction.ipynb` | Attaches GHI, slope, hydro, nightlights to settlements |
| `01_data_exploration.ipynb` | EDA and data quality checks |
| `02_demand_model.ipynb` | MTF-tier demand estimation per settlement |
| `03_lcoe_model.ipynb` | LCOE computation + technology selection |
| `04_sensitivity_analysis.ipynb` | 17-scenario parametric sensitivity |

NB03 produces the CSV needed by the dashboard. Run its save cell (last cell) before launching Streamlit.

---

## Running the Dashboard

```bash
python -m streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. The app auto-loads the most recent CSV from `data/outputs/tables/`.

---

## Structure

```
├── config/assumptions.yaml       # All model parameters
├── data/raw/                     # VIDA settlements + rasters (not tracked)
├── notebooks/
│   ├── 00_gis_layer_extraction.ipynb   # GIS enrichment (GHI, slope, hydro, NTL)
│   ├── 01_data_exploration.ipynb       # EDA and settlement characterisation
│   ├── 02_demand_model.ipynb           # MTF-tier demand estimation
│   ├── 03_lcoe_model.ipynb             # LCOE + technology selection
│   └── 04_sensitivity_analysis.ipynb   # 17-scenario parametric sensitivity
├── src/
│   ├── costs/                    # grid_extension, mini_grid, shs, lcoe_calculator
│   ├── demand/                   # demand_estimator, mtf_tiers
│   ├── least_cost/               # technology_selector (timestep + grid creep)
│   └── utils/                    # spatial, plotting
└── streamlit_app.py              # Interactive dashboard (7 tabs)
```

---

## Key Methodological Choices

- **Demand:** VIDA satellite building data (47M buildings across 58 countries) — not population proxies
- **Grid routing:** Option B (nearest electrified settlement via cKDTree) — not raw MV backbone distance
- **Technology selection:** Three-timestep dynamic model — grid creep reduces distances across 2030/2035/2040 phases
- **Battery:** LFP chemistry, 10-year replacement (not lead-acid)
- **Hydro fix:** HydroSHEDS head ≥ 5 m threshold applied (fill values removed)

---

## Data Sources

| Layer | Source |
|---|---|
| Settlement polygons | VIDA / DRE Atlas (ESMAP/World Bank 2024) |
| Grid infrastructure | World Bank energydata.info (SBEE lines) |
| Solar irradiance | NASA POWER / Global Solar Atlas |
| Terrain / elevation | SRTM 30m (OpenTopography) |
| Hydro potential | HydroSHEDS / custom GPKG |
| Night-time lights | VIIRS DNB 2020 (NASA/NOAA) |
| Land cover | MODIS MCD12Q1 2022 (NASA LP DAAC) |
| Travel time | Oxford MAP accessibility raster (Weiss et al. 2020) |

---

## Requirements

```
python >= 3.10
geopandas, pandas, numpy, scipy
streamlit, plotly
pyyaml, rasterio
```

---

## License

MIT — see `LICENSE`
