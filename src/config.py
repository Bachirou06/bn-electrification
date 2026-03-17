"""
config.py
---------
Loads and exposes the central assumptions YAML as a Python dict.
All modules should import CONFIG from here — never hardcode parameters.
"""

import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config" / "assumptions.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()


# ── Convenience accessors ──────────────────────────────────────────────────────

GENERAL      = CONFIG["general"]
DEMAND       = CONFIG["demand"]
GRID         = CONFIG["grid_extension"]
MINIGRID     = CONFIG["mini_grid"]
SHS          = CONFIG["shs"]

DISCOUNT_RATE    = GENERAL["discount_rate"]
HORIZON          = GENERAL["planning_horizon_years"]
BASE_YEAR        = GENERAL["base_year"]
AVG_HH_SIZE      = GENERAL["avg_household_size"]
