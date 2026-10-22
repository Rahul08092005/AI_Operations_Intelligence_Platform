"""Configuration constants for Day 4 EDA."""


from ml.preprocessing.config import (
    PROCESSED,
    ROOT,
    SEED,
    TEMP_MAX_SPEC,
    VIB_CRITICAL,
    VIB_WARNING,
)

EXPLORATION_FRACTION = 0.7
VALIDATION_DAYS = 28  # validation slice for baseline forecasts (D50)
AI4I_TEST_FRACTION = 0.2
BOOTSTRAP_ITERATIONS = 2000
FIGURES = ROOT / "docs" / "figures"
NOTEBOOKS = ROOT / "ml" / "notebooks"
AI4I_CSV = ROOT / "data" / "raw" / "ai4i"  # directory containing ai4i2020.csv
AI4I_SPLIT_INDEX = PROCESSED / "ai4i_split_index.csv"

# Colorblind-safe categorical palette (Okabe-Ito)
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

__all__ = [
    "AI4I_CSV",
    "AI4I_SPLIT_INDEX",
    "AI4I_TEST_FRACTION",
    "BOOTSTRAP_ITERATIONS",
    "EXPLORATION_FRACTION",
    "FIGURES",
    "NOTEBOOKS",
    "PALETTE",
    "PROCESSED",
    "ROOT",
    "SEED",
    "TEMP_MAX_SPEC",
    "VALIDATION_DAYS",
    "VIB_CRITICAL",
    "VIB_WARNING",
]
