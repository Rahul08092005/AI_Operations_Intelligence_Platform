"""Configuration constants, feature definitions, and hyperparameter search spaces.

Implements architectural decisions D67 to D82 for Day 6 failure prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Seed for absolute reproducibility (Rule R1)
SEED = 42

# Directory paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"
NOTEBOOKS_DIR = ROOT / "ml" / "notebooks"

# File paths
RAW_AI4I_PATH = RAW_DIR / "ai4i" / "ai4i2020.csv"
SPLIT_INDEX_PATH = PROCESSED_DIR / "ai4i_split_index.csv"
FROZEN_CONFIG_PATH = MODELS_DIR / "failure_config_frozen.json"
MODEL_ARTIFACT_PATH = MODELS_DIR / "failure_model.pkl"
METADATA_ARTIFACT_PATH = MODELS_DIR / "failure_model_metadata.json"
MODEL_PKL_PATH = MODEL_ARTIFACT_PATH
MODEL_METADATA_PATH = METADATA_ARTIFACT_PATH
HOLDOUT_LOG_PATH = DOCS_DIR / "holdout_log.md"

EXPERIMENTS_LOG_PATH = PROCESSED_DIR / "failure_experiments.csv"
OOF_PREDICTIONS_PATH = PROCESSED_DIR / "failure_oof_predictions.parquet"
TEST_PREDICTIONS_PATH = PROCESSED_DIR / "failure_test_predictions.parquet"

DEMO_STATES_PATH = PROCESSED_DIR / "machine_ai4i_state.parquet"
MACHINE_RISK_DAILY_PATH = PROCESSED_DIR / "machine_risk_daily.parquet"
MACHINE_RISK_LATEST_PATH = PROCESSED_DIR / "machine_risk_latest.csv"

# Model version string
MODEL_VERSION = "failure-xgb-1.0.0"

# Problem framing statement (Decision D67, Rule R10)
FRAMING_STATEMENT = "Failure risk for the current operating state"
DEFAULT_FRAMING = FRAMING_STATEMENT

# Risk Categories Vocabulary (Decision D75)
RISK_LOW = "Low"
RISK_MEDIUM = "Medium"
RISK_HIGH = "High"
RISK_CRITICAL = "Critical"
RISK_LEVELS: list[str] = [RISK_LOW, RISK_MEDIUM, RISK_HIGH, RISK_CRITICAL]

# Type mapping: L=0, M=1, H=2 (Decision D69)
TYPE_MAP: dict[str, int] = {"L": 0, "M": 1, "H": 2}
TYPE_MAP_REV: dict[int, str] = {0: "L", 1: "M", 2: "H"}

# Raw column definitions
RAW_FEATURE_COLS = [
    "type",
    "air_temp_k",
    "process_temp_k",
    "rpm",
    "torque_nm",
    "tool_wear_min",
]

TARGET_COL = "machine_failure"

# Mode flags (used strictly for evaluation breakdown, never model features per R3, D70)
MODE_FLAG_COLS = ["twf", "hdf", "pwf", "osf", "rnf"]

# Identifier columns (strictly dropped per R3, D70)
IDENTIFIER_COLS = ["udi", "product_id"]

# Forbidden leakage columns (Rule R3, Decision D70)
FORBIDDEN_COLUMNS: list[str] = [
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
    "udi",
    "product_id",
    "machine_failure",
]

# Forbidden columns that must never enter the feature matrix
FORBIDDEN_COLS = set(MODE_FLAG_COLS + IDENTIFIER_COLS + [TARGET_COL])

# Feature Sets (Decision D69, Section 6.3)
FEATURE_SET_RAW = [
    "type_code",
    "air_temp_k",
    "process_temp_k",
    "rpm",
    "torque_nm",
    "tool_wear_min",
]

FEATURE_SET_PHYSICS = [
    "type_code",
    "air_temp_k",
    "process_temp_k",
    "rpm",
    "torque_nm",
    "tool_wear_min",
    "temp_diff_k",
    "power_w",
    "strain_nmin",
]

FEATURE_SETS: dict[str, list[str]] = {
    "raw": FEATURE_SET_RAW,
    "physics": FEATURE_SET_PHYSICS,
}

# Cross-Validation parameters (Decision D71)
CV_N_SPLITS = 5
CV_N_REPEATS = 2

# Pre-registered search space for XGBoost (Decision D72, D73, Section 6.5)
XGB_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 300, 450, 600],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [1.0, 5.0, 10.0],
    "gamma": [0.0, 0.5],
    "scale_pos_weight_mode": ["one", "sqrt_ratio", "full_ratio"],
}

# Maximum allowed configurations in experiment search (Rule R6, Decision D72)
MAX_SEARCH_DRAWS = 50
MAX_CONFIGS = 60

# Fixed XGBoost parameters
XGB_FIXED_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "n_jobs": -1,
    "random_state": SEED,
}

# Threshold targets on OOF probabilities (Decision D75, Section 6.6)
TARGET_RECALL_MED = 0.95
TARGET_PRECISION_CRIT = 0.90

# Risk categories order (Decision D75)
RISK_CATEGORIES: tuple[str, ...] = ("Low", "Medium", "High", "Critical")
RISK_CATEGORY_ORDER: list[str] = ["Low", "Medium", "High", "Critical"]

# Background sample size for SHAP (Decision D79)
SHAP_BACKGROUND_SIZE = 200

