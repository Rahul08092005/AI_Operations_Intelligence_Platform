"""Configuration and constants for Day 5 Anomaly Detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Paths
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"

# Reproducibility
SEED = 42

# Status Thresholds (Decision D57)
STATUS_WARNING_THRESHOLD = 0.50
STATUS_CRITICAL_THRESHOLD = 0.80

STATUS_NORMAL = "Normal"
STATUS_WARNING = "Warning"
STATUS_CRITICAL = "Critical"
STATUS_OFFLINE = "Offline"
STATUS_NODATA = "NoData"

STATUS_LEVELS = [STATUS_NORMAL, STATUS_WARNING, STATUS_CRITICAL, STATUS_OFFLINE, STATUS_NODATA]

STATE_RUNNING = "Running"
STATE_OFFLINE = "Offline"
STATE_NODATA = "NoData"

# Calibration anchors (Decision D57)
CALIBRATION_ANCHORS = {
    "median_score": 0.10,
    "warn_score": 0.50,
    "crit_score": 0.80,
    "saturation_score": 1.00,
}

# Feature Sets (Decision D59 / Section 6.4)
FEATURE_SET_A = [
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "energy_consumption",
]

FEATURE_SET_B = FEATURE_SET_A + [
    "motor_current",
    "load_ratio_1h",
]

FEATURE_SET_C = FEATURE_SET_B + [
    "temperature_roll_mean_1h",
    "temperature_roll_std_1h",
    "vibration_roll_mean_1h",
    "vibration_roll_std_1h",
    "pressure_roll_mean_1h",
    "pressure_roll_std_1h",
    "energy_consumption_roll_mean_1h",
    "temperature_change",
    "vibration_change",
    "pressure_change",
    "energy_change",
    "vibration_roll_mean_6h",
    "temperature_roll_mean_6h",
    "vibration_slope_24h",
    "temperature_slope_24h",
]

FEATURE_SET_D = [
    "pressure",
    "motor_current",
    "load_ratio_1h",
    "energy_consumption",
    "pressure_roll_mean_1h",
    "pressure_roll_std_1h",
    "pressure_change",
    "energy_change",
    "motor_current_roll_mean_1h",
    "motor_current_roll_std_1h",
    "motor_current_roll_std_6h",
]

FEATURE_SETS: dict[str, list[str]] = {
    "plan_core": FEATURE_SET_A,
    "core_context": FEATURE_SET_B,
    "core_temporal": FEATURE_SET_C,
    "real_only": FEATURE_SET_D,
}

# CV Search Grids (Section 6.5, Decision D63)
IFOREST_PARAM_GRID: dict[str, dict[str, list[Any]]] = {
    "plan_core": {
        "n_estimators": [200, 400],
        "max_samples": [256, 1024, 4096],
        "max_features": [1.0],
    },
    "core_context": {
        "n_estimators": [200, 400],
        "max_samples": [256, 1024, 4096],
        "max_features": [1.0],
    },
    "core_temporal": {
        "n_estimators": [200, 400],
        "max_samples": [256, 1024, 4096],
        "max_features": [1.0, 0.6],
    },
    "real_only": {
        "n_estimators": [200, 400],
        "max_samples": [256, 1024, 4096],
        "max_features": [1.0],
    },
}

# Post-processing Grid
POST_PROCESSING_GRID = {
    "q_warn": [0.99, 0.995, 0.999],
    "q_crit": [0.9995, 0.9999],
    "smoothing_window": [6, 12, 36],  # 30 min, 1h, 3h in 5-min intervals
}

# Episodes & Buffers (Decision D56)
EXCLUDED_EPISODE_TYPES = [
    "real_failure",
    "real_precursor",
    "hero_degradation",
    "mild_drift",
    "synthetic_failure_precursor",
    "synthetic_failure_outage",
    "pre_maintenance_drift",
    "planned_maintenance_outage",
]

TRUE_ANOMALY_EPISODE_TYPES = [
    "real_failure",
    "real_precursor",
    "hero_degradation",
    "mild_drift",
    "synthetic_failure_precursor",
]

NEGATIVE_EPISODE_TYPES = [
    "normal",
    "benign_spike",
]

IGNORED_EPISODE_TYPES = [
    "synthetic_failure_outage",
    "planned_maintenance_outage",
]

BUFFER_HOURS = 24

# Alert Episode extraction (Decision D58)
ALERT_MIN_INTERVALS = 3
ALERT_MERGE_GAP_HOURS = 2.0

# False Alarm Budget (Section 6.6)
MAX_WARNING_PER_30D = 1.0
MAX_CRITICAL_PER_30D = 0.0

# B1 Manual Thresholds
B1_VIB_WARN = 4.5
B1_VIB_CRIT = 7.0
B1_TEMP_WARN = 85.0  # Or temp spec
