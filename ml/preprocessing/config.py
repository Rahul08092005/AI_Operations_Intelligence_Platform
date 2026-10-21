"""Shared configuration and constants for data generation and preprocessing."""

from datetime import date
from pathlib import Path

SEED = 42
TIMELINE_START = date(2026, 3, 21)
TIMELINE_END = date(2026, 10, 19)  # last day of generated data (inclusive)
SENSOR_FREQ = "5min"
CURRENCY = "GBP"
I_OFF = 1.55  # Motor current cutoff (Amps) for running state
ENERGY_UNIT_RATE_GBP = 0.22  # Industrial UK electric tariff

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
UNIFIED = RAW / "unified"
PROCESSED = ROOT / "data" / "processed"
REFERENCE = Path(__file__).resolve().parent / "reference"

# Ensure directories exist
UNIFIED.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
REFERENCE.mkdir(parents=True, exist_ok=True)

# Operating thresholds (from the synthetic machine manual, used for scenario design only)
VIB_WARNING = 4.5  # mm/s
VIB_CRITICAL = 7.0  # mm/s
TEMP_MAX_SPEC = 85.0  # degrees C
INSPECTION_INTERVAL_H = 500
PREVENTIVE_INTERVAL_H = 2000

# Day 3: Pipeline Configuration
MAX_IMPUTE_INTERVALS = 3  # 15 minutes at 5-minute resolution
ROLLING_WINDOWS = {"1h": 12, "6h": 72, "24h": 288}  # in 5-minute intervals
ROLLING_MIN_FRACTION = 0.5
WARMUP_INTERVALS = 288  # first 24 hours per machine
OVERDUE_INSPECTION_H = INSPECTION_INTERVAL_H  # 500
OVERDUE_PREVENTIVE_H = PREVENTIVE_INTERVAL_H  # 2000
FAILURE_LOOKBACK_DAYS = 90
STAT_OUTLIER_ROBUST_Z = 6.0  # flag threshold (median / MAD based)
RETAIL_EXTREME_QTY = 10_000  # extreme quantity threshold

