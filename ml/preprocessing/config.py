"""Shared configuration and constants for data generation and preprocessing."""

from datetime import date
from pathlib import Path

SEED = 42
TIMELINE_END = date(2026, 10, 19)  # last day of generated data (inclusive)
SENSOR_FREQ = "5min"
CURRENCY = "GBP"

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
UNIFIED = RAW / "unified"
REFERENCE = Path(__file__).resolve().parent / "reference"

# Ensure directories exist
UNIFIED.mkdir(parents=True, exist_ok=True)
REFERENCE.mkdir(parents=True, exist_ok=True)

# Operating thresholds (from the synthetic machine manual, used for scenario design only)
VIB_WARNING = 4.5  # mm/s
VIB_CRITICAL = 7.0  # mm/s
TEMP_MAX_SPEC = 85.0  # degrees C
INSPECTION_INTERVAL_H = 500
PREVENTIVE_INTERVAL_H = 2000
