"""Inference module for real-time and windowed anomaly detection (Task T13).

Provides:
- load_detector(path=None): cached loader for the primary AnomalyDetector.
- score_window(sensor_window_df, machine_id=None, ...): scores a window of raw sensor
  data using feature engineering and trailing score smoothing, ensuring train/serve parity.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ml.anomaly_detection.config import MODELS_DIR, PROCESSED_DIR
from ml.anomaly_detection.detector import AnomalyDetector
from ml.preprocessing.features_sensor import engineer_sensor_features

logger = logging.getLogger(__name__)

# Module-level detector cache
_CACHED_DETECTOR: AnomalyDetector | None = None
_CACHED_PATH: Path | None = None

MIN_HISTORY_HOURS = 24.0
MIN_HISTORY_INTERVALS = 288  # 24h at 5-minute cadence


def load_detector(path: str | Path | None = None) -> AnomalyDetector:
    """Load the anomaly detector artifact with module-level caching."""
    global _CACHED_DETECTOR, _CACHED_PATH

    target_path = Path(path) if path is not None else MODELS_DIR / "anomaly_model.pkl"
    target_path = target_path.resolve()

    if _CACHED_DETECTOR is not None and _CACHED_PATH == target_path:
        return _CACHED_DETECTOR

    logger.info("Loading anomaly detector from %s", target_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Anomaly detector artifact not found at: {target_path}")

    detector = AnomalyDetector.load(target_path)
    _CACHED_DETECTOR = detector
    _CACHED_PATH = target_path
    return detector


def clear_detector_cache() -> None:
    """Clear the cached detector instance."""
    global _CACHED_DETECTOR, _CACHED_PATH
    _CACHED_DETECTOR = None
    _CACHED_PATH = None


def score_window(
    sensor_window_df: pd.DataFrame,
    machine_id: str | None = None,
    df_maint: pd.DataFrame | None = None,
    detector: AnomalyDetector | None = None,
    model_path: str | Path | None = None,
    eval_start: pd.Timestamp | str | None = None,
    return_all: bool = False,
) -> pd.DataFrame:
    """Score a window of raw cleaned sensor data for one or more machines.

    Parameters
    ----------
    sensor_window_df : pd.DataFrame
        Raw cleaned sensor rows (columns matching data/processed/sensors.parquet).
        Must contain timestamp, machine_id, and raw signal columns.
    machine_id : str | None
        If provided, filters sensor_window_df to this machine.
    df_maint : pd.DataFrame | None
        Maintenance records. If None, attempts to load from PROCESSED_DIR, or falls
        back to an empty schema if unavailable.
    detector : AnomalyDetector | None
        Preloaded detector instance. If None, uses load_detector().
    model_path : str | Path | None
        Path to detector artifact if loading on demand.
    eval_start : pd.Timestamp | str | None
        If specified, only return scored rows with timestamp >= eval_start.
    return_all : bool
        If True, returns all scored rows in the window. Otherwise returns rows
        after the required 24h history period (or after eval_start).

    Returns
    -------
    pd.DataFrame
        Scored DataFrame containing timestamp, machine_id, state, raw_score,
        anomaly_score, anomaly_status, and factor attributions.
    """
    if sensor_window_df.empty:
        raise ValueError("sensor_window_df is empty.")

    df = sensor_window_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if machine_id is not None:
        df = df[df["machine_id"] == machine_id]
        if df.empty:
            raise ValueError(f"No rows found for machine_id={machine_id} in provided window.")

    # Check window duration requirement
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    span = max_ts - min_ts
    min_span = pd.Timedelta(hours=MIN_HISTORY_HOURS)

    if span < min_span and len(df) < MIN_HISTORY_INTERVALS:
        raise ValueError(
            f"Window is too short for the longest rolling feature (24h). "
            f"Provided window spans {span} ({len(df)} intervals); required at least {min_span}."
        )

    # Resolve maintenance records
    if df_maint is None:
        maint_path = PROCESSED_DIR / "maintenance.parquet"
        if maint_path.exists():
            df_maint = pd.read_parquet(maint_path)
        else:
            df_maint = pd.DataFrame(columns=["machine_id", "performed_date", "maintenance_type"])

    # Build features using same pipeline as training (D9, D65)
    features_df = engineer_sensor_features(df_sensors=df, df_maint=df_maint)

    # Resolve detector
    if detector is None:
        detector = load_detector(model_path)

    # Score features
    scored_df = detector.score_frame(features_df)

    # Filter to requested evaluation period
    if return_all:
        return scored_df

    if eval_start is not None:
        eval_ts = pd.to_datetime(eval_start, utc=True)
        return scored_df[scored_df["timestamp"] >= eval_ts].reset_index(drop=True)

    # Default: return rows that have at least 24h history preceding them
    cutoff_ts = min_ts + pd.Timedelta(hours=MIN_HISTORY_HOURS)
    return scored_df[scored_df["timestamp"] >= cutoff_ts].reset_index(drop=True)
