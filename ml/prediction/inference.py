"""Serving inference module for failure risk prediction.

Implements Decision D80 and Day 7 Serving Contract:
- load_model(path=None): loads and caches FailureModel singleton in memory.
- predict_one(payload: dict) -> dict: single operating state prediction.
- predict_batch(df: pd.DataFrame) -> pd.DataFrame: batch operating state prediction.
- Provides input validation, feature derivation, probability calibration,
  risk categorization, input envelope warnings, model version, and framing statement.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ml.prediction.config import DEFAULT_FRAMING, MODEL_PKL_PATH
from ml.prediction.model import FailureModel
from ml.prediction.thresholds import RiskThresholds

logger = logging.getLogger(__name__)

# Cached model singleton
_MODEL_CACHE: FailureModel | None = None


def load_model(path: Path | str | None = None) -> FailureModel:
    """Load and cache FailureModel singleton (for FastAPI lifespan hook on Day 7)."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None and path is None:
        return _MODEL_CACHE

    target_path = Path(path) if path is not None else MODEL_PKL_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"FailureModel artifact not found at {target_path}")

    logger.info("Loading FailureModel bundle from %s...", target_path)
    model = FailureModel.load(target_path)
    if path is None:
        _MODEL_CACHE = model
    return model


def clear_model_cache() -> None:
    """Clear cached model singleton (used in test fixtures)."""
    global _MODEL_CACHE
    _MODEL_CACHE = None


def _format_thresholds(thresh: Any) -> dict[str, float]:
    """Extract standard threshold dictionary from RiskThresholds or dict."""
    if isinstance(thresh, RiskThresholds):
        return {
            "t_med": round(thresh.t_med, 4),
            "t_high": round(thresh.t_high, 4),
            "t_crit": round(thresh.t_crit, 4),
        }
    elif isinstance(thresh, dict):
        return {
            "t_med": round(float(thresh.get("t_med", 0.046)), 4),
            "t_high": round(float(thresh.get("t_high", 0.50)), 4),
            "t_crit": round(float(thresh.get("t_crit", 0.50)), 4),
        }
    return {"t_med": 0.046, "t_high": 0.50, "t_crit": 0.50}


def predict_one(
    payload: dict[str, Any],
    model: FailureModel | None = None,
) -> dict[str, Any]:
    """Score a single industrial operating state payload.

    Parameters
    ----------
    payload : dict
        Raw operating features: type (L/M/H), air_temperature_k (or air_temp_k),
        process_temperature_k (or process_temp_k), rotational_speed_rpm (or rpm),
        torque_nm, tool_wear_min.
    model : FailureModel, optional
        Explicit model instance (defaults to cached load_model()).

    Returns
    -------
    dict
        Serving response conforming to Day 7 API contract (D80):
        failure_probability, risk_level, thresholds, model_version, input_warnings, framing.
    """
    active_model = model or load_model()
    df = pd.DataFrame([payload])
    res_df = active_model.predict_risk(df)
    row = res_df.iloc[0]

    warnings = row["input_warnings"]
    warning_list = warnings if isinstance(warnings, list) else ([warnings] if warnings else [])

    return {
        "failure_probability": float(row["failure_probability"]),
        "risk_level": str(row["risk_level"]),
        "thresholds": _format_thresholds(active_model.thresholds),
        "model_version": str(row["model_version"]),
        "input_warnings": warning_list,
        "framing": DEFAULT_FRAMING,
    }


def predict_batch(
    df: pd.DataFrame,
    model: FailureModel | None = None,
) -> pd.DataFrame:
    """Score a batch of industrial operating states.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing operational input columns.
    model : FailureModel, optional
        Explicit model instance (defaults to cached load_model()).

    Returns
    -------
    pd.DataFrame
        Scored batch containing failure_probability, risk_level, threshold_med,
        threshold_high, threshold_crit, input_warnings, model_version, framing.
    """
    active_model = model or load_model()
    return active_model.predict_risk(df)
