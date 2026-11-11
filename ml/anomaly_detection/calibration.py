"""Score calibration, smoothing, and status mapping.

Implements Decision D57:
- Raw score definition: raw = -score_samples(X) (higher = more anomalous)
- Piecewise-linear monotonic score calibration anchored to normal training quantiles:
  * median -> 0.10
  * q_warn -> 0.50
  * q_crit -> 0.80
  * saturation -> 1.00 (at raw_q_crit + 2 * (raw_q_crit - raw_q_warn))
- Trailing mean smoothing (window intervals, min_periods = window // 2)
  restarts across Offline or NoData gaps > 15 minutes
- Status mapping: Normal (< 0.5), Warning (0.5 to < 0.8), Critical (>= 0.8),
  Offline and NoData from state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml.anomaly_detection.config import (
    CALIBRATION_ANCHORS,
    STATE_OFFLINE,
    STATE_RUNNING,
    STATUS_CRITICAL,
    STATUS_CRITICAL_THRESHOLD,
    STATUS_NODATA,
    STATUS_NORMAL,
    STATUS_OFFLINE,
    STATUS_WARNING,
    STATUS_WARNING_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class ScoreCalibrator:
    """Calibrates raw Isolation Forest scores to a documented [0, 1] range."""

    q_warn: float = 0.995
    q_crit: float = 0.9995
    raw_min: float = 0.0
    raw_median: float = 0.5
    raw_q_warn: float = 0.6
    raw_q_crit: float = 0.7
    raw_sat: float = 0.9

    def fit(
        self,
        raw_train_scores: np.ndarray,
        q_warn: float = 0.995,
        q_crit: float = 0.9995,
    ) -> ScoreCalibrator:
        """Fit calibration anchors strictly on normal training raw scores.

        Parameters
        ----------
        raw_train_scores : np.ndarray
            1D array of raw scores (-score_samples(X)) from normal training rows.
        q_warn : float
            Quantile of normal training scores mapped to 0.50 (Warning threshold).
        q_crit : float
            Quantile of normal training scores mapped to 0.80 (Critical threshold).
        """
        valid = raw_train_scores[~np.isnan(raw_train_scores)]
        if len(valid) == 0:
            raise ValueError("No valid scores to fit ScoreCalibrator.")

        self.q_warn = float(q_warn)
        self.q_crit = float(q_crit)

        self.raw_min = float(np.min(valid))
        self.raw_median = float(np.median(valid))
        self.raw_q_warn = float(np.quantile(valid, self.q_warn))
        self.raw_q_crit = float(np.quantile(valid, self.q_crit))

        # Enforce strict monotonicity among anchors with a small epsilon
        eps = 1e-6
        if self.raw_median <= self.raw_min:
            self.raw_median = self.raw_min + eps
        if self.raw_q_warn <= self.raw_median:
            self.raw_q_warn = self.raw_median + eps
        if self.raw_q_crit <= self.raw_q_warn:
            self.raw_q_crit = self.raw_q_warn + eps

        # Saturation point: raw_q_crit + 2 * (raw_q_crit - raw_q_warn)
        diff = self.raw_q_crit - self.raw_q_warn
        self.raw_sat = float(self.raw_q_crit + 2.0 * max(diff, eps))

        logger.debug(
            "Calibrator fitted: min=%.4f, med=%.4f, q_warn=%.4f, q_crit=%.4f, sat=%.4f",
            self.raw_min,
            self.raw_median,
            self.raw_q_warn,
            self.raw_q_crit,
            self.raw_sat,
        )
        return self

    def map_scores(self, raw: np.ndarray | pd.Series) -> np.ndarray:
        """Map raw scores to [0, 1] using piecewise linear interpolation.

        Anchors:
        - raw <= raw_min: 0.00
        - raw == raw_median: 0.10
        - raw == raw_q_warn: 0.50
        - raw == raw_q_crit: 0.80
        - raw >= raw_sat: 1.00
        """
        is_pandas = isinstance(raw, pd.Series)
        raw_arr = raw.to_numpy(dtype=np.float64) if is_pandas else np.asarray(raw, dtype=np.float64)

        xp = np.array(
            [self.raw_min, self.raw_median, self.raw_q_warn, self.raw_q_crit, self.raw_sat],
            dtype=np.float64,
        )
        fp = np.array(
            [
                0.00,
                CALIBRATION_ANCHORS["median_score"],
                CALIBRATION_ANCHORS["warn_score"],
                CALIBRATION_ANCHORS["crit_score"],
                CALIBRATION_ANCHORS["saturation_score"],
            ],
            dtype=np.float64,
        )

        mapped = np.interp(raw_arr, xp, fp, left=0.0, right=1.0)
        # Preserve NaNs from input
        mapped[np.isnan(raw_arr)] = np.nan
        return mapped.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        """Serialize calibrator parameters to dictionary."""
        return {
            "q_warn": self.q_warn,
            "q_crit": self.q_crit,
            "raw_min": self.raw_min,
            "raw_median": self.raw_median,
            "raw_q_warn": self.raw_q_warn,
            "raw_q_crit": self.raw_q_crit,
            "raw_sat": self.raw_sat,
            "anchors": dict(CALIBRATION_ANCHORS),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoreCalibrator:
        """Reconstruct ScoreCalibrator from serialized dictionary."""
        return cls(
            q_warn=data["q_warn"],
            q_crit=data["q_crit"],
            raw_min=data["raw_min"],
            raw_median=data["raw_median"],
            raw_q_warn=data["raw_q_warn"],
            raw_q_crit=data["raw_q_crit"],
            raw_sat=data["raw_sat"],
        )


def smooth_scores(
    mapped_scores: pd.Series,
    machine_ids: pd.Series,
    timestamps: pd.Series,
    can_score: pd.Series,
    window: int = 12,
    max_gap_minutes: float = 15.0,
) -> pd.Series:
    """Compute trailing mean per machine on consecutive scored intervals only.

    Parameters
    ----------
    mapped_scores : pd.Series
        Mapped 5-minute scores in [0, 1].
    machine_ids : pd.Series
        Machine identifier per row.
    timestamps : pd.Series
        Timestamps of each interval.
    can_score : pd.Series
        Boolean mask of scored rows.
    window : int
        Rolling window size in intervals (default 12 = 1 hour on 5-min grid).
    max_gap_minutes : float
        Maximum allowed gap between consecutive scored readings before window restarts.

    Returns
    -------
    smoothed : pd.Series[float]
        Trailing smoothed anomaly score, NaN on non-scored rows.
    """
    smoothed = pd.Series(np.nan, index=mapped_scores.index, dtype=np.float32)
    min_periods = max(1, window // 2)

    df_scored = pd.DataFrame(
        {
            "machine_id": machine_ids,
            "timestamp": timestamps,
            "score": mapped_scores,
            "orig_idx": mapped_scores.index,
        }
    )[can_score.to_numpy()].copy()

    if len(df_scored) == 0:
        return smoothed

    df_scored["timestamp"] = pd.to_datetime(df_scored["timestamp"], utc=True)
    df_scored = df_scored.sort_values(["machine_id", "timestamp"])

    # Process per machine
    for _, grp in df_scored.groupby("machine_id", observed=True):
        ts = grp["timestamp"]
        # Detect gaps > max_gap_minutes between consecutive scored intervals
        delta_min = ts.diff().dt.total_seconds().fillna(0.0) / 60.0
        gap_breaks = (delta_min > max_gap_minutes).cumsum()

        # Compute rolling trailing mean within each continuous block
        rolled = (
            grp["score"]
            .groupby(gap_breaks)
            .rolling(window=window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
        )

        smoothed.loc[grp["orig_idx"]] = rolled.to_numpy(dtype=np.float32)

    return smoothed


def to_status(anomaly_scores: pd.Series, states: pd.Series) -> pd.Series:
    """Convert smoothed anomaly scores and states to standard status vocabulary.

    Status levels:
    - If state == 'Offline': 'Offline'
    - If state != 'Running' or anomaly_score is NaN: 'NoData'
    - Else if anomaly_score < 0.50: 'Normal'
    - Else if anomaly_score < 0.80: 'Warning'
    - Else: 'Critical'
    """
    status = pd.Series(STATUS_NODATA, index=anomaly_scores.index, dtype="string")

    # Offline state
    offline_mask = states == STATE_OFFLINE
    status[offline_mask] = STATUS_OFFLINE

    # Running rows with valid scores
    running_mask = (states == STATE_RUNNING) & anomaly_scores.notna()

    scores = anomaly_scores.loc[running_mask]
    normal_mask = scores < STATUS_WARNING_THRESHOLD
    warning_mask = (scores >= STATUS_WARNING_THRESHOLD) & (scores < STATUS_CRITICAL_THRESHOLD)
    critical_mask = scores >= STATUS_CRITICAL_THRESHOLD

    status.loc[scores[normal_mask].index] = STATUS_NORMAL
    status.loc[scores[warning_mask].index] = STATUS_WARNING
    status.loc[scores[critical_mask].index] = STATUS_CRITICAL

    return status
