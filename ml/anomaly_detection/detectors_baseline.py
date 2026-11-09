"""Baseline anomaly detectors: manual thresholds (B1) and robust z-score max (B2).

Implements Decision D60:
- B1: Manual threshold heuristic (vibration > 4.5 or temp > 85 for Warning,
  vibration > 7.0 for Critical)
- B2: Maximum absolute per-machine robust z-score across Feature Set A (plan_core).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ml.anomaly_detection.baselines import MachineBaseline
from ml.anomaly_detection.calibration import smooth_scores
from ml.anomaly_detection.config import (
    B1_TEMP_WARN,
    B1_VIB_CRIT,
    B1_VIB_WARN,
    FEATURE_SET_A,
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
from ml.anomaly_detection.masks import scoring_mask, training_mask

logger = logging.getLogger(__name__)


class B1ManualThresholdDetector:
    """B1 baseline detector using domain engineering limit thresholds.

    Rules:
    - Warning: vibration > 4.5 mm/s OR temperature > 85.0 deg_C (1h trailing mean)
    - Critical: vibration > 7.0 mm/s (1h trailing mean)
    - State gating: Offline and NoData intervals are not scored/alerted.
    """

    def __init__(
        self,
        vib_warn: float = B1_VIB_WARN,
        vib_crit: float = B1_VIB_CRIT,
        temp_warn: float = B1_TEMP_WARN,
        smoothing_window: int = 12,
    ) -> None:
        self.vib_warn = vib_warn
        self.vib_crit = vib_crit
        self.temp_warn = temp_warn
        self.smoothing_window = smoothing_window
        self.feature_list = ["vibration", "temperature"]

    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame | None = None,
        period: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ) -> B1ManualThresholdDetector:
        """B1 is rule-based and requires no fitting, but accepts fit arguments for uniformity."""
        return self

    def score_frame(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Score a frame of sensor features using manual threshold rules."""
        can_score, state = scoring_mask(features_df, self.feature_list)

        vib = pd.Series(np.nan, index=features_df.index, dtype=np.float32)
        temp = pd.Series(np.nan, index=features_df.index, dtype=np.float32)

        scored_idx = can_score.to_numpy()
        if np.any(scored_idx):
            vib.loc[scored_idx] = features_df.loc[scored_idx, "vibration"].astype(np.float32)
            temp.loc[scored_idx] = features_df.loc[scored_idx, "temperature"].astype(np.float32)

        # Smooth signals over 1h trailing window (restarting after gaps > 15m)
        vib_smooth = smooth_scores(
            mapped_scores=vib,
            machine_ids=features_df["machine_id"],
            timestamps=features_df["timestamp"],
            can_score=can_score,
            window=self.smoothing_window,
        )
        temp_smooth = smooth_scores(
            mapped_scores=temp,
            machine_ids=features_df["machine_id"],
            timestamps=features_df["timestamp"],
            can_score=can_score,
            window=self.smoothing_window,
        )

        # Determine status
        status = pd.Series(STATUS_NODATA, index=features_df.index, dtype="string")
        status[state == STATE_OFFLINE] = STATUS_OFFLINE

        running_mask = (state == STATE_RUNNING) & vib_smooth.notna() & temp_smooth.notna()
        v_s = vib_smooth.loc[running_mask]
        t_s = temp_smooth.loc[running_mask]

        is_crit = v_s >= self.vib_crit
        is_warn = (v_s >= self.vib_warn) | (t_s >= self.temp_warn)
        is_warn_only = is_warn & (~is_crit)
        is_norm = (~is_warn) & (~is_crit)

        status.loc[v_s[is_norm].index] = STATUS_NORMAL
        status.loc[v_s[is_warn_only].index] = STATUS_WARNING
        status.loc[v_s[is_crit].index] = STATUS_CRITICAL

        # Map to calibrated-like score: Normal -> 0.15, Warning -> 0.65, Critical -> 0.90
        score = pd.Series(np.nan, index=features_df.index, dtype=np.float32)
        score.loc[v_s[is_norm].index] = 0.15
        score.loc[v_s[is_warn_only].index] = 0.65
        score.loc[v_s[is_crit].index] = 0.90

        out = pd.DataFrame(index=features_df.index)
        out["machine_id"] = features_df["machine_id"]
        out["timestamp"] = features_df["timestamp"]
        out["state"] = state
        out["anomaly_score_raw"] = score
        out["anomaly_score_5min"] = score
        out["anomaly_score"] = score
        out["anomaly_status"] = status

        # Contributing factors for B1: vibration and temperature
        res_factors = self._explain(features_df, can_score)
        for c in res_factors.columns:
            out[c] = res_factors[c]

        return out

    def _explain(self, features_df: pd.DataFrame, can_score: pd.Series) -> pd.DataFrame:
        """Assign vibration and temperature deviations as factors."""
        n = len(features_df)
        res = pd.DataFrame(index=features_df.index)
        for rank in range(1, 4):
            res[f"factor_{rank}"] = pd.Series([None] * n, index=features_df.index, dtype="string")
            res[f"factor_{rank}_z"] = pd.Series(np.nan, index=features_df.index, dtype=np.float32)
            res[f"factor_{rank}_direction"] = pd.Series(
                [None] * n, index=features_df.index, dtype="string"
            )

        scored_idx = can_score.to_numpy()
        if not np.any(scored_idx):
            return res

        vib = features_df.loc[scored_idx, "vibration"]
        temp = features_df.loc[scored_idx, "temperature"]

        # Approximate z-like deviation relative to warning thresholds
        vib_ratio = (vib / self.vib_warn).to_numpy(dtype=np.float32)
        temp_ratio = (temp / self.temp_warn).to_numpy(dtype=np.float32)

        orig_idx = features_df.index[scored_idx]
        vib_higher = vib_ratio >= temp_ratio

        # Rank 1
        res.loc[orig_idx, "factor_1"] = np.where(vib_higher, "vibration", "temperature")
        res.loc[orig_idx, "factor_1_z"] = np.where(vib_higher, vib_ratio * 4.0, temp_ratio * 4.0)
        res.loc[orig_idx, "factor_1_direction"] = "high"

        # Rank 2
        res.loc[orig_idx, "factor_2"] = np.where(vib_higher, "temperature", "vibration")
        res.loc[orig_idx, "factor_2_z"] = np.where(vib_higher, temp_ratio * 4.0, vib_ratio * 4.0)
        res.loc[orig_idx, "factor_2_direction"] = "high"

        return res


class B2RobustZDetector:
    """B2 baseline detector: maximum absolute robust z across Feature Set A."""

    def __init__(
        self,
        z_warn: float = 4.0,
        z_crit: float = 8.0,
        smoothing_window: int = 12,
        feature_list: list[str] | None = None,
    ) -> None:
        self.z_warn = float(z_warn)
        self.z_crit = float(z_crit)
        self.smoothing_window = smoothing_window
        self.feature_list = list(feature_list or FEATURE_SET_A)
        self.baseline: MachineBaseline = MachineBaseline()

    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        period: tuple[pd.Timestamp, pd.Timestamp],
        machines_df: pd.DataFrame | None = None,
    ) -> B2RobustZDetector:
        """Fit per-machine robust baseline on training rows."""
        train_mask, _ = training_mask(features_df, labels_df, self.feature_list, period)
        train_features = features_df.loc[train_mask]
        self.baseline.fit(train_features, self.feature_list, machines_df)
        return self

    def score_frame(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Score a frame of sensor features using max robust z across feature set A."""
        can_score, state = scoring_mask(features_df, self.feature_list)
        n = len(features_df)

        z_max_raw = np.full(n, np.nan, dtype=np.float32)
        score_5min = np.full(n, np.nan, dtype=np.float32)

        scored_idx = can_score.to_numpy()
        if np.any(scored_idx):
            scored_df = features_df.loc[scored_idx]
            z_df = self.baseline.transform(scored_df)[self.feature_list]
            z_mat = z_df.to_numpy(dtype=np.float32)
            abs_z_mat = np.abs(z_mat)

            # Max absolute z per row
            max_abs_z = np.max(abs_z_mat, axis=1)
            z_max_raw[scored_idx] = max_abs_z

            # Map raw max z to [0, 1] scale:
            # 0 -> 0.0, 1 -> 0.10, z_warn -> 0.50, z_crit -> 0.80, sat -> 1.00
            diff = max(self.z_crit - self.z_warn, 1.0)
            z_sat = self.z_crit + 2.0 * diff

            xp = np.array([0.0, 1.0, self.z_warn, self.z_crit, z_sat], dtype=np.float64)
            fp = np.array(
                [
                    0.0,
                    STATUS_WARNING_THRESHOLD * 0.2,
                    STATUS_WARNING_THRESHOLD,
                    STATUS_CRITICAL_THRESHOLD,
                    1.0,
                ],
                dtype=np.float64,
            )
            mapped = np.interp(max_abs_z, xp, fp, left=0.0, right=1.0)
            score_5min[scored_idx] = mapped.astype(np.float32)

        # Trailing smoothing
        smoothed = smooth_scores(
            mapped_scores=pd.Series(score_5min, index=features_df.index),
            machine_ids=features_df["machine_id"],
            timestamps=features_df["timestamp"],
            can_score=can_score,
            window=self.smoothing_window,
        )

        status = pd.Series(STATUS_NODATA, index=features_df.index, dtype="string")
        status[state == STATE_OFFLINE] = STATUS_OFFLINE

        running_mask = (state == STATE_RUNNING) & smoothed.notna()
        scores = smoothed.loc[running_mask]

        normal_mask = scores < STATUS_WARNING_THRESHOLD
        warning_mask = (scores >= STATUS_WARNING_THRESHOLD) & (scores < STATUS_CRITICAL_THRESHOLD)
        critical_mask = scores >= STATUS_CRITICAL_THRESHOLD

        status.loc[scores[normal_mask].index] = STATUS_NORMAL
        status.loc[scores[warning_mask].index] = STATUS_WARNING
        status.loc[scores[critical_mask].index] = STATUS_CRITICAL

        out = pd.DataFrame(index=features_df.index)
        out["machine_id"] = features_df["machine_id"]
        out["timestamp"] = features_df["timestamp"]
        out["state"] = state
        out["anomaly_score_raw"] = z_max_raw
        out["anomaly_score_5min"] = score_5min
        out["anomaly_score"] = smoothed
        out["anomaly_status"] = status

        # Contributing factors from baseline transform
        res_factors = self._explain(features_df, can_score, top_k=3)
        for c in res_factors.columns:
            out[c] = res_factors[c]

        return out

    def _explain(
        self, features_df: pd.DataFrame, can_score: pd.Series, top_k: int = 3
    ) -> pd.DataFrame:
        """Top features by absolute robust z-score."""
        n = len(features_df)
        res = pd.DataFrame(index=features_df.index)
        for rank in range(1, top_k + 1):
            res[f"factor_{rank}"] = pd.Series([None] * n, index=features_df.index, dtype="string")
            res[f"factor_{rank}_z"] = pd.Series(np.nan, index=features_df.index, dtype=np.float32)
            res[f"factor_{rank}_direction"] = pd.Series(
                [None] * n, index=features_df.index, dtype="string"
            )

        scored_idx = can_score.to_numpy()
        if not np.any(scored_idx):
            return res

        scored_df = features_df.loc[scored_idx]
        z_df = self.baseline.transform(scored_df)[self.feature_list]
        z_mat = z_df.to_numpy(dtype=np.float32)
        abs_z_mat = np.abs(z_mat)

        feature_arr = np.array(self.feature_list)
        sorted_indices = np.argsort(-abs_z_mat, axis=1)[:, :top_k]
        orig_indices = features_df.index[scored_idx]

        for rank in range(top_k):
            idx_k = sorted_indices[:, rank]
            r_feats = feature_arr[idx_k]
            r_z = z_mat[np.arange(len(z_mat)), idx_k]
            r_dirs = np.where(r_z >= 0, "high", "low")

            r_num = rank + 1
            res.loc[orig_indices, f"factor_{r_num}"] = r_feats
            res.loc[orig_indices, f"factor_{r_num}_z"] = r_z
            res.loc[orig_indices, f"factor_{r_num}_direction"] = r_dirs

        return pd.DataFrame(res, index=features_df.index)
