"""AnomalyDetector class with fit, score, explain, and persistence.

Implements Decision D53 (One global Isolation Forest on per-machine robust z-scores),
Decision D54 (State gating), Decision D57 (Piecewise-linear score calibration),
and Decision D61 (Contributing factors).
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

from ml.anomaly_detection.baselines import MachineBaseline
from ml.anomaly_detection.calibration import ScoreCalibrator, smooth_scores, to_status
from ml.anomaly_detection.config import (
    FEATURE_SETS,
    SEED,
    STATUS_CRITICAL_THRESHOLD,
    STATUS_WARNING_THRESHOLD,
)
from ml.anomaly_detection.masks import scoring_mask, training_mask

logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Return the current short git commit hash or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


class AnomalyDetector:
    """Global Isolation Forest anomaly detector with robust baseline normalization."""

    def __init__(self) -> None:
        self.model: IsolationForest | None = None
        self.baseline: MachineBaseline = MachineBaseline()
        self.calibrator: ScoreCalibrator = ScoreCalibrator()
        self.feature_set_name: str = "plan_core"
        self.feature_list: list[str] = []
        self.smoothing_window: int = 12
        self.status_thresholds: dict[str, float] = {
            "warning": STATUS_WARNING_THRESHOLD,
            "critical": STATUS_CRITICAL_THRESHOLD,
        }
        self.training_period: tuple[str, str] = ("", "")
        self.n_training_rows: int = 0
        self.params: dict[str, Any] = {}
        self.post_params: dict[str, Any] = {}
        self.library_versions: dict[str, str] = {}
        self.seed: int = SEED
        self.created_at: str = ""
        self.git_commit: str = "unknown"
        self.metrics_summary: dict[str, Any] | None = None
        self.exclusion_report: list[dict[str, Any]] | None = None

    def fit(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        period: tuple[pd.Timestamp, pd.Timestamp],
        params: dict[str, Any] | None = None,
        feature_set: str = "plan_core",
        post: dict[str, Any] | None = None,
        machines_df: pd.DataFrame | None = None,
    ) -> AnomalyDetector:
        """Fit baseline normalization, Isolation Forest, and score calibrator.

        Parameters
        ----------
        features_df : pd.DataFrame
            Complete features DataFrame.
        labels_df : pd.DataFrame
            Ground truth evaluation labels.
        period : tuple[pd.Timestamp, pd.Timestamp]
            (start_time, end_time) of training period (end_time exclusive).
        params : dict[str, Any] | None
            Isolation Forest parameters (e.g. n_estimators, max_samples, max_features).
        feature_set : str
            Feature set name: 'plan_core', 'core_context', 'core_temporal', or 'real_only'.
        post : dict[str, Any] | None
            Post-processing parameters (q_warn, q_crit, smoothing_window).
        machines_df : pd.DataFrame | None
            Optional machine metadata for equipment type grouping.
        """
        self.feature_set_name = feature_set
        if feature_set in FEATURE_SETS:
            self.feature_list = list(FEATURE_SETS[feature_set])
        else:
            raise ValueError(f"Unknown feature set: {feature_set}. Available: {list(FEATURE_SETS)}")

        self.params = dict(
            params or {"n_estimators": 200, "max_samples": 1024, "max_features": 1.0}
        )
        self.post_params = dict(
            post or {"q_warn": 0.995, "q_crit": 0.9995, "smoothing_window": 12}
        )
        self.smoothing_window = int(self.post_params.get("smoothing_window", 12))
        self.seed = SEED

        start_ts, end_ts = period
        self.training_period = (str(start_ts), str(end_ts))

        # 1. Training mask (Rule R2, R4, R5, D54, D56)
        train_mask, excl_report = training_mask(features_df, labels_df, self.feature_list, period)
        self.exclusion_report = excl_report.to_dict(orient="records")

        train_features = features_df.loc[train_mask].copy()
        self.n_training_rows = len(train_features)
        if self.n_training_rows == 0:
            raise ValueError("Training mask resulted in 0 rows.")

        logger.info(
            "Fitting %s detector on %d training rows (period %s to %s)",
            self.feature_set_name,
            self.n_training_rows,
            start_ts,
            end_ts,
        )

        # 2. Fit per-machine robust baselines (Rule R3, D53)
        self.baseline = MachineBaseline().fit(train_features, self.feature_list, machines_df)

        # 3. Transform training features to robust z-scores
        z_train = self.baseline.transform(train_features)[self.feature_list].to_numpy(
            dtype=np.float32
        )

        # 4. Fit Isolation Forest
        iforest_kwargs = dict(self.params)
        iforest_kwargs["random_state"] = self.seed
        iforest_kwargs["contamination"] = "auto"
        iforest_kwargs["bootstrap"] = False
        iforest_kwargs["n_jobs"] = -1

        self.model = IsolationForest(**iforest_kwargs)
        self.model.fit(z_train)

        # 5. Compute raw anomaly scores on normal training data (D57)
        # raw = -score_samples(X) (higher = more anomalous)
        raw_train = -self.model.score_samples(z_train)

        # 6. Fit ScoreCalibrator against normal training quantiles
        self.calibrator = ScoreCalibrator().fit(
            raw_train_scores=raw_train,
            q_warn=self.post_params.get("q_warn", 0.995),
            q_crit=self.post_params.get("q_crit", 0.9995),
        )

        # 7. Metadata capture
        self.created_at = datetime.now(UTC).isoformat()
        self.git_commit = get_git_commit_hash()
        self.library_versions = {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        }

        return self

    def score_frame(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Score a frame of sensor features.

        Returns DataFrame containing:
        - machine_id, timestamp
        - state ('Running', 'Offline', 'NoData')
        - anomaly_score_raw
        - anomaly_score_5min
        - anomaly_score (smoothed trailing 1h mean)
        - anomaly_status ('Normal', 'Warning', 'Critical', 'Offline', 'NoData')
        - factor_1, factor_1_z, factor_1_direction
        - factor_2, factor_2_z, factor_2_direction
        - factor_3, factor_3_z, factor_3_direction
        """
        if self.model is None:
            raise RuntimeError("AnomalyDetector must be fitted or loaded before scoring.")

        # Validate required columns
        required_cols = {"machine_id", "timestamp", "running"}.union(self.feature_list)
        missing = required_cols - set(features_df.columns)
        if missing:
            raise ValueError(f"Features DataFrame missing required columns: {sorted(missing)}")

        # 1. State gating (Rule R5, Decision D54)
        can_score, state = scoring_mask(features_df, self.feature_list)

        n = len(features_df)
        raw_scores = np.full(n, np.nan, dtype=np.float32)
        score_5min = np.full(n, np.nan, dtype=np.float32)

        # 2. Score only running, complete rows
        scored_idx = can_score.to_numpy()
        if np.any(scored_idx):
            scored_df = features_df.loc[scored_idx]
            z_scored = self.baseline.transform(scored_df)[self.feature_list].to_numpy(
                dtype=np.float32
            )

            raw_vals = -self.model.score_samples(z_scored)
            raw_scores[scored_idx] = raw_vals.astype(np.float32)

            mapped_vals = self.calibrator.map_scores(raw_vals)
            score_5min[scored_idx] = mapped_vals.astype(np.float32)

        # 3. Trailing smoothing (Decision D57)
        smoothed = smooth_scores(
            mapped_scores=pd.Series(score_5min, index=features_df.index),
            machine_ids=features_df["machine_id"],
            timestamps=features_df["timestamp"],
            can_score=can_score,
            window=self.smoothing_window,
        )

        # 4. Status mapping (Decision D57)
        status = to_status(smoothed, state)

        # Assemble output DataFrame
        out = pd.DataFrame(index=features_df.index)
        out["machine_id"] = features_df["machine_id"]
        out["timestamp"] = features_df["timestamp"]
        out["state"] = state
        out["anomaly_score_raw"] = raw_scores
        out["anomaly_score_5min"] = score_5min
        out["anomaly_score"] = smoothed
        out["anomaly_status"] = status

        # 5. Explain contributing factors (Decision D61)
        explanations = self.explain(features_df, can_score=can_score, top_k=3)
        for col in explanations.columns:
            out[col] = explanations[col]

        return out

    def explain(
        self,
        features_df: pd.DataFrame,
        can_score: pd.Series | None = None,
        top_k: int = 3,
    ) -> pd.DataFrame:
        """Compute contributing factors based on absolute per-machine robust z-scores (D61).

        Top k features by |Z|, direction ('high' vs 'low'), and anomaly_type rule.
        """
        n = len(features_df)
        if can_score is None:
            can_score, _ = scoring_mask(features_df, self.feature_list)

        # Pre-allocate factor columns
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

        # Get top-k indices per row via argpartition/argsort
        feature_arr = np.array(self.feature_list)

        # Sort features descending by absolute z
        sorted_indices = np.argsort(-abs_z_mat, axis=1)[:, :top_k]

        orig_scored_indices = features_df.index[scored_idx]

        for rank in range(top_k):
            rank_feat_idx = sorted_indices[:, rank]
            rank_feats = feature_arr[rank_feat_idx]
            rank_z = z_mat[np.arange(len(z_mat)), rank_feat_idx]
            rank_dirs = np.where(rank_z >= 0, "high", "low")

            r_num = rank + 1
            res.loc[orig_scored_indices, f"factor_{r_num}"] = rank_feats
            res.loc[orig_scored_indices, f"factor_{r_num}_z"] = rank_z
            res.loc[orig_scored_indices, f"factor_{r_num}_direction"] = rank_dirs

        return pd.DataFrame(res, index=features_df.index)

    def save(self, model_path: Path | str, metadata_path: Path | str | None = None) -> None:
        """Save the fitted detector to pkl and human-readable metadata to json."""
        m_path = Path(model_path)
        m_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model pickle with joblib
        joblib.dump(self, m_path, compress=3)
        logger.info("Saved model artifact to %s", m_path)

        # Save readable metadata
        if metadata_path is None:
            metadata_path = m_path.parent / f"{m_path.stem}_metadata.json"
        meta_path = Path(metadata_path)

        meta_dict = self.get_metadata()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)
        logger.info("Saved model metadata to %s", meta_path)

    def get_metadata(self) -> dict[str, Any]:
        """Return comprehensive metadata dictionary."""
        return {
            "feature_set_name": self.feature_set_name,
            "feature_list": self.feature_list,
            "n_features": len(self.feature_list),
            "smoothing_window": self.smoothing_window,
            "status_thresholds": self.status_thresholds,
            "training_period": self.training_period,
            "n_training_rows": self.n_training_rows,
            "params": self.params,
            "post_params": self.post_params,
            "calibrator": self.calibrator.to_dict(),
            "library_versions": self.library_versions,
            "seed": self.seed,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "metrics_summary": self.metrics_summary,
            "exclusion_report": self.exclusion_report,
        }

    @classmethod
    def load(cls, path: Path | str) -> AnomalyDetector:
        """Load fitted AnomalyDetector instance from pickle."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        obj = joblib.load(p)
        if not isinstance(obj, AnomalyDetector):
            raise TypeError(f"Loaded object is not AnomalyDetector (got {type(obj)})")
        return obj
