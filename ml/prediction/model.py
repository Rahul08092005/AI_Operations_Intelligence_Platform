"""Core FailureModel class and model artifact bundle.

Implements Decision D70, D74, D75, D79, D80:
- Wraps fitted XGBoost booster, calibrator, thresholds, and training metadata.
- Methods: fit, predict_proba, predict_margin, predict_risk, save, load.
- Provides booster margins for SHAP additivity checks (TreeExplainer readiness).
- Houses training statistics for input validation warnings and background sample for Day 10.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

from ml.prediction.calibration import IsotonicCalibrator, SigmoidCalibrator
from ml.prediction.config import (
    DEFAULT_FRAMING,
    MODEL_VERSION,
    SEED,
    SHAP_BACKGROUND_SIZE,
)
from ml.prediction.features import build_features, get_feature_names
from ml.prediction.thresholds import RiskThresholds, assign_risk_category
from ml.prediction.validation import validate_inputs

logger = logging.getLogger(__name__)


class FailureModel:
    """End-to-end failure prediction model bundle."""

    def __init__(
        self,
        booster: XGBClassifier | None = None,
        calibrator: SigmoidCalibrator | IsotonicCalibrator | None = None,
        thresholds: RiskThresholds | dict[str, float] | None = None,
        feature_set: str = "physics",
        metadata: dict[str, Any] | None = None,
        shap_background: pd.DataFrame | None = None,
        training_stats: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.booster = booster
        self.calibrator = calibrator
        self.thresholds = thresholds
        self.feature_set = feature_set
        self.metadata = metadata or {}
        self.shap_background = shap_background
        self.training_stats = training_stats or {}
        self.model_version = self.metadata.get("model_version", MODEL_VERSION)

    def fit(
        self,
        train_df: pd.DataFrame,
        params: dict[str, Any],
        feature_set: str = "physics",
        calibrator: SigmoidCalibrator | IsotonicCalibrator | None = None,
        thresholds: RiskThresholds | dict[str, float] | None = None,
        oof_summary: dict[str, Any] | None = None,
        git_hash: str | None = None,
    ) -> FailureModel:
        """Fit final XGBoost booster on training partition and assemble bundle."""
        y_train = train_df["machine_failure"].to_numpy(dtype=np.int32)
        X_train = build_features(train_df, feature_set=feature_set)

        logger.info(
            "Fitting final XGBClassifier on %d rows (%d failures, feature_set=%s)...",
            len(train_df),
            int(np.sum(y_train)),
            feature_set,
        )

        booster = XGBClassifier(**params)
        booster.fit(X_train, y_train)

        # Store training stats for validation envelope
        training_stats: dict[str, dict[str, float]] = {}
        for col in [
            "air_temperature_k",
            "process_temperature_k",
            "rotational_speed_rpm",
            "torque_nm",
            "tool_wear_min",
        ]:
            col_match = [c for c in train_df.columns if c.lower() == col or col in c.lower()]
            if col_match:
                c = col_match[0]
                training_stats[col] = {
                    "min": float(train_df[c].min()),
                    "max": float(train_df[c].max()),
                    "mean": float(train_df[c].mean()),
                    "std": float(train_df[c].std()),
                }

        # Background sample for SHAP (200 deterministic rows)
        rng = np.random.default_rng(SEED)
        bg_size = min(SHAP_BACKGROUND_SIZE, len(X_train))
        bg_indices = rng.choice(len(X_train), size=bg_size, replace=False)
        shap_bg = X_train.iloc[bg_indices].copy().reset_index(drop=True)

        # Package metadata
        now_utc = datetime.datetime.now(datetime.UTC).isoformat()
        thresh_dict = (
            thresholds.to_dict()
            if isinstance(thresholds, RiskThresholds)
            else (thresholds or {})
        )

        def _get_pkg_version(name: str) -> str:
            try:
                return importlib.metadata.version(name)
            except Exception:
                return "unknown"

        meta: dict[str, Any] = {
            "model_version": MODEL_VERSION,
            "created_at_utc": now_utc,
            "git_commit": git_hash or "unknown",
            "feature_set": feature_set,
            "feature_names": get_feature_names(feature_set),
            "hyperparameters": params,
            "thresholds": thresh_dict,
            "calibrator_type": type(calibrator).__name__ if calibrator else "None",
            "n_train_rows": len(train_df),
            "n_train_failures": int(np.sum(y_train)),
            "train_prevalence": float(np.mean(y_train)),
            "oof_summary": oof_summary or {},
            "library_versions": {
                "xgboost": _get_pkg_version("xgboost"),
                "scikit-learn": _get_pkg_version("scikit-learn"),
                "numpy": _get_pkg_version("numpy"),
                "pandas": _get_pkg_version("pandas"),
                "joblib": _get_pkg_version("joblib"),
            },
            "framing": DEFAULT_FRAMING,
        }

        self.booster = booster
        self.calibrator = calibrator
        self.thresholds = thresholds
        self.feature_set = feature_set
        self.metadata = meta
        self.shap_background = shap_bg
        self.training_stats = training_stats
        self.model_version = MODEL_VERSION

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict calibrated failure probabilities for input operating states."""
        if self.booster is None:
            raise ValueError("Model is not fitted.")

        X = build_features(df, feature_set=self.feature_set)
        raw_probs = self.booster.predict_proba(X)[:, 1]

        if self.calibrator is not None:
            return self.calibrator.predict_proba(raw_probs)
        return raw_probs

    def predict_margin(self, df: pd.DataFrame) -> np.ndarray:
        """Predict raw booster margin (log-odds) for SHAP additivity verification."""
        if self.booster is None:
            raise ValueError("Model is not fitted.")

        X = build_features(df, feature_set=self.feature_set)
        dmat = xgb.DMatrix(X)
        return self.booster.get_booster().predict(dmat, output_margin=True)

    def predict_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Serving prediction returning probabilities, categories, warnings, and metadata."""
        if self.booster is None:
            raise ValueError("Model is not fitted.")

        # Input validation
        val_result = validate_inputs(df, self.training_stats)

        # Calibrated probability
        probs = self.predict_proba(df)

        # Categories
        categories = assign_risk_category(probs, self.thresholds)

        # Extract numeric threshold values
        if isinstance(self.thresholds, RiskThresholds):
            t_med, t_high, t_crit = (
                self.thresholds.t_med,
                self.thresholds.t_high,
                self.thresholds.t_crit,
            )
        elif isinstance(self.thresholds, dict):
            t_med, t_high, t_crit = (
                float(self.thresholds["t_med"]),
                float(self.thresholds["t_high"]),
                float(self.thresholds["t_crit"]),
            )
        else:
            t_med, t_high, t_crit = 0.046, 0.5, 0.5

        result_df = pd.DataFrame(
            {
                "failure_probability": np.round(probs, 4),
                "risk_level": categories,
                "threshold_med": t_med,
                "threshold_high": t_high,
                "threshold_crit": t_crit,
                "input_warnings": [w if w else None for w in val_result.warnings],
                "model_version": self.model_version,
                "framing": DEFAULT_FRAMING,
            }
        )

        return result_df

    def save(self, path: Path | str) -> None:
        """Save model bundle to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target_path)
        logger.info("Saved FailureModel bundle to %s", target_path)

    @classmethod
    def load(cls, path: Path | str) -> FailureModel:
        """Load model bundle from disk."""
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"FailureModel artifact not found at: {target_path}")
        model = joblib.load(target_path)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not a FailureModel: {type(model)}")
        return model
