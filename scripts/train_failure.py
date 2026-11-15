"""Train and package the final FailureModel artifact.

Implements Task T7, Decision D70, D74, D75, D79:
1. Loads full AI4I training partition (8,000 rows).
2. Selects primary hyperparameter configuration from failure_experiments.csv.
3. Fits calibrator and derives risk thresholds from OOF predictions.
4. Fits final XGBoost model on full training set.
5. Performs SHAP readiness check (TreeExplainer additivity check on 500 rows).
6. Serializes FailureModel to models/failure_model.pkl and models/failure_model_metadata.json.
7. Verifies serialization parity (1e-9 tolerance).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from ml.prediction.calibration import choose_calibrator
from ml.prediction.config import (
    EXPERIMENTS_LOG_PATH,
    MODEL_METADATA_PATH,
    MODEL_PKL_PATH,
    OOF_PREDICTIONS_PATH,
    XGB_FIXED_PARAMS,
)
from ml.prediction.data import load_train
from ml.prediction.features import build_features
from ml.prediction.model import FailureModel
from ml.prediction.search import select_primary
from ml.prediction.thresholds import compute_thresholds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_failure")


def get_git_commit_hash() -> str:
    """Retrieve current git HEAD commit hash if available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    logger.info("Starting Failure Model training and packaging (Task T7)...")

    # 1. Load training data
    train_df = load_train()
    logger.info("Loaded training data: %d rows", len(train_df))

    # 2. Load search experiments and select primary configuration
    exp_file = Path(EXPERIMENTS_LOG_PATH)
    if not exp_file.exists():
        raise FileNotFoundError(f"Experiments log not found at {exp_file}. Run search first.")

    exp_df = pd.read_csv(exp_file)
    primary_cfg = select_primary(exp_df)
    primary_id = primary_cfg["config_id"]
    logger.info("Selected primary config: %s", primary_id)

    # 3. Load OOF predictions for calibrator selection and thresholds
    oof_file = Path(OOF_PREDICTIONS_PATH)
    if not oof_file.exists():
        raise FileNotFoundError(f"OOF predictions not found at {oof_file}. Run search first.")

    oof_df = pd.read_parquet(oof_file)
    # Use repeat 1 for calibration and thresholds (Decision D75)
    rep1_oof = oof_df[oof_df["repeat"] == 1].reset_index(drop=True)

    logger.info("Evaluating calibrator options on repeat 1 OOF predictions...")
    comparison, calibrator = choose_calibrator(
        oof_probs=rep1_oof["y_prob"].to_numpy(),
        y_true=rep1_oof["y_true"].to_numpy(),
        fold_indices=rep1_oof["fold"].to_numpy(),
    )
    logger.info("Selected calibrator: %s (%s)", comparison.chosen_method, comparison.reason)

    # Calibrate repeat 1 OOF probabilities and compute thresholds
    cal_oof_probs = calibrator.predict_proba(rep1_oof["y_prob"].to_numpy())
    thresholds = compute_thresholds(rep1_oof["y_true"].to_numpy(), cal_oof_probs)
    logger.info("Derived thresholds: %s", thresholds.to_dict())

    # 4. Prepare parameters for final XGBoost fit
    xgb_params = {
        **XGB_FIXED_PARAMS,
        "n_estimators": int(primary_cfg["n_estimators"]),
        "max_depth": int(primary_cfg["max_depth"]),
        "learning_rate": float(primary_cfg["learning_rate"]),
        "subsample": float(primary_cfg["subsample"]),
        "colsample_bytree": float(primary_cfg["colsample_bytree"]),
        "min_child_weight": int(primary_cfg["min_child_weight"]),
        "reg_lambda": float(primary_cfg["reg_lambda"]),
        "gamma": float(primary_cfg["gamma"]),
        "scale_pos_weight": float(primary_cfg["scale_pos_weight"]),
    }

    oof_summary = {
        "primary_config_id": primary_id,
        "mean_pr_auc": float(primary_cfg["mean_pr_auc"]),
        "mean_roc_auc": float(primary_cfg["mean_roc_auc"]),
        "mean_brier": float(primary_cfg["mean_brier"]),
        "calibrator_comparison": {
            "chosen": comparison.chosen_method,
            "brier_improvement_pct": comparison.brier_improvement_pct,
            "sigmoid_brier": comparison.sigmoid_brier,
            "isotonic_brier": comparison.isotonic_brier,
            "sigmoid_ece": comparison.sigmoid_ece,
            "isotonic_ece": comparison.isotonic_ece,
            "reason": comparison.reason,
        },
    }

    git_hash = get_git_commit_hash()

    # 5. Fit full FailureModel bundle
    model = FailureModel()
    model.fit(
        train_df=train_df,
        params=xgb_params,
        feature_set=str(primary_cfg["feature_set"]),
        calibrator=calibrator,
        thresholds=thresholds,
        oof_summary=oof_summary,
        git_hash=git_hash,
    )

    # 6. SHAP readiness check (500 training rows, TreeExplainer additivity check)
    logger.info("Running SHAP readiness check (500 rows, TreeExplainer additivity)...")
    X_train = build_features(train_df, feature_set=str(primary_cfg["feature_set"]))
    sample_500 = X_train.iloc[:500]

    booster = model.booster
    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_values(sample_500)
    expected_val = explainer.expected_value

    # Compute booster margins directly
    margins = model.predict_margin(train_df.iloc[:500])

    # For binary classification with single output margin:
    shap_sums = np.sum(shap_vals, axis=1) + expected_val
    max_diff = float(np.max(np.abs(shap_sums - margins)))
    logger.info("SHAP additivity maximum absolute difference: %.2e", max_diff)

    assert max_diff <= 1e-3, f"SHAP additivity check failed: max_diff {max_diff:.2e} > 1e-3"
    logger.info("SHAP readiness check PASSED! Model is 100%% ready for Day 10 explanations.")
    model.metadata["shap_readiness"] = {
        "status": "PASSED",
        "max_additivity_diff": max_diff,
        "sample_size": 500,
    }

    # 7. Save model artifact and metadata
    pkl_path = Path(MODEL_PKL_PATH)
    meta_path = Path(MODEL_METADATA_PATH)

    model.save(pkl_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(model.metadata, f, indent=2)
    logger.info("Saved metadata JSON to %s", meta_path)

    # 8. Reload check in fresh instance
    logger.info("Verifying save/load serialization parity...")
    loaded_model = FailureModel.load(pkl_path)
    p_orig = model.predict_proba(train_df.iloc[:100])
    p_load = loaded_model.predict_proba(train_df.iloc[:100])
    max_load_diff = float(np.max(np.abs(p_orig - p_load)))
    assert max_load_diff <= 1e-9, f"Load parity failed: max_diff {max_load_diff:.2e} > 1e-9"
    logger.info("Serialization parity verified (max diff <= 1e-9).")

    logger.info("Task T7 Complete! Model successfully trained, checked, and saved.")


if __name__ == "__main__":
    main()
