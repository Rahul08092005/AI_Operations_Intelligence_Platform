"""Hyperparameter random search and primary model selection.

Implements Decision D72, D73, and Rule R6:
- 50 pre-registered random draws from XGB_PARAM_GRID.
- Evaluated on Repeated Stratified K-Fold (5x2) on the F1 (physics) feature set.
- Incremental logging to failure_experiments.csv.
- Pre-registered selection rule: max mean out-of-fold PR-AUC with parsimony tie-breaking.
- Ablations: F0 (raw features), scale_pos_weight=1.0, Logistic Regression, Majority baseline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ml.prediction.baselines import LogisticRegressionBaseline, MajorityBaseline
from ml.prediction.config import (
    EXPERIMENTS_LOG_PATH,
    FEATURE_SET_PHYSICS,
    FEATURE_SET_RAW,
    MAX_CONFIGS,
    MAX_SEARCH_DRAWS,
    OOF_PREDICTIONS_PATH,
    SEED,
    XGB_FIXED_PARAMS,
    XGB_PARAM_GRID,
)
from ml.prediction.cv import CVResult, run_cv
from ml.prediction.data import load_train
from ml.prediction.features import build_features

logger = logging.getLogger(__name__)


def compute_scale_pos_weight(y: np.ndarray | pd.Series, mode: str) -> float:
    """Compute scale_pos_weight based on class distribution and selected mode."""
    y_arr = np.asarray(y, dtype=np.int32)
    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))
    if n_pos == 0:
        return 1.0

    ratio = n_neg / n_pos
    if mode == "one":
        return 1.0
    elif mode == "sqrt_ratio":
        return float(np.sqrt(ratio))
    elif mode == "full_ratio":
        return float(ratio)
    else:
        raise ValueError(f"Unknown scale_pos_weight mode: {mode}")


def generate_search_draws(
    n_draws: int = MAX_SEARCH_DRAWS, seed: int = SEED
) -> list[dict[str, Any]]:
    """Draw deterministic hyperparameter configurations from pre-registered ranges."""
    rng = np.random.default_rng(seed)
    configs: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    keys = list(XGB_PARAM_GRID.keys())

    # Generate distinct configurations
    attempts = 0
    max_attempts = n_draws * 20
    while len(configs) < n_draws and attempts < max_attempts:
        attempts += 1
        cfg: dict[str, Any] = {}
        for k in keys:
            choices = XGB_PARAM_GRID[k]
            idx = rng.integers(0, len(choices))
            cfg[k] = choices[idx]

        # Use tuple of sorted items as signature for uniqueness
        sig = tuple(sorted(cfg.items()))
        if sig not in seen:
            seen.add(sig)
            configs.append(cfg)

    if len(configs) < n_draws:
        logger.warning("Could only generate %d unique configs from grid", len(configs))

    return configs


def select_primary(experiments_df: pd.DataFrame) -> pd.Series:
    """Pre-registered selection rule for primary failure model (Section 6.6).

    Rules:
    1. Highest mean pooled out-of-fold PR-AUC (average of repeat 1 and repeat 2).
    2. Ties within 0.002:
       a. Lower Brier score
       b. Fewer trees * depth (smaller model complexity)
       c. Lower n_estimators
    """
    # Filter to XGBoost runs on physics feature set only
    xgb_physics = experiments_df[
        (experiments_df["model_type"] == "xgboost")
        & (experiments_df["feature_set"] == "physics")
    ].copy()

    if xgb_physics.empty:
        raise ValueError("No XGBoost physics configurations found in experiments log.")

    # Sort descending by PR-AUC
    xgb_physics = xgb_physics.sort_values("mean_pr_auc", ascending=False).reset_index(drop=True)
    best_pr_auc = xgb_physics.loc[0, "mean_pr_auc"]

    # Candidate set within 0.002 tolerance
    tie_candidates = xgb_physics[xgb_physics["mean_pr_auc"] >= best_pr_auc - 0.002].copy()

    # Compute model complexity proxy: n_estimators * max_depth
    tie_candidates["complexity"] = tie_candidates["n_estimators"] * tie_candidates["max_depth"]

    # Sort by tie-breaking criteria:
    # higher PR-AUC, lower Brier, lower complexity, lower n_estimators
    tie_candidates = tie_candidates.sort_values(
        by=["mean_pr_auc", "mean_brier", "complexity", "n_estimators"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    selected = tie_candidates.iloc[0]
    logger.info(
        "Selected primary model: %s (PR-AUC=%.4f, ROC-AUC=%.4f, Brier=%.4f, n_est=%d, depth=%d)",
        selected["config_id"],
        selected["mean_pr_auc"],
        selected["mean_roc_auc"],
        selected["mean_brier"],
        selected["n_estimators"],
        selected["max_depth"],
    )
    return selected


def run_experiments(
    out_csv_path: Path | str = EXPERIMENTS_LOG_PATH,
    out_oof_path: Path | str = OOF_PREDICTIONS_PATH,
    n_draws: int = MAX_SEARCH_DRAWS,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Execute hyperparameter random search, ablations, and baseline evaluations.

    Returns
    -------
    experiments_df : pd.DataFrame
        Complete log of evaluated configurations and metrics.
    primary_config : pd.Series
        Row of the selected primary configuration.
    oof_predictions : pd.DataFrame
        Out-of-fold predictions for the selected primary model across both repeats.
    """
    csv_file = Path(out_csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    train_df = load_train()
    y_train = train_df["machine_failure"].to_numpy(dtype=np.int32)
    udis = train_df["udi"].to_numpy()

    # Pre-build feature sets
    X_raw = build_features(train_df, feature_set="raw")
    X_physics = build_features(train_df, feature_set="physics")

    draws = generate_search_draws(n_draws=n_draws, seed=SEED)
    logger.info("Generated %d hyperparameter search configurations", len(draws))

    records: list[dict[str, Any]] = []
    oof_dict: dict[str, CVResult] = {}

    total_runs = 0

    # 1. Evaluate 50 XGBoost configurations on physics feature set
    for i, cfg in enumerate(draws, start=1):
        total_runs += 1
        config_id = f"XGB-PHYS-{i:02d}"
        scale_weight = compute_scale_pos_weight(y_train, cfg["scale_pos_weight_mode"])

        xgb_params = {
            **XGB_FIXED_PARAMS,
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
            "subsample": cfg["subsample"],
            "colsample_bytree": cfg["colsample_bytree"],
            "min_child_weight": cfg["min_child_weight"],
            "reg_lambda": cfg["reg_lambda"],
            "gamma": cfg["gamma"],
            "scale_pos_weight": scale_weight,
        }

        cv_res = run_cv(lambda p=xgb_params: XGBClassifier(**p), X_physics, y_train)
        oof_dict[config_id] = cv_res

        rec = {
            "config_id": config_id,
            "model_type": "xgboost",
            "feature_set": "physics",
            "n_features": len(FEATURE_SET_PHYSICS),
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
            "subsample": cfg["subsample"],
            "colsample_bytree": cfg["colsample_bytree"],
            "min_child_weight": cfg["min_child_weight"],
            "reg_lambda": cfg["reg_lambda"],
            "gamma": cfg["gamma"],
            "scale_pos_weight_mode": cfg["scale_pos_weight_mode"],
            "scale_pos_weight": scale_weight,
            "mean_pr_auc": cv_res.mean_pr_auc,
            "mean_roc_auc": cv_res.mean_roc_auc,
            "mean_brier": cv_res.mean_brier,
            "mean_log_loss": cv_res.mean_log_loss,
            "rep1_pr_auc": cv_res.repeat_metrics[0]["pr_auc"],
            "rep2_pr_auc": cv_res.repeat_metrics[1]["pr_auc"],
        }
        records.append(rec)

        # Incremental write
        pd.DataFrame(records).to_csv(csv_file, index=False)
        logger.info(
            "[%d/%d] %s: PR-AUC=%.4f, ROC-AUC=%.4f, Brier=%.4f",
            i,
            len(draws),
            config_id,
            cv_res.mean_pr_auc,
            cv_res.mean_roc_auc,
            cv_res.mean_brier,
        )

    # Convert to DataFrame to select primary configuration
    exp_df = pd.DataFrame(records)
    primary_config = select_primary(exp_df)
    primary_id = primary_config["config_id"]
    primary_cv = oof_dict[primary_id]

    # Save primary OOF predictions with udi
    primary_oof = primary_cv.oof_df.copy()
    primary_oof["udi"] = udis[primary_oof["row_idx"].to_numpy()]
    if out_oof_path:
        oof_path = Path(out_oof_path)
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        primary_oof.to_parquet(oof_path, index=False)
        logger.info("Saved primary OOF predictions to %s", oof_path)

    # 2. Run Ablation 1: Best configuration on Raw Features (F0)
    best_draw = {
        "n_estimators": int(primary_config["n_estimators"]),
        "max_depth": int(primary_config["max_depth"]),
        "learning_rate": float(primary_config["learning_rate"]),
        "subsample": float(primary_config["subsample"]),
        "colsample_bytree": float(primary_config["colsample_bytree"]),
        "min_child_weight": int(primary_config["min_child_weight"]),
        "reg_lambda": float(primary_config["reg_lambda"]),
        "gamma": float(primary_config["gamma"]),
        "scale_pos_weight": float(primary_config["scale_pos_weight"]),
    }
    ablation_raw_params = {**XGB_FIXED_PARAMS, **best_draw}

    cv_raw = run_cv(lambda p=ablation_raw_params: XGBClassifier(**p), X_raw, y_train)
    total_runs += 1
    rec_raw = {
        "config_id": "ABL-RAW-01",
        "model_type": "xgboost",
        "feature_set": "raw",
        "n_features": len(FEATURE_SET_RAW),
        "n_estimators": best_draw["n_estimators"],
        "max_depth": best_draw["max_depth"],
        "learning_rate": best_draw["learning_rate"],
        "subsample": best_draw["subsample"],
        "colsample_bytree": best_draw["colsample_bytree"],
        "min_child_weight": best_draw["min_child_weight"],
        "reg_lambda": best_draw["reg_lambda"],
        "gamma": best_draw["gamma"],
        "scale_pos_weight_mode": primary_config["scale_pos_weight_mode"],
        "scale_pos_weight": best_draw["scale_pos_weight"],
        "mean_pr_auc": cv_raw.mean_pr_auc,
        "mean_roc_auc": cv_raw.mean_roc_auc,
        "mean_brier": cv_raw.mean_brier,
        "mean_log_loss": cv_raw.mean_log_loss,
        "rep1_pr_auc": cv_raw.repeat_metrics[0]["pr_auc"],
        "rep2_pr_auc": cv_raw.repeat_metrics[1]["pr_auc"],
    }
    records.append(rec_raw)

    # 3. Run Ablation 2: Best configuration with scale_pos_weight = 1.0 (if different)
    if float(primary_config["scale_pos_weight"]) != 1.0:
        ablation_spw1_params = {**XGB_FIXED_PARAMS, **best_draw, "scale_pos_weight": 1.0}
        cv_spw1 = run_cv(lambda p=ablation_spw1_params: XGBClassifier(**p), X_physics, y_train)
        total_runs += 1
        rec_spw1 = {
            "config_id": "ABL-SPW1-01",
            "model_type": "xgboost",
            "feature_set": "physics",
            "n_features": len(FEATURE_SET_PHYSICS),
            "n_estimators": best_draw["n_estimators"],
            "max_depth": best_draw["max_depth"],
            "learning_rate": best_draw["learning_rate"],
            "subsample": best_draw["subsample"],
            "colsample_bytree": best_draw["colsample_bytree"],
            "min_child_weight": best_draw["min_child_weight"],
            "reg_lambda": best_draw["reg_lambda"],
            "gamma": best_draw["gamma"],
            "scale_pos_weight_mode": "one",
            "scale_pos_weight": 1.0,
            "mean_pr_auc": cv_spw1.mean_pr_auc,
            "mean_roc_auc": cv_spw1.mean_roc_auc,
            "mean_brier": cv_spw1.mean_brier,
            "mean_log_loss": cv_spw1.mean_log_loss,
            "rep1_pr_auc": cv_spw1.repeat_metrics[0]["pr_auc"],
            "rep2_pr_auc": cv_spw1.repeat_metrics[1]["pr_auc"],
        }
        records.append(rec_spw1)

    # 4. Run Baseline B1: Logistic Regression on physics features
    cv_lr = run_cv(lambda: LogisticRegressionBaseline(), X_physics, y_train)
    total_runs += 1
    rec_lr = {
        "config_id": "BASE-LOGREG-01",
        "model_type": "logistic_regression",
        "feature_set": "physics",
        "n_features": len(FEATURE_SET_PHYSICS),
        "n_estimators": None,
        "max_depth": None,
        "learning_rate": None,
        "subsample": None,
        "colsample_bytree": None,
        "min_child_weight": None,
        "reg_lambda": None,
        "gamma": None,
        "scale_pos_weight_mode": "one",
        "scale_pos_weight": 1.0,
        "mean_pr_auc": cv_lr.mean_pr_auc,
        "mean_roc_auc": cv_lr.mean_roc_auc,
        "mean_brier": cv_lr.mean_brier,
        "mean_log_loss": cv_lr.mean_log_loss,
        "rep1_pr_auc": cv_lr.repeat_metrics[0]["pr_auc"],
        "rep2_pr_auc": cv_lr.repeat_metrics[1]["pr_auc"],
    }
    records.append(rec_lr)

    # 5. Run Baseline B0: Majority class baseline
    cv_maj = run_cv(lambda: MajorityBaseline(), X_physics, y_train)
    total_runs += 1
    rec_maj = {
        "config_id": "BASE-MAJORITY-01",
        "model_type": "majority",
        "feature_set": "physics",
        "n_features": len(FEATURE_SET_PHYSICS),
        "n_estimators": None,
        "max_depth": None,
        "learning_rate": None,
        "subsample": None,
        "colsample_bytree": None,
        "min_child_weight": None,
        "reg_lambda": None,
        "gamma": None,
        "scale_pos_weight_mode": "one",
        "scale_pos_weight": 1.0,
        "mean_pr_auc": cv_maj.mean_pr_auc,
        "mean_roc_auc": cv_maj.mean_roc_auc,
        "mean_brier": cv_maj.mean_brier,
        "mean_log_loss": cv_maj.mean_log_loss,
        "rep1_pr_auc": cv_maj.repeat_metrics[0]["pr_auc"],
        "rep2_pr_auc": cv_maj.repeat_metrics[1]["pr_auc"],
    }
    records.append(rec_maj)

    assert total_runs <= MAX_CONFIGS, f"Run cap exceeded: {total_runs} > {MAX_CONFIGS}"

    final_df = pd.DataFrame(records)
    final_df.to_csv(csv_file, index=False)
    logger.info("Experiments complete (%d total runs). Results saved to %s", total_runs, csv_file)

    return final_df, primary_config, primary_oof
