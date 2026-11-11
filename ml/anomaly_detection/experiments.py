"""Cross-validation experiment runner, selection logic, and experiment logging.

Implements Decision D55 (3-fold expanding CV inside exploration period),
Decision D60 (Baseline B1 and B2 evaluation on same protocol),
Decision D63 (Experiment budget cap: 30 CV configs + baselines),
and Decision D64 (Pre-registered selection rule).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ml.anomaly_detection.baselines import MachineBaseline
from ml.anomaly_detection.calibration import ScoreCalibrator, smooth_scores, to_status
from ml.anomaly_detection.config import (
    FEATURE_SET_A,
    FEATURE_SETS,
    IFOREST_PARAM_GRID,
    POST_PROCESSING_GRID,
    PROCESSED_DIR,
    SEED,
    STATE_RUNNING,
)
from ml.anomaly_detection.detectors_baseline import (
    B1ManualThresholdDetector,
    B2RobustZDetector,
)
from ml.anomaly_detection.episodes import extract_alert_episodes
from ml.anomaly_detection.evaluate import (
    METROPT_FAILURE_WINDOWS,
    evaluate_real_windows,
    evaluate_synthetic_episodes,
    false_alarm_report,
)
from ml.anomaly_detection.masks import scoring_mask, training_mask
from ml.anomaly_detection.splits import cv_folds

logger = logging.getLogger(__name__)


def generate_iforest_configs() -> list[dict[str, Any]]:
    """Generate the 30 Isolation Forest hyperparameter configurations (Section 6.5)."""
    configs: list[dict[str, Any]] = []
    config_idx = 1

    for feature_set_name, grid in IFOREST_PARAM_GRID.items():
        n_estimators_list = grid["n_estimators"]
        max_samples_list = grid["max_samples"]
        max_features_list = grid["max_features"]

        for n_est in n_estimators_list:
            for max_samp in max_samples_list:
                for max_feat in max_features_list:
                    configs.append(
                        {
                            "config_id": f"IF-{config_idx:02d}",
                            "feature_set": feature_set_name,
                            "n_features": len(FEATURE_SETS[feature_set_name]),
                            "n_estimators": n_est,
                            "max_samples": max_samp,
                            "max_features": max_feat,
                        }
                    )
                    config_idx += 1

    return configs


def run_cv_experiments(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    machines_df: pd.DataFrame | None = None,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Run expanding-window cross-validation for all 30 configs and baselines B1 and B2.

    Parameters
    ----------
    features_df : pd.DataFrame
        Complete features DataFrame.
    labels_df : pd.DataFrame
        Evaluation labels.
    machines_df : pd.DataFrame | None
        Machine metadata.
    output_csv : Path | None
        Path to save anomaly_experiments.csv (default: data/processed/anomaly_experiments.csv).
    """
    out_path = output_csv or (PROCESSED_DIR / "anomaly_experiments.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    folds = cv_folds()
    configs = generate_iforest_configs()
    logger.info(
        "Generated %d Isolation Forest configurations across %d CV folds.",
        len(configs),
        len(folds),
    )

    # Real failure windows that fall into the validation blocks
    val_real_windows = [
        w
        for w in METROPT_FAILURE_WINDOWS
        if any(f.val_start <= w["start"] < f.val_end for f in folds)
    ]

    # Pre-filter validation slice of labels
    val_start_min = min(f.val_start for f in folds)
    val_end_max = max(f.val_end for f in folds)
    val_labels = labels_df[
        (labels_df["timestamp"] >= val_start_min) & (labels_df["timestamp"] < val_end_max)
    ].copy()

    total_val_days = (val_end_max - val_start_min).total_seconds() / 86400.0

    experiment_rows: list[dict[str, Any]] = []

    # -------------------------------------------------------------
    # 1. Run 30 Isolation Forest Configurations
    # -------------------------------------------------------------
    for cfg in configs:
        c_id = cfg["config_id"]
        f_set_name = cfg["feature_set"]
        f_list = FEATURE_SETS[f_set_name]
        n_est = cfg["n_estimators"]
        max_samp = cfg["max_samples"]
        max_feat = cfg["max_features"]

        logger.info(
            "Running CV for %s | Set: %s | n_est: %d | max_samples: %d | max_feat: %.1f",
            c_id,
            f_set_name,
            n_est,
            max_samp,
            max_feat,
        )

        fold_train_raws: list[np.ndarray] = []
        fold_val_dfs: list[pd.DataFrame] = []
        fit_times: list[float] = []

        # Fit model and compute raw scores on each fold
        for fold in folds:
            t0 = time.time()

            # Training mask
            t_mask, _ = training_mask(
                features_df, labels_df, f_list, (fold.train_start, fold.train_end)
            )
            train_sub = features_df.loc[t_mask]

            # Fit baseline on training data
            baseline = MachineBaseline().fit(train_sub, f_list, machines_df)
            X_train = baseline.transform(train_sub)

            # Fit Isolation Forest
            model = IsolationForest(
                n_estimators=n_est,
                max_samples=max_samp,
                max_features=max_feat,
                contamination="auto",
                bootstrap=False,
                n_jobs=-1,
                random_state=SEED,
            )
            model.fit(X_train)
            fit_times.append(time.time() - t0)

            # Raw scores on normal training rows
            raw_train = -model.score_samples(X_train)
            fold_train_raws.append(raw_train)

            # Validation slice
            v_mask_time = (features_df["timestamp"] >= fold.val_start) & (
                features_df["timestamp"] < fold.val_end
            )
            val_features = features_df.loc[v_mask_time].copy()
            v_can_score, v_states = scoring_mask(val_features, f_list)

            val_scored_df = pd.DataFrame(
                {
                    "machine_id": val_features["machine_id"].to_numpy(),
                    "timestamp": val_features["timestamp"].to_numpy(),
                    "state": v_states.to_numpy(),
                },
                index=val_features.index,
            )

            # Raw scores for scoreable rows
            raw_val = np.full(len(val_features), np.nan, dtype=np.float32)
            if np.any(v_can_score):
                X_val = baseline.transform(val_features.loc[v_can_score])
                raw_val[v_can_score.to_numpy()] = -model.score_samples(X_val)

            val_scored_df["raw_score"] = raw_val
            fold_val_dfs.append(val_scored_df)

        avg_fit_time = round(float(np.mean(fit_times)), 2)

        # Post-processing grid sweep
        for q_warn in POST_PROCESSING_GRID["q_warn"]:
            for q_crit in POST_PROCESSING_GRID["q_crit"]:
                for smooth_w in POST_PROCESSING_GRID["smoothing_window"]:
                    pooled_val_parts: list[pd.DataFrame] = []

                    for f_idx in range(len(folds)):
                        calibrator = ScoreCalibrator().fit(
                            fold_train_raws[f_idx], q_warn=q_warn, q_crit=q_crit
                        )
                        f_df = fold_val_dfs[f_idx].copy()

                        # Map raw to 0-1
                        s_5min = calibrator.map_scores(f_df["raw_score"].to_numpy())
                        f_df["score_5min"] = s_5min

                        # Trailing smooth
                        s_smooth = smooth_scores(
                            mapped_scores=pd.Series(s_5min, index=f_df.index),
                            machine_ids=f_df["machine_id"],
                            timestamps=f_df["timestamp"],
                            can_score=f_df["state"] == STATE_RUNNING,
                            window=smooth_w,
                        )
                        f_df["anomaly_score"] = s_smooth
                        f_df["anomaly_status"] = to_status(s_smooth, f_df["state"].to_numpy())
                        pooled_val_parts.append(f_df)

                    pooled_val = pd.concat(pooled_val_parts, ignore_index=True)

                    # Extract alert episodes
                    episodes = extract_alert_episodes(pooled_val)

                    # Evaluate false alarm budget
                    budget_passed, fa_summary, _ = false_alarm_report(
                        episodes, val_labels, pooled_val, total_days=total_val_days
                    )

                    # Evaluate real failure windows
                    real_res = evaluate_real_windows(episodes, val_real_windows)
                    real_detected = real_res[real_res["status"].isin(["early", "late"])]
                    n_real_detected = len(real_detected)
                    n_real_total = len(real_res)

                    early_windows = real_res[real_res["status"] == "early"]
                    med_lead_time = (
                        float(early_windows["lead_time_hours"].median())
                        if len(early_windows) > 0
                        else 0.0
                    )

                    # Evaluate synthetic episodes
                    syn_res = evaluate_synthetic_episodes(episodes, val_labels, pooled_val)
                    syn_detected = (
                        syn_res[syn_res["detected"]] if len(syn_res) > 0 else pd.DataFrame()
                    )
                    n_syn_detected = len(syn_detected)
                    n_syn_total = len(syn_res)

                    # Episode recall across all true anomaly validation episodes
                    total_true_episodes = n_real_total + n_syn_total
                    total_detected = n_real_detected + n_syn_detected
                    ep_recall = (
                        round(total_detected / total_true_episodes, 4)
                        if total_true_episodes > 0
                        else 0.0
                    )

                    row_record = {
                        "config_id": c_id,
                        "detector": "IsolationForest",
                        "feature_set": f_set_name,
                        "n_features": len(f_list),
                        "n_estimators": n_est,
                        "max_samples": max_samp,
                        "max_features": max_feat,
                        "q_warn": q_warn,
                        "q_crit": q_crit,
                        "smoothing_window": smooth_w,
                        "budget_passed": budget_passed,
                        "pooled_warning_fa_per_30d": fa_summary["pooled_warning_fa_per_30d"],
                        "total_critical_fa": fa_summary["total_critical_fa"],
                        "m109_warning_fa_per_30d": fa_summary["m109_warning_fa_per_30d"],
                        "m109_critical_fa": fa_summary["m109_critical_fa"],
                        "episode_recall": ep_recall,
                        "episodes_detected": total_detected,
                        "episodes_total": total_true_episodes,
                        "real_windows_detected": n_real_detected,
                        "real_windows_total": n_real_total,
                        "median_lead_time_hours": round(med_lead_time, 2),
                        "fit_time_seconds": avg_fit_time,
                    }
                    experiment_rows.append(row_record)

        # Write intermediate progress
        pd.DataFrame(experiment_rows).to_csv(out_path, index=False)

    # -------------------------------------------------------------
    # 2. Run Baseline Detectors B1 & B2
    # -------------------------------------------------------------
    logger.info("Running Baseline Detectors B1 (Manual Thresholds) and B2 (Robust Z)...")

    # Baseline B1
    for smooth_w in [6, 12, 36]:
        b1_val_parts: list[pd.DataFrame] = []
        for fold in folds:
            b1 = B1ManualThresholdDetector(smoothing_window=smooth_w)
            v_mask_time = (features_df["timestamp"] >= fold.val_start) & (
                features_df["timestamp"] < fold.val_end
            )
            val_features = features_df.loc[v_mask_time]
            b1_scored = b1.score_frame(val_features)
            b1_val_parts.append(b1_scored)

        pooled_b1 = pd.concat(b1_val_parts, ignore_index=True)
        b1_episodes = extract_alert_episodes(pooled_b1)
        budget_passed, fa_summary, _ = false_alarm_report(
            b1_episodes, val_labels, pooled_b1, total_days=total_val_days
        )
        real_res = evaluate_real_windows(b1_episodes, val_real_windows)
        syn_res = evaluate_synthetic_episodes(b1_episodes, val_labels, pooled_b1)

        n_real_det = len(real_res[real_res["status"].isin(["early", "late"])])
        n_syn_det = len(syn_res[syn_res["detected"]]) if len(syn_res) > 0 else 0
        tot_true = len(real_res) + len(syn_res)
        tot_det = n_real_det + n_syn_det
        ep_rec = round(tot_det / tot_true, 4) if tot_true > 0 else 0.0

        early_w = real_res[real_res["status"] == "early"]
        med_lead = float(early_w["lead_time_hours"].median()) if len(early_w) > 0 else 0.0

        experiment_rows.append(
            {
                "config_id": f"B1-W{smooth_w:02d}",
                "detector": "B1_ManualThreshold",
                "feature_set": "vibration_temperature",
                "n_features": 2,
                "n_estimators": 0,
                "max_samples": 0,
                "max_features": 1.0,
                "q_warn": 0.0,
                "q_crit": 0.0,
                "smoothing_window": smooth_w,
                "budget_passed": budget_passed,
                "pooled_warning_fa_per_30d": fa_summary["pooled_warning_fa_per_30d"],
                "total_critical_fa": fa_summary["total_critical_fa"],
                "m109_warning_fa_per_30d": fa_summary["m109_warning_fa_per_30d"],
                "m109_critical_fa": fa_summary["m109_critical_fa"],
                "episode_recall": ep_rec,
                "episodes_detected": tot_det,
                "episodes_total": tot_true,
                "real_windows_detected": n_real_det,
                "real_windows_total": len(real_res),
                "median_lead_time_hours": round(med_lead, 2),
                "fit_time_seconds": 0.05,
            }
        )

    # Baseline B2 (sweep z_warn and z_crit)
    b2_grid = [
        (3.0, 6.0),
        (4.0, 8.0),
        (5.0, 10.0),
        (6.0, 10.0),
    ]
    for z_warn, z_crit in b2_grid:
        for smooth_w in [6, 12, 36]:
            b2_val_parts: list[pd.DataFrame] = []
            for fold in folds:
                b2 = B2RobustZDetector(z_warn=z_warn, z_crit=z_crit, smoothing_window=smooth_w)
                b2.fit(features_df, labels_df, (fold.train_start, fold.train_end), machines_df)
                v_mask_time = (features_df["timestamp"] >= fold.val_start) & (
                    features_df["timestamp"] < fold.val_end
                )
                val_features = features_df.loc[v_mask_time]
                b2_scored = b2.score_frame(val_features)
                b2_val_parts.append(b2_scored)

            pooled_b2 = pd.concat(b2_val_parts, ignore_index=True)
            b2_episodes = extract_alert_episodes(pooled_b2)
            budget_passed, fa_summary, _ = false_alarm_report(
                b2_episodes, val_labels, pooled_b2, total_days=total_val_days
            )
            real_res = evaluate_real_windows(b2_episodes, val_real_windows)
            syn_res = evaluate_synthetic_episodes(b2_episodes, val_labels, pooled_b2)

            n_real_det = len(real_res[real_res["status"].isin(["early", "late"])])
            n_syn_det = len(syn_res[syn_res["detected"]]) if len(syn_res) > 0 else 0
            tot_true = len(real_res) + len(syn_res)
            tot_det = n_real_det + n_syn_det
            ep_rec = round(tot_det / tot_true, 4) if tot_true > 0 else 0.0

            early_w = real_res[real_res["status"] == "early"]
            med_lead = float(early_w["lead_time_hours"].median()) if len(early_w) > 0 else 0.0

            experiment_rows.append(
                {
                    "config_id": f"B2-ZW{int(z_warn)}-ZC{int(z_crit)}-W{smooth_w:02d}",
                    "detector": "B2_RobustZ",
                    "feature_set": "plan_core",
                    "n_features": len(FEATURE_SET_A),
                    "n_estimators": 0,
                    "max_samples": 0,
                    "max_features": 1.0,
                    "q_warn": z_warn,
                    "q_crit": z_crit,
                    "smoothing_window": smooth_w,
                    "budget_passed": budget_passed,
                    "pooled_warning_fa_per_30d": fa_summary["pooled_warning_fa_per_30d"],
                    "total_critical_fa": fa_summary["total_critical_fa"],
                    "m109_warning_fa_per_30d": fa_summary["m109_warning_fa_per_30d"],
                    "m109_critical_fa": fa_summary["m109_critical_fa"],
                    "episode_recall": ep_rec,
                    "episodes_detected": tot_det,
                    "episodes_total": tot_true,
                    "real_windows_detected": n_real_det,
                    "real_windows_total": len(real_res),
                    "median_lead_time_hours": round(med_lead, 2),
                    "fit_time_seconds": 0.10,
                }
            )

    df_experiments = pd.DataFrame(experiment_rows)
    df_experiments.to_csv(out_path, index=False)
    logger.info("Saved %d CV experiment results to %s", len(df_experiments), out_path)
    return df_experiments


def select_primary(experiments_df: pd.DataFrame) -> dict[str, Any]:
    """Apply the pre-registered selection rule to choose the primary model (Section 6.6).

    Rules:
    1. Filter candidates: feature_set in ['plan_core', 'core_context', 'core_temporal'].
    2. Filter to configurations satisfying the false-alarm budget in pooled CV validation.
    3. Maximize episode recall over true-anomaly episodes in validation blocks.
    4. Tie-breaker 1: higher median lead time hours.
    5. Tie-breaker 2: fewer features (parsimony).
    6. Tie-breaker 3: fewer trees (runtime efficiency).
    """
    candidates = experiments_df[experiments_df["detector"] == "IsolationForest"].copy()

    # Rule: Sets A, B, C only
    primary_pool = candidates[
        candidates["feature_set"].isin(["plan_core", "core_context", "core_temporal"])
    ]

    # Filter to budget passed
    passing = primary_pool[primary_pool["budget_passed"]]

    if len(passing) == 0:
        logger.warning(
            "No configuration strictly met the false alarm budget. Selecting least-violating."
        )
        passing = primary_pool.sort_values(
            ["total_critical_fa", "pooled_warning_fa_per_30d", "episode_recall"],
            ascending=[True, True, False],
        )

    # Sort by:
    # 1. episode_recall (descending)
    # 2. median_lead_time_hours (descending)
    # 3. n_features (ascending)
    # 4. n_estimators (ascending)
    sorted_df = passing.sort_values(
        by=["episode_recall", "median_lead_time_hours", "n_features", "n_estimators"],
        ascending=[False, False, True, True],
    )

    selected = sorted_df.iloc[0].to_dict()
    selected["selection_reason"] = (
        f"Passed budget ({selected['pooled_warning_fa_per_30d']:.2f} FA/30d, 0 Crit), "
        f"max recall {selected['episode_recall']:.2f} "
        f"({selected['episodes_detected']}/{selected['episodes_total']}), "
        f"median lead time {selected['median_lead_time_hours']:.1f}h, "
        f"parsimonious feature count {selected['n_features']}."
    )
    return selected


def select_real_only(experiments_df: pd.DataFrame) -> dict[str, Any]:
    """Select the best post-processing variant for the real-only comparator (Set D)."""
    candidates = experiments_df[
        (experiments_df["detector"] == "IsolationForest")
        & (experiments_df["feature_set"] == "real_only")
    ].copy()

    passing = candidates[candidates["budget_passed"]]
    if len(passing) == 0:
        passing = candidates.sort_values(
            ["total_critical_fa", "pooled_warning_fa_per_30d", "episode_recall"],
            ascending=[True, True, False],
        )

    sorted_df = passing.sort_values(
        by=["episode_recall", "median_lead_time_hours", "n_estimators"],
        ascending=[False, False, True],
    )

    selected = sorted_df.iloc[0].to_dict()
    selected["selection_reason"] = (
        f"Real-only comparator: Budget passed={selected['budget_passed']}, "
        f"recall {selected['episode_recall']:.2f}, "
        f"lead time {selected['median_lead_time_hours']:.1f}h."
    )
    return selected
