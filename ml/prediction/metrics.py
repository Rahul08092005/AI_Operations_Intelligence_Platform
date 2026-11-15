"""Evaluation metrics, calibration diagnostics, and uncertainty intervals.

Implements Decision D71, D77, and Rules R7, R9:
- Precision-Recall AUC (PR-AUC) with empirical prevalence baseline.
- Plan metrics: Accuracy, Precision, Recall, F1, ROC-AUC at operating points.
- Imbalance-aware: Recall at Precision 0.90, Precision at Recall 0.80.
- Calibration quality: Brier score, Log loss, Expected Calibration Error (ECE).
- Diagnostic breakdowns: Per-mode recall, recall ceiling (C2), per-type recall.
- Stratified bootstrap confidence intervals (2,000 resamples).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.prediction.config import MODE_FLAG_COLS, SEED

logger = logging.getLogger(__name__)


def compute_expected_calibration_error(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Compute Expected Calibration Error (ECE) across probability bins.

    Parameters
    ----------
    y_true : 1D array of binary labels {0, 1}.
    y_prob : 1D array of predicted probabilities in [0, 1].
    n_bins : Number of bins (default 10).
    strategy : 'uniform' (equal width) or 'quantile' (equal frequency).

    Returns
    -------
    ece : float
        Weighted average difference between predicted confidence and observed accuracy.
    """
    y_t = np.asarray(y_true, dtype=np.int32)
    y_p = np.asarray(y_prob, dtype=np.float64)
    n = len(y_t)
    if n == 0:
        return 0.0

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.quantile(y_p, quantiles)
        bins[0] = 0.0
        bins[-1] = 1.0 + 1e-8
    else:
        bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)

    bin_indices = np.digitize(y_p, bins) - 1
    ece = 0.0

    for b in range(n_bins):
        mask = bin_indices == b
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_t[mask])
            bin_conf = np.mean(y_p[mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def recall_at_precision(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    target_precision: float = 0.90,
) -> tuple[float, float]:
    """Find maximum recall achievable while maintaining precision >= target_precision.

    Returns
    -------
    (recall, threshold) : tuple[float, float]
    """
    y_t = np.asarray(y_true, dtype=np.int32)
    y_p = np.asarray(y_prob, dtype=np.float64)

    # Sort descending by predicted probability
    desc_score_indices = np.argsort(y_p)[::-1]
    y_sorted = y_t[desc_score_indices]
    prob_sorted = y_p[desc_score_indices]

    n_pos = np.sum(y_t == 1)
    if n_pos == 0:
        return 0.0, 1.0

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    precisions = tps / (tps + fps)
    recalls = tps / n_pos

    # Find points where precision >= target_precision
    valid_idx = np.where(precisions >= target_precision)[0]
    if len(valid_idx) == 0:
        # Cannot attain target precision
        return 0.0, float(prob_sorted[0])

    best_idx = valid_idx[np.argmax(recalls[valid_idx])]
    return float(recalls[best_idx]), float(prob_sorted[best_idx])


def precision_at_recall(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    target_recall: float = 0.80,
) -> tuple[float, float]:
    """Find maximum precision achievable while maintaining recall >= target_recall.

    Returns
    -------
    (precision, threshold) : tuple[float, float]
    """
    y_t = np.asarray(y_true, dtype=np.int32)
    y_p = np.asarray(y_prob, dtype=np.float64)

    desc_score_indices = np.argsort(y_p)[::-1]
    y_sorted = y_t[desc_score_indices]
    prob_sorted = y_p[desc_score_indices]

    n_pos = np.sum(y_t == 1)
    if n_pos == 0:
        return 0.0, 1.0

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    precisions = tps / (tps + fps)
    recalls = tps / n_pos

    valid_idx = np.where(recalls >= target_recall)[0]
    if len(valid_idx) == 0:
        return 0.0, float(prob_sorted[-1])

    best_idx = valid_idx[np.argmax(precisions[valid_idx])]
    return float(precisions[best_idx]), float(prob_sorted[best_idx])


def compute_operating_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Compute full classification metrics at a specific operating threshold."""
    y_t = np.asarray(y_true, dtype=np.int32)
    y_p = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_p >= threshold).astype(np.int32)

    cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Pre-calculated baseline prevalence
    prevalence = float(np.mean(y_t))

    try:
        roc_auc = float(roc_auc_score(y_t, y_p))
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_t, y_p))
    except ValueError:
        pr_auc = prevalence

    brier = float(brier_score_loss(y_t, y_p))
    logloss = float(log_loss(y_t, np.clip(y_p, 1e-15, 1.0 - 1e-15)))
    ece = compute_expected_calibration_error(y_t, y_p, n_bins=10)

    rec_at_90p, th_90p = recall_at_precision(y_t, y_p, target_precision=0.90)
    prec_at_80r, th_80r = precision_at_recall(y_t, y_p, target_recall=0.80)

    return {
        "threshold": float(threshold),
        "prevalence": prevalence,
        "n_total": len(y_t),
        "n_positives": int(np.sum(y_t == 1)),
        "n_negatives": int(np.sum(y_t == 0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_t, y_pred)),
        "precision": float(precision_score(y_t, y_pred, zero_division=0)),
        "recall": float(recall_score(y_t, y_pred, zero_division=0)),
        "f1": float(f1_score(y_t, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "log_loss": logloss,
        "ece": ece,
        "recall_at_precision_90": rec_at_90p,
        "threshold_precision_90": th_90p,
        "precision_at_recall_80": prec_at_80r,
        "threshold_recall_80": th_80r,
    }


class BootstrapResult(NamedTuple):
    """Container for bootstrap estimate and confidence intervals."""

    point_estimate: float
    ci_lower: float
    ci_upper: float


def bootstrap_ci(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = SEED,
) -> BootstrapResult:
    """Compute stratified bootstrap confidence intervals resampled within class."""
    y_t = np.asarray(y_true, dtype=np.int32)
    y_p = np.asarray(y_prob, dtype=np.float64)

    point_est = float(metric_fn(y_t, y_p))

    pos_idx = np.where(y_t == 1)[0]
    neg_idx = np.where(y_t == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        return BootstrapResult(point_est, point_est, point_est)

    rng = np.random.default_rng(seed)
    boot_estimates = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        resampled_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        resampled_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        resampled_idx = np.concatenate([resampled_pos, resampled_neg])

        boot_estimates[i] = metric_fn(y_t[resampled_idx], y_p[resampled_idx])

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_estimates, alpha * 100))
    ci_upper = float(np.percentile(boot_estimates, (1.0 - alpha) * 100))

    return BootstrapResult(point_est, ci_lower, ci_upper)


def compute_recall_ceiling(df_eval: pd.DataFrame) -> dict[str, Any]:
    """Compute the theoretical recall ceiling excluding random failures (Caveat C2).

    Random failures ('rnf') have no physical signature and cannot be predicted.
    The recall ceiling is the fraction of true failures that have at least one
    non-random physical failure mode (twf, hdf, pwf, osf).
    """
    failures = df_eval[df_eval["machine_failure"] == 1].copy()
    n_failures = len(failures)
    if n_failures == 0:
        return {"n_failures": 0, "n_predictable": 0, "recall_ceiling": 1.0}

    non_random_modes = ["twf", "hdf", "pwf", "osf"]
    has_physical_mode = failures[non_random_modes].sum(axis=1) > 0
    n_predictable = int(has_physical_mode.sum())
    n_rnf_only = int(((failures["rnf"] == 1) & (~has_physical_mode)).sum())

    return {
        "n_failures": n_failures,
        "n_predictable": n_predictable,
        "n_rnf_only": n_rnf_only,
        "recall_ceiling": float(n_predictable / n_failures),
    }


def breakdown_by_failure_mode(
    df_eval: pd.DataFrame,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Compute detection recall per failure mode for rows exhibiting that mode."""
    y_p = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_p >= threshold).astype(np.int32)

    rows: list[dict[str, Any]] = []
    for mode in MODE_FLAG_COLS:
        if mode in df_eval.columns:
            mode_mask = df_eval[mode] == 1
            n_mode = int(mode_mask.sum())
            if n_mode > 0:
                rec = float(np.mean(y_pred[mode_mask]))
                mean_p = float(np.mean(y_p[mode_mask]))
            else:
                rec = 0.0
                mean_p = 0.0
            rows.append(
                {
                    "failure_mode": mode,
                    "count": n_mode,
                    "recall": rec,
                    "mean_probability": mean_p,
                    "is_random_noise": bool(mode == "rnf"),
                }
            )

    return pd.DataFrame(rows)


def breakdown_by_machine_type(
    df_eval: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Compute classification metrics broken down by machine variant type (L, M, H)."""
    y_t = np.asarray(y_true, dtype=np.int32)
    y_p = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_p >= threshold).astype(np.int32)

    rows: list[dict[str, Any]] = []
    for m_type in ["L", "M", "H"]:
        mask = df_eval["type"].astype(str).str.upper() == m_type
        n_type = int(mask.sum())
        if n_type > 0:
            yt_sub = y_t[mask]
            yp_sub = y_p[mask]
            pred_sub = y_pred[mask]
            n_fails = int(np.sum(yt_sub == 1))
            prev = float(np.mean(yt_sub))
            rec = float(recall_score(yt_sub, pred_sub, zero_division=0))
            prec = float(precision_score(yt_sub, pred_sub, zero_division=0))
            f1 = float(f1_score(yt_sub, pred_sub, zero_division=0))
            try:
                pr_auc = float(average_precision_score(yt_sub, yp_sub))
            except ValueError:
                pr_auc = prev
        else:
            n_fails = 0
            prev = rec = prec = f1 = pr_auc = 0.0

        rows.append(
            {
                "type": m_type,
                "n_rows": n_type,
                "n_failures": n_fails,
                "prevalence": prev,
                "recall": rec,
                "precision": prec,
                "f1": f1,
                "pr_auc": pr_auc,
            }
        )

    return pd.DataFrame(rows)
