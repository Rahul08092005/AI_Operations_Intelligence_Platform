"""Cross-validation framework with Repeated Stratified K-Fold.

Implements Decision D71:
- 5 folds x 2 repeats (10 total splits per configuration).
- Prevents leakage: all transformers and scalers fitted inside training fold.
- Selection metric: Mean out-of-fold PR-AUC (average of repeat 1 and repeat 2).
- Returns pooled out-of-fold prediction frames.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from ml.prediction.config import CV_N_REPEATS, CV_N_SPLITS, SEED

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    """Stores cross-validation evaluation results and out-of-fold predictions."""

    mean_pr_auc: float
    mean_roc_auc: float
    mean_brier: float
    mean_log_loss: float
    repeat_metrics: list[dict[str, float]] = field(default_factory=list)
    oof_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)


def run_cv(
    model_factory: Callable[[], Any],
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_splits: int = CV_N_SPLITS,
    n_repeats: int = CV_N_REPEATS,
    seed: int = SEED,
) -> CVResult:
    """Execute repeated stratified cross-validation on the training partition.

    Parameters
    ----------
    model_factory : Callable[[], Any]
        Function returning a new, un-fitted estimator implementing fit and predict_proba.
    X : Feature matrix.
    y : Target labels (binary 0/1).
    n_splits : Number of folds per repeat (default 5).
    n_repeats : Number of CV repeats (default 2).
    seed : Random state seed for splits.

    Returns
    -------
    CVResult
        Summary metrics and out-of-fold predictions.
    """
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_arr = np.asarray(y, dtype=np.int32)

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )

    oof_records: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []

    for split_idx, (train_idx, val_idx) in enumerate(rskf.split(X_df, y_arr)):
        repeat_idx = split_idx // n_splits + 1  # 1-indexed repeat
        fold_idx = split_idx % n_splits + 1      # 1-indexed fold

        X_train, y_train = X_df.iloc[train_idx], y_arr[train_idx]
        X_val, y_val = X_df.iloc[val_idx], y_arr[val_idx]

        model = model_factory()
        model.fit(X_train, y_train)

        # Predict probability of positive class (column 1)
        probs_val = model.predict_proba(X_val)[:, 1]

        # Fold metrics
        try:
            fold_pr_auc = float(average_precision_score(y_val, probs_val))
        except ValueError:
            fold_pr_auc = float(np.mean(y_val))

        try:
            fold_roc_auc = float(roc_auc_score(y_val, probs_val))
        except ValueError:
            fold_roc_auc = 0.5

        fold_brier = float(brier_score_loss(y_val, probs_val))
        fold_loss = float(log_loss(y_val, np.clip(probs_val, 1e-15, 1.0 - 1e-15)))

        fold_metrics.append(
            {
                "repeat": repeat_idx,
                "fold": fold_idx,
                "pr_auc": fold_pr_auc,
                "roc_auc": fold_roc_auc,
                "brier_score": fold_brier,
                "log_loss": fold_loss,
            }
        )

        for orig_idx, true_val, pred_prob in zip(val_idx, y_val, probs_val, strict=False):
            oof_records.append(
                {
                    "row_idx": int(orig_idx),
                    "repeat": int(repeat_idx),
                    "fold": int(fold_idx),
                    "y_true": int(true_val),
                    "y_prob": float(pred_prob),
                }
            )

    oof_df = pd.DataFrame(oof_records)

    # Compute pooled metrics per repeat
    repeat_summaries: list[dict[str, float]] = []
    for rep in range(1, n_repeats + 1):
        rep_sub = oof_df[oof_df["repeat"] == rep].sort_values("row_idx")
        rep_y = rep_sub["y_true"].to_numpy()
        rep_p = rep_sub["y_prob"].to_numpy()

        try:
            pr_auc_rep = float(average_precision_score(rep_y, rep_p))
        except ValueError:
            pr_auc_rep = float(np.mean(rep_y))

        try:
            roc_auc_rep = float(roc_auc_score(rep_y, rep_p))
        except ValueError:
            roc_auc_rep = 0.5

        brier_rep = float(brier_score_loss(rep_y, rep_p))
        logloss_rep = float(log_loss(rep_y, np.clip(rep_p, 1e-15, 1.0 - 1e-15)))

        repeat_summaries.append(
            {
                "repeat": rep,
                "pr_auc": pr_auc_rep,
                "roc_auc": roc_auc_rep,
                "brier_score": brier_rep,
                "log_loss": logloss_rep,
            }
        )

    mean_pr_auc = float(np.mean([r["pr_auc"] for r in repeat_summaries]))
    mean_roc_auc = float(np.mean([r["roc_auc"] for r in repeat_summaries]))
    mean_brier = float(np.mean([r["brier_score"] for r in repeat_summaries]))
    mean_log_loss = float(np.mean([r["log_loss"] for r in repeat_summaries]))

    return CVResult(
        mean_pr_auc=mean_pr_auc,
        mean_roc_auc=mean_roc_auc,
        mean_brier=mean_brier,
        mean_log_loss=mean_log_loss,
        repeat_metrics=repeat_summaries,
        oof_df=oof_df,
        fold_metrics=fold_metrics,
    )
