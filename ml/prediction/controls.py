"""Negative controls, leakage audits, and baseline comparisons.

Implements Rule R8, Decision D73, and Section 6.8:
- Shuffled labels control (ROC-AUC ~ 0.5, PR-AUC ~ prevalence).
- Univariate feature leakage scan (AUC > 0.99 is a red flag).
- Duplicate check across train and test input spaces.
- Forbidden column absence assertions.
- Physics vs Raw feature ablation gap calculation.
- Prevalence baseline reference.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from ml.prediction.config import FORBIDDEN_COLUMNS, SEED
from ml.prediction.cv import CVResult, run_cv

logger = logging.getLogger(__name__)


def assert_forbidden_columns_absent(df: pd.DataFrame) -> None:
    """Assert that none of the forbidden columns exist in the feature matrix."""
    forbidden = [col for col in df.columns if col.lower() in FORBIDDEN_COLUMNS]
    if forbidden:
        raise AssertionError(
            f"Rule R3 violation: Forbidden leakage columns detected in feature matrix: {forbidden}"
        )


def run_feature_leakage_scan(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Compute univariate ROC-AUC of every feature against machine failure.

    Flags any feature with AUC > 0.99 as a severe leakage suspect.
    """
    y_arr = np.asarray(y, dtype=np.int32)
    records: list[dict[str, Any]] = []

    for col in X.columns:
        vals = X[col].to_numpy(dtype=np.float64)
        if len(np.unique(vals)) <= 1:
            auc = 0.5
        else:
            raw_auc = float(roc_auc_score(y_arr, vals))
            # Direction invariant AUC: max(auc, 1 - auc)
            auc = max(raw_auc, 1.0 - raw_auc)

        is_red_flag = bool(auc >= 0.99)
        records.append(
            {
                "feature": col,
                "univariate_roc_auc": round(auc, 4),
                "is_red_flag": is_red_flag,
            }
        )

    df = pd.DataFrame(records)
    return df.sort_values("univariate_roc_auc", ascending=False).reset_index(drop=True)


def run_shuffled_label_control(
    X: pd.DataFrame,
    y: np.ndarray,
    xgb_params: dict[str, Any],
    seed: int = SEED,
) -> CVResult:
    """Negative control: Evaluate CV on randomly shuffled target labels.

    Expected: ROC-AUC near 0.5, PR-AUC near class prevalence (~0.034).
    """
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y)

    logger.info("Running Shuffled Label Negative Control (seed=%d)...", seed)
    res = run_cv(lambda p=xgb_params: XGBClassifier(**p), X, y_shuffled)
    logger.info(
        "Shuffled Label Result: PR-AUC=%.4f (prevalence=%.4f), ROC-AUC=%.4f, Brier=%.4f",
        res.mean_pr_auc,
        float(np.mean(y)),
        res.mean_roc_auc,
        res.mean_brier,
    )
    return res


def check_input_duplicates(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    input_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Check for duplicate rows across train and test partitions on input feature space."""
    if input_cols is None:
        input_cols = [
            "air_temperature_k",
            "process_temperature_k",
            "rotational_speed_rpm",
            "torque_nm",
            "tool_wear_min",
            "type",
        ]

    # Normalize column names in train and test
    cols_train = [c for c in train_df.columns if c.lower() in [i.lower() for i in input_cols]]
    cols_test = [c for c in test_df.columns if c.lower() in [i.lower() for i in input_cols]]

    df_tr = train_df[cols_train].copy()
    df_te = test_df[cols_test].copy()

    # Exact matches across train and test
    common_exact = pd.merge(df_tr, df_te, how="inner")

    return {
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "n_train_internal_duplicates": int(df_tr.duplicated().sum()),
        "n_test_internal_duplicates": int(df_te.duplicated().sum()),
        "n_exact_cross_partition_matches": len(common_exact),
    }
