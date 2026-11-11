"""Contributing factors attribution and explanation module.

Implements Decision D61:
- Top 3 model features by absolute per-machine robust z-score, with direction and magnitude.
- Anomaly type labeling based on dominant factor contribution.
- Stretch: SHAP agreement check on sample.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ml.anomaly_detection.baselines import MachineBaseline
from ml.anomaly_detection.masks import scoring_mask

logger = logging.getLogger(__name__)


def compute_contributing_factors(
    baseline: MachineBaseline,
    feature_list: list[str],
    features_df: pd.DataFrame,
    can_score: pd.Series | None = None,
    top_k: int = 3,
) -> pd.DataFrame:
    """Compute top-k contributing features for each scored sensor interval based on robust z-scores.

    Parameters
    ----------
    baseline : MachineBaseline
        Fitted machine baseline normalization.
    feature_list : list[str]
        List of model features.
    features_df : pd.DataFrame
        Sensor features DataFrame.
    can_score : pd.Series | None
        Boolean mask of scoreable intervals.
    top_k : int
        Number of top contributing factors to retain (default 3).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns factor_1, factor_1_z, factor_1_direction, ..., factor_k.
    """
    n = len(features_df)
    if can_score is None:
        can_score, _ = scoring_mask(features_df, feature_list)

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
    z_df = baseline.transform(scored_df)[feature_list]

    z_mat = z_df.to_numpy(dtype=np.float32)
    abs_z_mat = np.abs(z_mat)

    feature_arr = np.array(feature_list)
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

    return res


def check_shap_agreement(
    model: Any,
    baseline: MachineBaseline,
    feature_list: list[str],
    features_df: pd.DataFrame,
    sample_size: int = 2000,
) -> dict[str, Any]:
    """Audit Spearman rank correlation between robust z attribution and TreeExplainer SHAP.

    Note: This is a stretch diagnostic (Section 6.8).
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed; skipping SHAP agreement check.")
        return {"status": "skipped", "reason": "shap not installed"}

    can_score, _ = scoring_mask(features_df, feature_list)
    scored_sub = features_df.loc[can_score]
    if len(scored_sub) == 0:
        return {"status": "skipped", "reason": "no scoreable rows"}

    sample_n = min(sample_size, len(scored_sub))
    sample_df = scored_sub.sample(n=sample_n, random_state=42)

    X_sample = baseline.transform(sample_df)[feature_list].to_numpy(dtype=np.float32)
    abs_z = np.abs(X_sample)

    logger.info("Computing SHAP values on %d samples for agreement check...", sample_n)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    abs_shap = np.abs(shap_values)

    # Compute mean rank correlation across rows
    corrs = []
    for i in range(sample_n):
        r, _ = spearmanr(abs_z[i], abs_shap[i])
        if not np.isnan(r):
            corrs.append(r)

    mean_corr = float(np.mean(corrs)) if corrs else 0.0
    median_corr = float(np.median(corrs)) if corrs else 0.0

    return {
        "status": "success",
        "sample_size": sample_n,
        "mean_spearman_corr": round(mean_corr, 4),
        "median_spearman_corr": round(median_corr, 4),
    }
