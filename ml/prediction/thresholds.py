"""Risk threshold derivation and category assignment.

Implements Decision D75 and Section 6.6:
- t_med: highest probability threshold where recall >= 0.95
- t_high: threshold maximizing F1 score
- t_crit: lowest probability threshold where precision >= 0.90
- Monotonic enforcement: t_med <= t_high <= t_crit
- Risk categories: Low, Medium, High, Critical
- Category reliability reporting: row counts, share, empirical failure rates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from ml.prediction.config import (
    RISK_CATEGORY_ORDER,
    TARGET_PRECISION_CRIT,
    TARGET_RECALL_MED,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskThresholds:
    """Operating risk thresholds and metadata."""

    t_med: float
    t_high: float
    t_crit: float
    med_recall: float
    high_f1: float
    high_precision: float
    high_recall: float
    crit_precision: float
    monotonicity_adjusted: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_med": round(self.t_med, 4),
            "t_high": round(self.t_high, 4),
            "t_crit": round(self.t_crit, 4),
            "med_recall": round(self.med_recall, 4),
            "high_f1": round(self.high_f1, 4),
            "high_precision": round(self.high_precision, 4),
            "high_recall": round(self.high_recall, 4),
            "crit_precision": round(self.crit_precision, 4),
            "monotonicity_adjusted": self.monotonicity_adjusted,
            "notes": self.notes,
        }


def compute_thresholds(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    target_recall_med: float = TARGET_RECALL_MED,
    target_precision_crit: float = TARGET_PRECISION_CRIT,
) -> RiskThresholds:
    """Derive operating risk thresholds from out-of-fold calibrated probabilities.

    Parameters
    ----------
    y_true : array-like
        True binary target values.
    y_prob : array-like
        Calibrated predicted probabilities.
    target_recall_med : float
        Target recall for t_med (default: 0.95).
    target_precision_crit : float
        Target precision for t_crit (default: 0.90).

    Returns
    -------
    RiskThresholds
        Container with t_med, t_high, t_crit and performance diagnostic at each.
    """
    y_arr = np.asarray(y_true, dtype=np.int32)
    p_arr = np.asarray(y_prob, dtype=np.float64)

    precisions, recalls, thresholds = precision_recall_curve(y_arr, p_arr)
    # precision_recall_curve returns thresholds of length len(precisions)-1
    # precisions[-1] is 1.0, recalls[-1] is 0.0 without corresponding threshold
    p_thresh = precisions[:-1]
    r_thresh = recalls[:-1]

    # Compute F1 for each threshold
    denom = p_thresh + r_thresh
    f1_scores = np.where(denom > 0, 2 * (p_thresh * r_thresh) / denom, 0.0)

    # 1. t_med: highest probability threshold where recall >= target_recall_med
    med_candidates_mask = r_thresh >= target_recall_med
    if np.any(med_candidates_mask):
        # Among all thresholds with recall >= target, take the highest threshold
        best_med_idx = np.where(med_candidates_mask)[0][-1]
        t_med = float(thresholds[best_med_idx])
        med_recall = float(r_thresh[best_med_idx])
    else:
        # Fallback to lowest threshold
        t_med = float(thresholds[0]) if len(thresholds) > 0 else 0.0
        med_recall = float(r_thresh[0]) if len(r_thresh) > 0 else 1.0

    # 2. t_high: threshold maximizing F1
    best_f1_idx = int(np.argmax(f1_scores))
    t_high = float(thresholds[best_f1_idx])
    high_f1 = float(f1_scores[best_f1_idx])
    high_precision = float(p_thresh[best_f1_idx])
    high_recall = float(r_thresh[best_f1_idx])

    # 3. t_crit: lowest threshold where precision >= target_precision_crit
    crit_candidates_mask = p_thresh >= target_precision_crit
    notes_list: list[str] = []
    if np.any(crit_candidates_mask):
        best_crit_idx = np.where(crit_candidates_mask)[0][0]
        t_crit = float(thresholds[best_crit_idx])
        crit_precision = float(p_thresh[best_crit_idx])
    else:
        # Fallback to threshold that achieves maximum precision overall
        best_crit_idx = int(np.argmax(p_thresh))
        t_crit = float(thresholds[best_crit_idx])
        crit_precision = float(p_thresh[best_crit_idx])
        notes_list.append(
            f"Target precision {target_precision_crit} unattainable; used max achievable "
            f"precision {crit_precision:.4f} at threshold {t_crit:.4f}."
        )

    # Enforce monotonicity: t_med <= t_high <= t_crit
    monotonicity_adjusted = False
    orig_high, orig_crit = t_high, t_crit
    if t_high < t_med:
        monotonicity_adjusted = True
        t_high = t_med
        notes_list.append(
            f"Monotonic adjustment: t_high raised from {orig_high:.4f} to t_med ({t_med:.4f})."
        )

    if t_crit < t_high:
        monotonicity_adjusted = True
        t_crit = t_high
        notes_list.append(
            f"Monotonic adjustment: t_crit raised from {orig_crit:.4f} to t_high ({t_high:.4f})."
        )

    notes = (
        " ".join(notes_list)
        if notes_list
        else "Derived thresholds strictly satisfy monotonicity."
    )

    logger.info(
        "Thresholds: t_med=%.4f (R=%.4f), t_high=%.4f (F1=%.4f), t_crit=%.4f (P=%.4f)",
        t_med,
        med_recall,
        t_high,
        high_f1,
        t_crit,
        crit_precision,
    )

    return RiskThresholds(
        t_med=t_med,
        t_high=t_high,
        t_crit=t_crit,
        med_recall=med_recall,
        high_f1=high_f1,
        high_precision=high_precision,
        high_recall=high_recall,
        crit_precision=crit_precision,
        monotonicity_adjusted=monotonicity_adjusted,
        notes=notes,
    )


def assign_risk_category(
    probs: np.ndarray | pd.Series,
    thresholds: RiskThresholds | dict[str, float],
) -> np.ndarray:
    """Assign categorical risk level from continuous calibrated probabilities.

    Categories:
    - Low: prob < t_med
    - Medium: t_med <= prob < t_high
    - High: t_high <= prob < t_crit
    - Critical: prob >= t_crit
    """
    p_arr = np.asarray(probs, dtype=np.float64)
    if isinstance(thresholds, RiskThresholds):
        t_med = thresholds.t_med
        t_high = thresholds.t_high
        t_crit = thresholds.t_crit
    else:
        t_med = float(thresholds["t_med"])
        t_high = float(thresholds["t_high"])
        t_crit = float(thresholds["t_crit"])

    categories = np.empty(len(p_arr), dtype=object)
    categories[p_arr < t_med] = "Low"
    categories[(p_arr >= t_med) & (p_arr < t_high)] = "Medium"
    categories[(p_arr >= t_high) & (p_arr < t_crit)] = "High"
    categories[p_arr >= t_crit] = "Critical"

    return categories


def compute_category_reliability(
    y_true: np.ndarray | pd.Series,
    probs: np.ndarray | pd.Series,
    thresholds: RiskThresholds | dict[str, float],
) -> pd.DataFrame:
    """Calculate empirical failure rates and sample share for each risk category."""
    y_arr = np.asarray(y_true, dtype=np.int32)
    p_arr = np.asarray(probs, dtype=np.float64)
    cats = assign_risk_category(p_arr, thresholds)

    total_rows = len(y_arr)
    rows: list[dict[str, Any]] = []

    if isinstance(thresholds, RiskThresholds):
        t_med, t_high, t_crit = thresholds.t_med, thresholds.t_high, thresholds.t_crit
    else:
        t_med, t_high, t_crit = (
            float(thresholds["t_med"]),
            float(thresholds["t_high"]),
            float(thresholds["t_crit"]),
        )

    tier_ranges = {
        "Low": f"[0.0000, {t_med:.4f})",
        "Medium": f"[{t_med:.4f}, {t_high:.4f})",
        "High": f"[{t_high:.4f}, {t_crit:.4f})",
        "Critical": f"[{t_crit:.4f}, 1.0000]",
    }

    for cat in RISK_CATEGORY_ORDER:
        mask = cats == cat
        n_in_cat = int(np.sum(mask))
        n_pos = int(np.sum(y_arr[mask] == 1))
        share = float(n_in_cat / total_rows) if total_rows > 0 else 0.0
        obs_rate = float(n_pos / n_in_cat) if n_in_cat > 0 else 0.0

        rows.append(
            {
                "risk_category": cat,
                "probability_range": tier_ranges[cat],
                "n_rows": n_in_cat,
                "share_pct": round(share * 100.0, 2),
                "n_failures": n_pos,
                "observed_failure_rate": round(obs_rate, 4),
            }
        )

    return pd.DataFrame(rows)
