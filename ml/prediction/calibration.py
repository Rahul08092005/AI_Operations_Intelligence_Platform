"""Probability calibration and calibrator selection.

Implements Decision D74 and Rule R5:
- Nested cross-fitting comparing Platt scaling (sigmoid) vs Isotonic regression on OOF predictions.
- Decision rule: Select isotonic ONLY if nested Brier score improves by >= 5% relative to sigmoid
  AND expected calibration error (ECE) does not worsen. Otherwise select sigmoid.
- Final chosen calibrator is fitted on all out-of-fold predictions from training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ml.prediction.metrics import compute_expected_calibration_error

logger = logging.getLogger(__name__)


class SigmoidCalibrator:
    """Platt scaling calibrator wrapping LogisticRegression on raw probabilities."""

    def __init__(self) -> None:
        self.lr = LogisticRegression(solver="lbfgs", C=1.0, random_state=42)

    def fit(self, probs: np.ndarray, y: np.ndarray) -> SigmoidCalibrator:
        X = np.asarray(probs, dtype=np.float64).reshape(-1, 1)
        y_arr = np.asarray(y, dtype=np.int32)
        self.lr.fit(X, y_arr)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        X = np.asarray(probs, dtype=np.float64).reshape(-1, 1)
        return self.lr.predict_proba(X)[:, 1]


class IsotonicCalibrator:
    """Isotonic regression calibrator with clipping to [0, 1]."""

    def __init__(self) -> None:
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, probs: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        X = np.asarray(probs, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        self.iso.fit(X, y_arr)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        X = np.asarray(probs, dtype=np.float64)
        return np.clip(self.iso.predict(X), 0.0, 1.0)


@dataclass(frozen=True)
class CalibratorComparison:
    """Diagnostic comparison of sigmoid vs isotonic calibrators under nested CV."""

    chosen_method: Literal["sigmoid", "isotonic"]
    sigmoid_brier: float
    sigmoid_ece: float
    isotonic_brier: float
    isotonic_ece: float
    brier_improvement_pct: float
    reason: str


def evaluate_nested_calibrators(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    fold_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform 5-fold nested cross-fitting for sigmoid and isotonic calibrators.

    Parameters
    ----------
    oof_probs : np.ndarray
        Raw out-of-fold predicted probabilities.
    y_true : np.ndarray
        True binary target values.
    fold_indices : np.ndarray
        Fold index (0 to 4) for each sample.

    Returns
    -------
    sigmoid_preds : np.ndarray
        Nested out-of-fold calibrated probabilities from sigmoid.
    isotonic_preds : np.ndarray
        Nested out-of-fold calibrated probabilities from isotonic.
    """
    unique_folds = np.unique(fold_indices)
    sigmoid_nested = np.zeros_like(oof_probs, dtype=np.float64)
    isotonic_nested = np.zeros_like(oof_probs, dtype=np.float64)

    for fold in unique_folds:
        val_mask = fold_indices == fold
        train_mask = ~val_mask

        # Sigmoid
        sig = SigmoidCalibrator().fit(oof_probs[train_mask], y_true[train_mask])
        sigmoid_nested[val_mask] = sig.predict_proba(oof_probs[val_mask])

        # Isotonic
        iso = IsotonicCalibrator().fit(oof_probs[train_mask], y_true[train_mask])
        isotonic_nested[val_mask] = iso.predict_proba(oof_probs[val_mask])

    return sigmoid_nested, isotonic_nested


def choose_calibrator(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    fold_indices: np.ndarray,
) -> tuple[CalibratorComparison, SigmoidCalibrator | IsotonicCalibrator]:
    """Execute pre-registered calibrator selection rule (Decision D74).

    Rule:
    1. Evaluate sigmoid and isotonic via nested cross-fitting across folds.
    2. Choose isotonic only if nested Brier score improves by >= 5.0% AND
       ECE is not worse than sigmoid (within 1e-4 tolerance).
    3. Fit the chosen calibrator on all OOF predictions.
    """
    sig_nested, iso_nested = evaluate_nested_calibrators(oof_probs, y_true, fold_indices)

    # Compute Brier scores
    sig_brier = float(np.mean((sig_nested - y_true) ** 2))
    iso_brier = float(np.mean((iso_nested - y_true) ** 2))

    # Compute ECE with 10 bins
    sig_ece = compute_expected_calibration_error(y_true, sig_nested, n_bins=10)
    iso_ece = compute_expected_calibration_error(y_true, iso_nested, n_bins=10)

    # Relative Brier improvement: (sig - iso) / sig
    if sig_brier > 0:
        brier_imp = float((sig_brier - iso_brier) / sig_brier)
    else:
        brier_imp = 0.0

    brier_imp_pct = brier_imp * 100.0

    # Decision rule: Brier improvement >= 5% and ECE not worse
    if brier_imp >= 0.05 and iso_ece <= sig_ece + 1e-4:
        chosen = "isotonic"
        reason = (
            f"Isotonic selected: Brier improved by {brier_imp_pct:.2f}% (>= 5%) "
            f"and ECE ({iso_ece:.4f}) did not worsen relative to sigmoid ({sig_ece:.4f})."
        )
        final_calibrator: SigmoidCalibrator | IsotonicCalibrator = (
            IsotonicCalibrator().fit(oof_probs, y_true)
        )
    else:
        chosen = "sigmoid"
        if brier_imp < 0.05:
            reason = (
                f"Sigmoid selected: Isotonic Brier improvement {brier_imp_pct:.2f}% "
                f"was below the 5% threshold."
            )
        else:
            reason = (
                f"Sigmoid selected: Isotonic ECE ({iso_ece:.4f}) worsened relative to "
                f"sigmoid ({sig_ece:.4f}) despite Brier gain."
            )
        final_calibrator = SigmoidCalibrator().fit(oof_probs, y_true)

    comparison = CalibratorComparison(
        chosen_method=chosen,
        sigmoid_brier=sig_brier,
        sigmoid_ece=sig_ece,
        isotonic_brier=iso_brier,
        isotonic_ece=iso_ece,
        brier_improvement_pct=brier_imp_pct,
        reason=reason,
    )
    logger.info("Calibrator choice: %s (%s)", chosen, reason)
    return comparison, final_calibrator
