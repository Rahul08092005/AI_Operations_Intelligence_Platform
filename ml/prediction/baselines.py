"""Baseline models for failure prediction benchmarking.

Implements Decision D76:
- Majority class baseline: predicts negative class with probability equal to empirical prevalence.
- Logistic Regression baseline: L2 regularized logistic regression with StandardScaler
  fitted strictly inside cross-validation training folds.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.prediction.config import SEED

logger = logging.getLogger(__name__)


class MajorityBaseline:
    """Majority class baseline (Decision D76).

    Always predicts class 0 (no failure) with predicted probability equal to the
    empirical training prevalence. Achieves high accuracy (~96.6%) but 0 true positives,
    illustrating the metric pitfall of class imbalance (EQ40).
    """

    def __init__(self) -> None:
        self.prevalence: float = 0.0

    def fit(self, X: Any, y: np.ndarray | pd.Series) -> MajorityBaseline:
        """Fit empirical target prevalence on training labels."""
        y_arr = np.asarray(y, dtype=np.int32)
        self.prevalence = float(np.mean(y_arr))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return 2D probability array [[1 - p, p], ...] where p is training prevalence."""
        n = len(X)
        p = self.prevalence
        prob_matrix = np.empty((n, 2), dtype=np.float64)
        prob_matrix[:, 0] = 1.0 - p
        prob_matrix[:, 1] = p
        return prob_matrix

    def predict(self, X: Any) -> np.ndarray:
        """Always predicts class 0."""
        n = len(X)
        return np.zeros(n, dtype=np.int32)


class LogisticRegressionBaseline:
    """Standardized Logistic Regression baseline (Decision D76).

    Combines StandardScaler and LogisticRegression in an isolated scikit-learn Pipeline
    to ensure feature scaling statistics never leak from validation folds.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str | dict[int, float] | None = None,
        max_iter: int = 1000,
        random_state: int = SEED,
    ) -> None:
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

        self.pipeline: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.C,
                        class_weight=self.class_weight,
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series
    ) -> LogisticRegressionBaseline:
        """Fit scaler and logistic regression on training fold."""
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return 2D predicted probability array."""
        return self.pipeline.predict_proba(X)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return binary class predictions."""
        return self.pipeline.predict(X)
