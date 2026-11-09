"""Time splits and cross-validation folds for anomaly detection.

Implements Decision D44 (Exploration vs Holdout split) and Decision D55
(Expanding-window 3-fold CV inside the exploration period with 24-hour safety gaps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ml.eda.splits import exploration_cutoff, get_sensor_dates


@dataclass(frozen=True)
class CVFold:
    """Cross-validation fold boundaries."""

    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_idx": self.fold_idx,
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
            "val_start": str(self.val_start),
            "val_end": str(self.val_end),
        }


def get_periods() -> tuple[tuple[pd.Timestamp, pd.Timestamp], tuple[pd.Timestamp, pd.Timestamp]]:
    """Return ((exp_start, exp_end), (holdout_start, holdout_end)) timestamps.

    Exploration period covers 0.00 to 0.70 of sensor dates.
    Holdout period covers 0.70 to 1.00 of sensor dates (until timeline end).
    """
    dates = get_sensor_dates()
    if not dates:
        raise ValueError("No sensor dates available.")

    exp_start = dates[0]
    holdout_start = exploration_cutoff()
    exp_end = holdout_start  # Exploration ends at start of holdout day

    # Holdout ends at 00:00 UTC of day following the last date
    holdout_end = dates[-1] + pd.Timedelta(days=1)

    return (exp_start, exp_end), (holdout_start, holdout_end)


def cv_folds() -> list[CVFold]:
    """Return the three expanding-window CV folds inside the exploration period.

    Fold 1: Train 0.00 to 0.35 (minus 24h gap), Val 0.35 to 0.50
    Fold 2: Train 0.00 to 0.50 (minus 24h gap), Val 0.50 to 0.60
    Fold 3: Train 0.00 to 0.60 (minus 24h gap), Val 0.60 to 0.70
    """
    dates = get_sensor_dates()
    n = len(dates)

    exp_start = dates[0]
    idx_35 = int(round(n * 0.35))
    idx_50 = int(round(n * 0.50))
    idx_60 = int(round(n * 0.60))
    idx_70 = int(round(n * 0.70))

    d_35 = dates[idx_35]
    d_50 = dates[idx_50]
    d_60 = dates[idx_60]
    d_70 = dates[idx_70]

    gap = pd.Timedelta(hours=24)

    folds = [
        CVFold(
            fold_idx=1,
            train_start=exp_start,
            train_end=d_35 - gap,
            val_start=d_35,
            val_end=d_50,
        ),
        CVFold(
            fold_idx=2,
            train_start=exp_start,
            train_end=d_50 - gap,
            val_start=d_50,
            val_end=d_60,
        ),
        CVFold(
            fold_idx=3,
            train_start=exp_start,
            train_end=d_60 - gap,
            val_start=d_60,
            val_end=d_70,
        ),
    ]

    return folds


def final_fit_period() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the (start, end) timestamps for the full exploration fit [0.00, 0.70)."""
    (exp_start, exp_end), _ = get_periods()
    return exp_start, exp_end
