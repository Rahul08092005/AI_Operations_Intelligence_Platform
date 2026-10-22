"""Splits utilities: Exploration vs Holdout time split, and AI4I stratified split."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.eda.config import (
    AI4I_SPLIT_INDEX,
    AI4I_TEST_FRACTION,
    EXPLORATION_FRACTION,
    PROCESSED,
    SEED,
)


@functools.lru_cache(maxsize=1)
def get_sensor_dates() -> list[pd.Timestamp]:
    """Return sorted unique dates (00:00 UTC) present in the sensor dataset."""
    path = PROCESSED / "sensors.parquet"
    if path.exists():
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["timestamp"])
        ts_series = table["timestamp"].to_pandas()
    else:
        # Fallback to operations dataset dates if sensors not present
        ops_path = PROCESSED / "operations_dataset.parquet"
        if not ops_path.exists():
            raise FileNotFoundError("Neither sensors.parquet nor operations_dataset.parquet found.")
        import pyarrow.parquet as pq

        table = pq.read_table(ops_path, columns=["date"])
        ts_series = pd.to_datetime(table["date"].to_pandas())

    dates = pd.to_datetime(ts_series, utc=True).dt.floor("D").drop_duplicates().sort_values()
    return list(dates)


@functools.lru_cache(maxsize=1)
def exploration_cutoff() -> pd.Timestamp:
    """Return the first timestamp of the holdout period (00:00 UTC).

    Calculated as the date at the EXPLORATION_FRACTION quantile of all sensor dates.
    """
    dates = get_sensor_dates()
    n_dates = len(dates)
    if n_dates == 0:
        raise ValueError("No dates found in sensor dataset.")

    # Fraction of days in exploration
    exp_count = int(round(n_dates * EXPLORATION_FRACTION))
    # Cutoff is the start of the first holdout day at 00:00 UTC
    return dates[exp_count]


def is_exploration(ts: Any) -> Any:
    """Return boolean mask or bool indicating whether timestamp/date is in exploration.

    Disjoint and strictly chronological: ts < exploration_cutoff().
    """
    cutoff = exploration_cutoff()

    if isinstance(ts, pd.DatetimeIndex):
        s = ts
        if s.tz is None:
            s = s.tz_localize("UTC")
        else:
            s = s.tz_convert("UTC")
        return s < cutoff

    if isinstance(ts, pd.Series):
        s = pd.to_datetime(ts)
        if s.dt.tz is None:
            s = s.dt.tz_localize("UTC")
        else:
            s = s.dt.tz_convert("UTC")
        return s < cutoff

    if isinstance(ts, np.ndarray):
        s = pd.to_datetime(ts)
        if s.tz is None:
            s = s.tz_localize("UTC")
        else:
            s = s.tz_convert("UTC")
        return s < cutoff

    # Scalar handling
    val = pd.to_datetime(ts)
    if val.tz is None:
        val = val.tz_localize("UTC")
    else:
        val = val.tz_convert("UTC")
    return val < cutoff


def is_holdout(ts: Any) -> Any:
    """Return boolean mask or bool indicating whether timestamp/date is in holdout."""
    return ~is_exploration(ts)


def make_ai4i_split() -> pd.DataFrame:
    """Create stratified train/test split (80/20) for AI4I on `machine_failure`."""
    from ml.eda.load import load_ai4i

    df = load_ai4i(partition=None)
    train_df, test_df = train_test_split(
        df[["udi", "machine_failure"]],
        test_size=AI4I_TEST_FRACTION,
        stratify=df["machine_failure"],
        random_state=SEED,
        shuffle=True,
    )

    train_res = pd.DataFrame({"udi": train_df["udi"], "partition": "train"})
    test_res = pd.DataFrame({"udi": test_df["udi"], "partition": "test"})
    split_df = (
        pd.concat([train_res, test_res], ignore_index=True)
        .sort_values("udi")
        .reset_index(drop=True)
    )
    return split_df


def save_ai4i_split(path: Path | str = AI4I_SPLIT_INDEX) -> Path:
    """Save the AI4I stratified split index to CSV."""
    split_df = make_ai4i_split()
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(target_path, index=False)
    return target_path


__all__ = [
    "exploration_cutoff",
    "get_sensor_dates",
    "is_exploration",
    "is_holdout",
    "make_ai4i_split",
    "save_ai4i_split",
]
