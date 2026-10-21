"""Sensor cleaning module: duplicates, regular 5-minute grid, faults, and short-gap imputation.

Implements non-destructive cleaning according to policies D30 and D31:
- Tier 1: Physically impossible values become NaN and trigger `is_sensor_fault`.
- Gaps <= 3 intervals (15 mins) are imputed and flagged with `is_imputed`.
- Longer gaps remain NaN and retain explicit `is_gap` flags.
- First 24 hours (288 intervals) per machine flagged as `is_warmup`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ml.preprocessing.config import (
    I_OFF,
    MAX_IMPUTE_INTERVALS,
    SENSOR_FREQ,
    WARMUP_INTERVALS,
)
from ml.preprocessing.validation_rules import SENSOR_LIMITS

logger = logging.getLogger(__name__)

INTERPOLATION_SIGNALS = ["temperature", "vibration", "pressure", "rpm", "motor_current"]


@dataclass
class CleaningLog:
    """Detailed summary log of cleaning actions performed."""

    raw_rows: int = 0
    exact_duplicates_dropped: int = 0
    conflicting_duplicates_resolved: int = 0
    explicit_gap_rows_created: int = 0
    hard_range_violations: dict[str, int] = field(default_factory=dict)
    imputed_values_per_signal: dict[str, int] = field(default_factory=dict)
    sensor_fault_rows: int = 0
    imputed_rows: int = 0
    warmup_rows: int = 0
    clean_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _impute_bounded_runs(
    series: pd.Series, max_gap: int, method: str = "linear"
) -> tuple[pd.Series, pd.Series]:
    """Impute missing runs of length <= max_gap only. Longer runs remain NaN."""
    s = series.copy()
    is_na = s.isna()
    if not is_na.any():
        return s, pd.Series(False, index=s.index)

    # Identify contiguous blocks of NaNs
    blocks = (is_na != is_na.shift()).cumsum()
    missing_blocks = blocks[is_na]
    counts = missing_blocks.map(missing_blocks.value_counts())

    # Only blocks with length <= max_gap are eligible
    eligible = is_na.copy()
    eligible.loc[is_na] = counts <= max_gap

    if method == "linear":
        s_filled = s.interpolate(method="linear")
    elif method == "ffill":
        s_filled = s.ffill()
    else:
        raise ValueError(f"Unsupported imputation method: {method}")

    s_out = s.copy()
    s_out[eligible] = s_filled[eligible]
    was_imputed = eligible & s_out.notna() & is_na

    return s_out, was_imputed


def clean_sensors_per_machine(
    df_m: pd.DataFrame, i_off: float = I_OFF, max_impute: int = MAX_IMPUTE_INTERVALS
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean sensor data for a single machine."""
    m_id = str(df_m["machine_id"].iloc[0])
    raw_count = len(df_m)
    log_m: dict[str, Any] = {
        "machine_id": m_id,
        "raw_rows": raw_count,
        "exact_dups": 0,
        "conflict_dups": 0,
        "gaps_created": 0,
        "faults": {},
        "imputed": {},
    }

    # Step 1: Deduplication
    exact_mask = df_m.duplicated()
    log_m["exact_dups"] = int(exact_mask.sum())
    df_dedup = df_m[~exact_mask].copy()

    # Conflicting duplicates on timestamp: keep row with fewest NaNs, then first
    ts_dup_mask = df_dedup.duplicated(subset=["timestamp"], keep=False)
    log_m["conflict_dups"] = int(ts_dup_mask.sum())
    if log_m["conflict_dups"] > 0:
        df_dedup["_na_count"] = df_dedup.isna().sum(axis=1)
        df_dedup = (
            df_dedup.sort_values(by=["timestamp", "_na_count"])
            .drop_duplicates(subset=["timestamp"], keep="first")
            .drop(columns=["_na_count"])
        )

    # Step 2: Regular 5-minute grid reindexing
    df_dedup = df_dedup.sort_values(by="timestamp").set_index("timestamp")
    min_ts = df_dedup.index.min()
    max_ts = df_dedup.index.max()
    full_grid = pd.date_range(min_ts, max_ts, freq=SENSOR_FREQ, tz=min_ts.tz)

    existing_idx = df_dedup.index
    missing_idx = full_grid.difference(existing_idx)
    log_m["gaps_created"] = len(missing_idx)

    # Reindex to regular grid
    df_grid = df_dedup.reindex(full_grid)
    df_grid["machine_id"] = m_id
    df_grid["is_gap"] = df_grid.index.isin(missing_idx)
    df_grid["is_sensor_fault"] = False
    df_grid["is_imputed"] = False

    # Step 3: Hard-range violations -> set to NaN and set is_sensor_fault
    for col, lim in SENSOR_LIMITS.items():
        if col in df_grid.columns:
            s = df_grid[col]
            viol_mask = s.notna() & ((s < lim.hard_min) | (s > lim.hard_max))
            viol_count = int(viol_mask.sum())
            if viol_count > 0:
                log_m["faults"][col] = viol_count
                df_grid.loc[viol_mask, col] = np.nan
                df_grid.loc[viol_mask, "is_sensor_fault"] = True

    # Step 4: Short-gap imputation (length <= MAX_IMPUTE_INTERVALS)
    for col in INTERPOLATION_SIGNALS:
        if col in df_grid.columns:
            filled_s, was_imp = _impute_bounded_runs(df_grid[col], max_impute, method="linear")
            df_grid[col] = filled_s.astype("float32")
            df_grid["is_imputed"] = df_grid["is_imputed"] | was_imp
            log_m["imputed"][col] = int(was_imp.sum())

    # comp_state: forward fill
    if "comp_state" in df_grid.columns:
        filled_c, was_imp_c = _impute_bounded_runs(
            df_grid["comp_state"], max_impute, method="ffill"
        )
        df_grid["comp_state"] = filled_c.astype("float32")
        df_grid["is_imputed"] = df_grid["is_imputed"] | was_imp_c
        log_m["imputed"]["comp_state"] = int(was_imp_c.sum())

    # energy_consumption: recompute from motor_current (0.59 * I * 5 / 60)
    if "motor_current" in df_grid.columns:
        df_grid["energy_consumption"] = (0.59 * df_grid["motor_current"] * 5.0 / 60.0).astype(
            "float32"
        )

    # operating_hours: forward fill only, non-decreasing
    if "operating_hours" in df_grid.columns:
        filled_h, was_imp_h = _impute_bounded_runs(
            df_grid["operating_hours"], max_impute, method="ffill"
        )
        # Enforce strictly non-decreasing via cummax
        df_grid["operating_hours"] = filled_h.cummax().astype("float32")
        df_grid["is_imputed"] = df_grid["is_imputed"] | was_imp_h
        log_m["imputed"]["operating_hours"] = int(was_imp_h.sum())

    # Step 5: Recompute running
    if "motor_current" in df_grid.columns:
        df_grid["running"] = (df_grid["motor_current"] > i_off).fillna(False)
    else:
        df_grid["running"] = False

    # Step 6: Warmup flag (first WARMUP_INTERVALS rows)
    df_grid["is_warmup"] = False
    if len(df_grid) >= WARMUP_INTERVALS:
        df_grid.iloc[:WARMUP_INTERVALS, df_grid.columns.get_loc("is_warmup")] = True
    else:
        df_grid["is_warmup"] = True

    df_clean = df_grid.reset_index().rename(columns={"index": "timestamp"})
    df_clean["machine_id"] = df_clean["machine_id"].astype("category")

    return df_clean, log_m


def clean_sensors(
    df_sensors: pd.DataFrame, i_off: float = I_OFF, max_impute: int = MAX_IMPUTE_INTERVALS
) -> tuple[pd.DataFrame, CleaningLog]:
    """Clean fleet sensor data machine-by-machine."""
    logger.info("Cleaning sensor telemetry for %d rows...", len(df_sensors))
    overall_log = CleaningLog(raw_rows=len(df_sensors))
    cleaned_frames: list[pd.DataFrame] = []

    for _m_id, group in df_sensors.groupby("machine_id", observed=True):
        m_clean, m_log = clean_sensors_per_machine(group, i_off=i_off, max_impute=max_impute)
        cleaned_frames.append(m_clean)

        overall_log.exact_duplicates_dropped += m_log["exact_dups"]
        overall_log.conflicting_duplicates_resolved += m_log["conflict_dups"]
        overall_log.explicit_gap_rows_created += m_log["gaps_created"]

        for col, cnt in m_log["faults"].items():
            overall_log.hard_range_violations[col] = (
                overall_log.hard_range_violations.get(col, 0) + cnt
            )

        for col, cnt in m_log["imputed"].items():
            overall_log.imputed_values_per_signal[col] = (
                overall_log.imputed_values_per_signal.get(col, 0) + cnt
            )

    df_all = pd.concat(cleaned_frames, ignore_index=True)
    df_all = df_all.sort_values(by=["machine_id", "timestamp"]).reset_index(drop=True)

    overall_log.sensor_fault_rows = int(df_all["is_sensor_fault"].sum())
    overall_log.imputed_rows = int(df_all["is_imputed"].sum())
    overall_log.warmup_rows = int(df_all["is_warmup"].sum())
    overall_log.clean_rows = len(df_all)

    logger.info(
        "Sensor cleaning complete: %d raw -> %d clean rows (gaps=%d, imputed_rows=%d, faults=%d)",
        overall_log.raw_rows,
        overall_log.clean_rows,
        overall_log.explicit_gap_rows_created,
        overall_log.imputed_rows,
        overall_log.sensor_fault_rows,
    )

    return df_all, overall_log
