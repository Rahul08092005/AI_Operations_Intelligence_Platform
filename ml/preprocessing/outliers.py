"""Tier 2 statistical outlier detection and anomaly preservation checks.

Flags unusual observations without removing or altering values:
- Computes trailing robust z-scores (median and MAD of previous 24h, shifted by 1).
- Sets `is_outlier_flag = True` and populates `outlier_signals`.
- Performs 4 automated anomaly preservation checks.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml.preprocessing.config import ROLLING_WINDOWS, STAT_OUTLIER_ROBUST_Z

logger = logging.getLogger(__name__)

OUTLIER_SIGNALS = [
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "energy_consumption",
    "motor_current",
]


@dataclass
class PreservationReport:
    """Automated anomaly preservation verification results."""

    zero_real_data_destroyed: bool
    peak_preservation_temperature: bool
    peak_preservation_vibration: bool
    benign_spikes_preserved: bool
    tier2_flag_rates: dict[str, float]
    passed: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_machine_outlier_flags(
    df_m: pd.DataFrame,
    threshold: float = STAT_OUTLIER_ROBUST_Z,
    window: int = ROLLING_WINDOWS["24h"],  # 288 intervals
) -> pd.DataFrame:
    """Compute Tier 2 statistical outlier flags for one machine."""
    df = df_m.copy()
    min_periods = int(window * 0.5)  # 144

    flagged_signals_per_row: list[list[str]] = [[] for _ in range(len(df))]

    for sig in OUTLIER_SIGNALS:
        if sig not in df.columns:
            continue

        s = df[sig].astype("float64")
        # Shift by 1 interval so current value is excluded from its own baseline
        s_past = s.shift(1)
        med = s_past.rolling(window=window, min_periods=min_periods).median()
        abs_dev = (s_past - med).abs()
        mad = abs_dev.rolling(window=window, min_periods=min_periods).median()
        # Scale MAD to normal standard deviation estimate
        norm_mad = 1.4826 * mad
        # Avoid division by zero
        norm_mad = norm_mad.replace(0, np.nan)

        rob_z = (s - med).abs() / norm_mad
        outlier_mask = rob_z > threshold

        # Record signal names for flagged rows
        for idx in np.flatnonzero(outlier_mask.fillna(False).to_numpy()):
            flagged_signals_per_row[idx].append(sig)

    df["is_outlier_flag"] = [len(sigs) > 0 for sigs in flagged_signals_per_row]
    df["outlier_signals"] = [",".join(sigs) if sigs else "" for sigs in flagged_signals_per_row]

    return df


def flag_outliers(
    df_sensors: pd.DataFrame, threshold: float = STAT_OUTLIER_ROBUST_Z
) -> pd.DataFrame:
    """Compute Tier 2 outlier flags across all machines."""
    logger.info("Computing Tier 2 robust outlier flags (Z > %.1f)...", threshold)
    flagged_frames: list[pd.DataFrame] = []

    for _m_id, group in df_sensors.groupby("machine_id", observed=True):
        f_df = compute_machine_outlier_flags(group, threshold=threshold)
        flagged_frames.append(f_df)

    df_out = pd.concat(flagged_frames, ignore_index=True)
    df_out = df_out.sort_values(by=["machine_id", "timestamp"]).reset_index(drop=True)

    total_flags = int(df_out["is_outlier_flag"].sum())
    logger.info("Tier 2 outlier flagging complete: %d rows flagged", total_flags)
    return df_out


def run_preservation_checks(
    df_raw_sensors: pd.DataFrame,
    df_clean_sensors: pd.DataFrame,
    df_failure_windows: pd.DataFrame,
    shift_days: int = 2240,
) -> PreservationReport:
    """Verify that data cleaning preserved real anomalies and degradation."""
    logger.info("Running automated anomaly preservation checks...")
    details: dict[str, Any] = {}

    # 1. Zero real data destroyed on M-104 inside shifted failure windows
    m104_clean = df_clean_sensors[df_clean_sensors["machine_id"] == "M-104"]
    in_fw_faults = 0
    total_fw_rows = 0

    shift_delta = pd.Timedelta(days=shift_days)
    for _, row in df_failure_windows.iterrows():
        start_shifted = pd.to_datetime(row["start_time"], utc=True) + shift_delta
        end_shifted = pd.to_datetime(row["end_time"], utc=True) + shift_delta
        fw_mask = (m104_clean["timestamp"] >= start_shifted) & (
            m104_clean["timestamp"] <= end_shifted
        )
        fw_rows = m104_clean[fw_mask]
        total_fw_rows += len(fw_rows)
        in_fw_faults += int(fw_rows["is_sensor_fault"].sum())

    check1_passed = in_fw_faults == 0
    details["real_failure_faults_count"] = in_fw_faults
    details["total_failure_window_rows"] = total_fw_rows

    # 2. Peak preservation: max M-104 vibration and temperature in last 7 days
    m104_raw = df_raw_sensors[df_raw_sensors["machine_id"] == "M-104"]
    max_ts = m104_clean["timestamp"].max()
    last_7d_ts = max_ts - pd.Timedelta(days=7)

    m104_raw_7d = m104_raw[pd.to_datetime(m104_raw["timestamp"], utc=True) >= last_7d_ts]
    m104_clean_7d = m104_clean[m104_clean["timestamp"] >= last_7d_ts]

    raw_temp_max = float(m104_raw_7d["temperature"].max())
    clean_temp_max = float(m104_clean_7d["temperature"].max())
    check2_temp = abs(raw_temp_max - clean_temp_max) < 1e-3

    raw_vib_max = float(m104_raw_7d["vibration"].max())
    clean_vib_max = float(m104_clean_7d["vibration"].max())
    check2_vib = abs(raw_vib_max - clean_vib_max) < 1e-3

    details["peak_temp_raw"] = raw_temp_max
    details["peak_temp_clean"] = clean_temp_max
    details["peak_vib_raw"] = raw_vib_max
    details["peak_vib_clean"] = clean_vib_max

    # 3. Benign spikes kept on M-109
    m109_raw = df_raw_sensors[df_raw_sensors["machine_id"] == "M-109"].sort_values("timestamp")
    m109_clean = df_clean_sensors[df_clean_sensors["machine_id"] == "M-109"].sort_values(
        "timestamp"
    )

    if len(m109_raw) > 0 and len(m109_clean) > 0:
        raw_med = m109_raw["vibration"].rolling(288, min_periods=72).median()
        clean_med = m109_clean["vibration"].rolling(288, min_periods=72).median()

        raw_spikes = int((m109_raw["vibration"] > 1.5 * raw_med).sum())
        clean_spikes = int((m109_clean["vibration"] > 1.5 * clean_med).sum())
        check3_passed = raw_spikes == clean_spikes
        details["m109_raw_spikes"] = raw_spikes
        details["m109_clean_spikes"] = clean_spikes
    else:
        check3_passed = True  # Not present in sample mode

    # 4. Tier 2 flag rates per machine
    flag_rates: dict[str, float] = {}
    for m_id, grp in df_clean_sensors.groupby("machine_id", observed=True):
        rate = float((grp["is_outlier_flag"].sum() / len(grp)) * 100.0)
        flag_rates[str(m_id)] = round(rate, 3)

    details["flag_rates_pct"] = flag_rates

    overall_passed = bool(check1_passed and check2_temp and check2_vib and check3_passed)

    logger.info(
        "Preservation checks result: %s (real_faults=%d, peak_temp_diff=%.3f, peak_vib_diff=%.3f)",
        "PASSED" if overall_passed else "FAILED",
        in_fw_faults,
        abs(raw_temp_max - clean_temp_max),
        abs(raw_vib_max - clean_vib_max),
    )

    return PreservationReport(
        zero_real_data_destroyed=check1_passed,
        peak_preservation_temperature=check2_temp,
        peak_preservation_vibration=check2_vib,
        benign_spikes_preserved=check3_passed,
        tier2_flag_rates=flag_rates,
        passed=overall_passed,
        details=details,
    )
