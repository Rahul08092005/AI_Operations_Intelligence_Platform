"""Sensor feature engineering module for high-frequency telemetry.

Produces `data/processed/sensor_features.parquet` with trailing rolling statistics,
vectorized rate of change, least-squares slopes, operating context, and
leakage-safe maintenance context features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.preprocessing.config import (
    FAILURE_LOOKBACK_DAYS,
    OVERDUE_INSPECTION_H,
    OVERDUE_PREVENTIVE_H,
    PROCESSED,
    ROLLING_MIN_FRACTION,
    ROLLING_WINDOWS,
    TEMP_MAX_SPEC,
    VIB_CRITICAL,
    VIB_WARNING,
)

logger = logging.getLogger(__name__)

NUMERIC_SIGNALS = [
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "energy_consumption",
    "motor_current",
]


def _compute_vectorized_slope_24h(series: pd.Series, window: int = 288) -> pd.Series:
    """Compute vectorized least-squares slope per hour over trailing 24 hours."""
    dt = 5.0 / 60.0  # 5 minutes in hours
    x = np.arange(window, dtype=np.float64) * dt
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)
    weights = (x - x_mean) / denom

    values = series.astype(np.float64).to_numpy()
    slopes = np.full(len(values), np.nan, dtype=np.float32)

    if len(values) >= window:
        conv = np.convolve(values, weights[::-1], mode="valid")
        slopes[window - 1 :] = conv.astype(np.float32)

    return pd.Series(slopes, index=series.index)


def compute_maintenance_features_by_date(
    df_maint: pd.DataFrame,
    df_sensors: pd.DataFrame,
    machine_id: str,
    unique_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """Precompute date-level strictly past maintenance features for a machine."""
    maint_m = df_maint[df_maint["machine_id"] == machine_id].copy()
    maint_m["m_date"] = pd.to_datetime(maint_m["performed_date"], utc=True).dt.floor("D")

    # Compute machine's average daily running hours from the sensor data
    m_sensors = df_sensors[df_sensors["machine_id"] == machine_id]
    daily_running_h = m_sensors.groupby(m_sensors["timestamp"].dt.floor("D"))["running"].sum() * (
        5.0 / 60.0
    )
    mean_daily_running_h = float(daily_running_h.mean()) if len(daily_running_h) > 0 else 16.0

    # End-of-day operating hours map
    eod_op_hours: dict[pd.Timestamp, float] = {}
    for d, grp in m_sensors.groupby(m_sensors["timestamp"].dt.floor("D")):
        valid_op = grp["operating_hours"].dropna()
        last_val = valid_op.iloc[-1] if len(valid_op) > 0 else np.nan
        eod_op_hours[d] = float(last_val)

    rows: list[dict[str, Any]] = []

    for d in unique_dates:
        # Strictly past maintenance events (date < d)
        past_events = maint_m[maint_m["m_date"] < d]

        # 1. Any maintenance
        if len(past_events) > 0:
            last_event = past_events.sort_values("m_date").iloc[-1]
            days_since_maint = float((d - last_event["m_date"]).days)
        else:
            days_since_maint = np.nan

        # 2. Inspections
        insp_type = past_events["maintenance_type"].astype(str).str.capitalize() == "Inspection"
        inspections = past_events[insp_type]
        if len(inspections) > 0:
            last_insp = inspections.sort_values("m_date").iloc[-1]
            insp_date = last_insp["m_date"]
            days_since_insp = float((d - insp_date).days)
            if insp_date in eod_op_hours and not np.isnan(eod_op_hours[insp_date]):
                op_h_insp_base = eod_op_hours[insp_date]
                est_insp = 0
            else:
                op_h_insp_base = -1.0
                est_insp = 1
        else:
            days_since_insp = np.nan
            op_h_insp_base = np.nan
            est_insp = 0

        # 3. Preventive maintenance
        pm_type = (
            past_events["maintenance_type"].astype(str).str.capitalize() == "Preventive"
        )
        preventives = past_events[pm_type]
        if len(preventives) > 0:
            last_pm = preventives.sort_values("m_date").iloc[-1]
            pm_date = last_pm["m_date"]
            days_since_pm = float((d - pm_date).days)
            if pm_date in eod_op_hours and not np.isnan(eod_op_hours[pm_date]):
                op_h_pm_base = eod_op_hours[pm_date]
                est_pm = 0
            else:
                op_h_pm_base = -1.0
                est_pm = 1
        else:
            # No past PM recorded: machine is overdue (> 2000 hours)
            days_since_pm = float((d - pd.Timestamp("2026-03-21", tz="UTC")).days) + 90.0
            op_h_pm_base = -2.0  # marker for no past PM (guaranteed overdue)
            est_pm = 1

        # 4. Failures (Corrective)
        corr_type = past_events["maintenance_type"].astype(str).str.capitalize() == "Corrective"
        correctives = past_events[corr_type]
        if len(correctives) > 0:
            last_fail = correctives.sort_values("m_date").iloc[-1]
            days_since_fail = float((d - last_fail["m_date"]).days)
            failure_count = len(correctives)
            prev_failure = 1.0 if days_since_fail <= FAILURE_LOOKBACK_DAYS else 0.0
        else:
            days_since_fail = np.nan
            failure_count = 0.0
            prev_failure = 0.0

        # 5. Costs & counts in trailing 90 days
        d_90d_ago = d - pd.Timedelta(days=90)
        events_90d = past_events[past_events["m_date"] >= d_90d_ago]
        cost_col = "cost_gbp" if "cost_gbp" in past_events.columns else "maintenance_cost"
        maint_cost_avg = float(past_events[cost_col].mean()) if len(past_events) > 0 else 0.0
        maint_cost_90d = float(events_90d[cost_col].sum()) if len(events_90d) > 0 else 0.0

        rows.append(
            {
                "date": d,
                "days_since_maintenance": days_since_maint,
                "days_since_inspection": days_since_insp,
                "days_since_preventive": days_since_pm,
                "op_h_insp_base": op_h_insp_base,
                "op_h_pm_base": op_h_pm_base,
                "hours_since_estimated": float(est_insp or est_pm),
                "maintenance_count": float(len(past_events)),
                "maintenance_count_90d": float(len(events_90d)),
                "maintenance_cost_avg": maint_cost_avg,
                "maintenance_cost_90d": maint_cost_90d,
                "failure_count": float(failure_count),
                "days_since_failure": days_since_fail,
                "previous_failure": float(prev_failure),
                "mean_daily_running_h": mean_daily_running_h,
            }
        )

    return pd.DataFrame(rows)


def engineer_sensor_features(
    df_sensors: pd.DataFrame,
    df_maint: pd.DataFrame,
) -> pd.DataFrame:
    """Compute rich sensor features across all machines."""
    logger.info("Computing sensor rolling, trend, and context features...")
    machine_dfs: list[pd.DataFrame] = []

    for m_id, group in df_sensors.groupby("machine_id", observed=True):
        m_id_str = str(m_id)
        df_m = group.sort_values("timestamp").copy()

        # 8b. Rolling statistics
        for sig in NUMERIC_SIGNALS:
            if sig not in df_m.columns:
                continue
            s = df_m[sig]
            for win_name, win_size in ROLLING_WINDOWS.items():
                min_p = int(win_size * ROLLING_MIN_FRACTION)
                df_m[f"{sig}_roll_mean_{win_name}"] = (
                    s.rolling(win_size, min_periods=min_p).mean().astype(np.float32)
                )
                df_m[f"{sig}_roll_std_{win_name}"] = (
                    s.rolling(win_size, min_periods=min_p).std().astype(np.float32)
                )

        # Max features for temperature and vibration
        for sig in ["temperature", "vibration"]:
            if sig in df_m.columns:
                s = df_m[sig]
                min_p_1h = int(ROLLING_WINDOWS["1h"] * ROLLING_MIN_FRACTION)
                min_p_24h = int(ROLLING_WINDOWS["24h"] * ROLLING_MIN_FRACTION)
                df_m[f"{sig}_roll_max_1h"] = (
                    s.rolling(ROLLING_WINDOWS["1h"], min_periods=min_p_1h).max().astype(np.float32)
                )
                df_m[f"{sig}_roll_max_24h"] = (
                    s.rolling(ROLLING_WINDOWS["24h"], min_periods=min_p_24h)
                    .max()
                    .astype(np.float32)
                )

        # 8c. Changes and trends
        for sig in ["temperature", "vibration", "pressure"]:
            if sig in df_m.columns:
                s = df_m[sig]
                df_m[f"{sig}_change"] = (s - s.shift(ROLLING_WINDOWS["1h"])).astype(np.float32)
                df_m[f"{sig}_change_24h"] = (s - s.shift(ROLLING_WINDOWS["24h"])).astype(np.float32)

        if "energy_consumption" in df_m.columns:
            s_e = df_m["energy_consumption"]
            df_m["energy_change"] = (s_e - s_e.shift(ROLLING_WINDOWS["1h"])).astype(np.float32)
            df_m["energy_change_24h"] = (s_e - s_e.shift(ROLLING_WINDOWS["24h"])).astype(np.float32)

        # Vectorized slopes (24h)
        if "vibration" in df_m.columns:
            df_m["vibration_slope_24h"] = _compute_vectorized_slope_24h(df_m["vibration"])
        if "temperature" in df_m.columns:
            df_m["temperature_slope_24h"] = _compute_vectorized_slope_24h(df_m["temperature"])

        # 8d. Operating-context features
        if "running" in df_m.columns:
            r = df_m["running"].astype(np.float32)
            df_m["running_ratio_1h"] = (
                r.rolling(ROLLING_WINDOWS["1h"], min_periods=1).mean().astype(np.float32)
            )
            df_m["running_ratio_24h"] = (
                r.rolling(ROLLING_WINDOWS["24h"], min_periods=1).mean().astype(np.float32)
            )

        if "comp_state" in df_m.columns:
            c = df_m["comp_state"].astype(np.float32)
            df_m["load_ratio_1h"] = (
                c.rolling(ROLLING_WINDOWS["1h"], min_periods=1).mean().astype(np.float32)
            )

        vib = df_m["vibration"]
        temp = df_m["temperature"]
        df_m["vibration_above_warning"] = np.where(
            vib.isna(), np.nan, (vib > VIB_WARNING).astype(np.float32)
        )
        df_m["vibration_above_critical"] = np.where(
            vib.isna(), np.nan, (vib > VIB_CRITICAL).astype(np.float32)
        )
        df_m["temperature_above_spec"] = np.where(
            temp.isna(), np.nan, (temp > TEMP_MAX_SPEC).astype(np.float32)
        )

        df_m["vibration_above_warning_ratio_24h"] = (
            df_m["vibration_above_warning"]
            .rolling(ROLLING_WINDOWS["24h"], min_periods=1)
            .mean()
            .astype(np.float32)
        )

        # 8e. Maintenance context (date-level precomputation joined by date)
        df_m["date"] = df_m["timestamp"].dt.floor("D")
        unique_dates = df_m["date"].drop_duplicates().tolist()
        maint_date_df = compute_maintenance_features_by_date(
            df_maint=df_maint,
            df_sensors=df_sensors,
            machine_id=m_id_str,
            unique_dates=unique_dates,
        )

        df_m = df_m.merge(maint_date_df, on="date", how="left")

        # Operating hours since inspection / preventive
        insp_direct = df_m["op_h_insp_base"] >= 0
        df_m["operating_hours_since_inspection"] = np.where(
            insp_direct,
            df_m["operating_hours"] - df_m["op_h_insp_base"],
            df_m["days_since_inspection"] * df_m["mean_daily_running_h"],
        ).astype(np.float32)

        pm_direct = df_m["op_h_pm_base"] >= 0
        pm_no_prior = df_m["op_h_pm_base"] == -2.0
        df_m["operating_hours_since_preventive"] = np.where(
            pm_direct,
            df_m["operating_hours"] - df_m["op_h_pm_base"],
            np.where(
                pm_no_prior,
                2100.0 + (df_m["days_since_preventive"] * df_m["mean_daily_running_h"]),
                df_m["days_since_preventive"] * df_m["mean_daily_running_h"],
            ),
        ).astype(np.float32)

        # Overdue flags
        df_m["inspection_overdue"] = (
            (df_m["operating_hours_since_inspection"] >= OVERDUE_INSPECTION_H)
            .astype(np.float32)
            .fillna(0.0)
        )
        df_m["preventive_overdue"] = (
            (df_m["operating_hours_since_preventive"] >= OVERDUE_PREVENTIVE_H)
            .astype(np.float32)
            .fillna(1.0)
        )

        # Drop temporary join columns
        df_m = df_m.drop(columns=["date", "op_h_insp_base", "op_h_pm_base", "mean_daily_running_h"])

        machine_dfs.append(df_m)

    df_features = pd.concat(machine_dfs, ignore_index=True)
    df_features = df_features.sort_values(by=["machine_id", "timestamp"]).reset_index(drop=True)

    # Cast all engineered float columns to float32
    for c in df_features.columns:
        if df_features[c].dtype == "float64":
            df_features[c] = df_features[c].astype(np.float32)

    logger.info(
        "Sensor feature engineering complete: %d columns generated", len(df_features.columns)
    )
    return df_features


def generate_feature_catalog(output_dir: Path | None = None) -> pd.DataFrame:
    """Generate the complete feature catalog in CSV and Markdown formats."""
    out_dir = output_dir or PROCESSED
    docs_dir = out_dir.parents[1] / "docs"

    specs = [
        (
            "timestamp",
            "keys",
            "5-min UTC sensor timestamp",
            "ISO 8601",
            "None",
            "timestamp",
            "L1",
            "timestamp",
        ),
        (
            "machine_id",
            "keys",
            "Machine identifier",
            "category",
            "None",
            "machine_id",
            "L1",
            "machine_id",
        ),
        (
            "temperature",
            "base_signals",
            "Cleaned temperature",
            "deg_C",
            "current",
            "temperature",
            "L1",
            "temperature",
        ),
        (
            "vibration",
            "base_signals",
            "Cleaned RMS vibration",
            "mm/s",
            "current",
            "vibration",
            "L1",
            "vibration",
        ),
        (
            "pressure",
            "base_signals",
            "Cleaned discharge pressure",
            "bar",
            "current",
            "pressure",
            "L1",
            "pressure",
        ),
        (
            "rpm",
            "base_signals",
            "Cleaned shaft rotational speed",
            "rpm",
            "current",
            "rpm",
            "L1",
            "rpm",
        ),
        (
            "energy_consumption",
            "base_signals",
            "5-min energy consumption",
            "kWh",
            "current",
            "energy_consumption",
            "L1",
            "energy_consumption",
        ),
        (
            "operating_hours",
            "base_signals",
            "Monotonic operating hours",
            "hours",
            "current",
            "operating_hours",
            "L1",
            "operating_hours",
        ),
        (
            "motor_current",
            "base_signals",
            "Cleaned motor current",
            "A",
            "current",
            "motor_current",
            "L1",
            "motor_current",
        ),
        (
            "comp_state",
            "base_signals",
            "Valve load state (0=load, 1=offload)",
            "binary",
            "current",
            "comp_state",
            "L1",
            "comp_state",
        ),
        (
            "running",
            "base_signals",
            "Running flag (current > 1.55A)",
            "boolean",
            "current",
            "motor_current",
            "L1",
            "running",
        ),
        (
            "is_imputed",
            "quality_flags",
            "Row contains imputed values",
            "boolean",
            "None",
            "all_signals",
            "L5",
            "is_imputed",
        ),
        (
            "is_gap",
            "quality_flags",
            "Row created by 5-min grid alignment",
            "boolean",
            "None",
            "timestamp",
            "L1",
            "is_gap",
        ),
        (
            "is_sensor_fault",
            "quality_flags",
            "Row had physical limit violation",
            "boolean",
            "None",
            "all_signals",
            "L1",
            "is_sensor_fault",
        ),
        (
            "is_outlier_flag",
            "quality_flags",
            "Tier 2 outlier flag (|Z| > 6.0)",
            "boolean",
            "24h trailing",
            "numeric_signals",
            "L1",
            "is_outlier_flag",
        ),
        (
            "outlier_signals",
            "quality_flags",
            "Signals triggering outlier flag",
            "string",
            "24h trailing",
            "numeric_signals",
            "L1",
            "outlier_signals",
        ),
        (
            "is_warmup",
            "quality_flags",
            "First 24h of machine series",
            "boolean",
            "None",
            "timestamp",
            "L1",
            "is_warmup",
        ),
        (
            "temperature_roll_mean_1h",
            "rolling_stats",
            "1h trailing mean temperature",
            "deg_C",
            "1h (12)",
            "temperature",
            "L1",
            "rolling_temperature_mean_1h",
        ),
        (
            "temperature_roll_std_1h",
            "rolling_stats",
            "1h trailing std temperature",
            "deg_C",
            "1h (12)",
            "temperature",
            "L1",
            "rolling_temperature_std_1h",
        ),
        (
            "temperature_roll_mean_6h",
            "rolling_stats",
            "6h trailing mean temperature",
            "deg_C",
            "6h (72)",
            "temperature",
            "L1",
            "rolling_temperature_mean_6h",
        ),
        (
            "temperature_roll_std_6h",
            "rolling_stats",
            "6h trailing std temperature",
            "deg_C",
            "6h (72)",
            "temperature",
            "L1",
            "rolling_temperature_std_6h",
        ),
        (
            "temperature_roll_mean_24h",
            "rolling_stats",
            "24h trailing mean temperature",
            "deg_C",
            "24h (288)",
            "temperature",
            "L1",
            "rolling_temperature_mean_24h",
        ),
        (
            "temperature_roll_std_24h",
            "rolling_stats",
            "24h trailing std temperature",
            "deg_C",
            "24h (288)",
            "temperature",
            "L1",
            "rolling_temperature_std_24h",
        ),
        (
            "temperature_roll_max_1h",
            "rolling_stats",
            "1h trailing max temperature",
            "deg_C",
            "1h (12)",
            "temperature",
            "L1",
            "temperature_max_1h",
        ),
        (
            "temperature_roll_max_24h",
            "rolling_stats",
            "24h trailing max temperature",
            "deg_C",
            "24h (288)",
            "temperature",
            "L1",
            "temperature_max_24h",
        ),
        (
            "vibration_roll_mean_1h",
            "rolling_stats",
            "1h trailing mean vibration",
            "mm/s",
            "1h (12)",
            "vibration",
            "L1",
            "rolling_vibration_mean_1h",
        ),
        (
            "vibration_roll_std_1h",
            "rolling_stats",
            "1h trailing std vibration",
            "mm/s",
            "1h (12)",
            "vibration",
            "L1",
            "rolling_vibration_std_1h",
        ),
        (
            "vibration_roll_mean_6h",
            "rolling_stats",
            "6h trailing mean vibration",
            "mm/s",
            "6h (72)",
            "vibration",
            "L1",
            "rolling_vibration_mean_6h",
        ),
        (
            "vibration_roll_std_6h",
            "rolling_stats",
            "6h trailing std vibration",
            "mm/s",
            "6h (72)",
            "vibration",
            "L1",
            "rolling_vibration_std_6h",
        ),
        (
            "vibration_roll_mean_24h",
            "rolling_stats",
            "24h trailing mean vibration",
            "mm/s",
            "24h (288)",
            "vibration",
            "L1",
            "rolling_vibration_mean_24h",
        ),
        (
            "vibration_roll_std_24h",
            "rolling_stats",
            "24h trailing std vibration",
            "mm/s",
            "24h (288)",
            "vibration",
            "L1",
            "rolling_vibration_std_24h",
        ),
        (
            "vibration_roll_max_1h",
            "rolling_stats",
            "1h trailing max vibration",
            "mm/s",
            "1h (12)",
            "vibration",
            "L1",
            "vibration_max_1h",
        ),
        (
            "vibration_roll_max_24h",
            "rolling_stats",
            "24h trailing max vibration",
            "mm/s",
            "24h (288)",
            "vibration",
            "L1",
            "vibration_max_24h",
        ),
        (
            "temperature_change",
            "trends",
            "Temperature delta vs 1h earlier",
            "deg_C",
            "1h (12)",
            "temperature",
            "L1",
            "temperature_change",
        ),
        (
            "vibration_change",
            "trends",
            "Vibration delta vs 1h earlier",
            "mm/s",
            "1h (12)",
            "vibration",
            "L1",
            "vibration_change",
        ),
        (
            "pressure_change",
            "trends",
            "Pressure delta vs 1h earlier",
            "bar",
            "1h (12)",
            "pressure",
            "L1",
            "pressure_change",
        ),
        (
            "energy_change",
            "trends",
            "Energy delta vs 1h earlier",
            "kWh",
            "1h (12)",
            "energy_consumption",
            "L1",
            "energy_change",
        ),
        (
            "temperature_change_24h",
            "trends",
            "Temperature delta vs 24h earlier",
            "deg_C",
            "24h (288)",
            "temperature",
            "L1",
            "temperature_change_24h",
        ),
        (
            "vibration_change_24h",
            "trends",
            "Vibration delta vs 24h earlier",
            "mm/s",
            "24h (288)",
            "vibration",
            "L1",
            "vibration_change_24h",
        ),
        (
            "pressure_change_24h",
            "trends",
            "Pressure delta vs 24h earlier",
            "bar",
            "24h (288)",
            "pressure",
            "L1",
            "pressure_change_24h",
        ),
        (
            "energy_change_24h",
            "trends",
            "Energy delta vs 24h earlier",
            "kWh",
            "24h (288)",
            "energy_consumption",
            "L1",
            "energy_change_24h",
        ),
        (
            "vibration_slope_24h",
            "trends",
            "LS vibration slope/h (24h)",
            "mm/s per h",
            "24h (288)",
            "vibration",
            "L1",
            "vibration_slope_24h",
        ),
        (
            "temperature_slope_24h",
            "trends",
            "LS temperature slope/h (24h)",
            "deg_C per h",
            "24h (288)",
            "temperature",
            "L1",
            "temperature_slope_24h",
        ),
        (
            "running_ratio_1h",
            "context",
            "Fraction 1h running",
            "fraction",
            "1h (12)",
            "running",
            "L1",
            "running_ratio_1h",
        ),
        (
            "running_ratio_24h",
            "context",
            "Fraction 24h running",
            "fraction",
            "24h (288)",
            "running",
            "L1",
            "running_ratio_24h",
        ),
        (
            "load_ratio_1h",
            "context",
            "Fraction 1h under load",
            "fraction",
            "1h (12)",
            "comp_state",
            "L1",
            "load_ratio_1h",
        ),
        (
            "vibration_above_warning",
            "context",
            "Vibration > 4.5 mm/s flag",
            "boolean",
            "current",
            "vibration",
            "L1",
            "vibration_above_warning",
        ),
        (
            "vibration_above_critical",
            "context",
            "Vibration > 7.0 mm/s flag",
            "boolean",
            "current",
            "vibration",
            "L1",
            "vibration_above_critical",
        ),
        (
            "temperature_above_spec",
            "context",
            "Temperature > 85 deg_C flag",
            "boolean",
            "current",
            "temperature",
            "L1",
            "temperature_above_spec",
        ),
        (
            "vibration_above_warning_ratio_24h",
            "context",
            "Trailing 24h share in warning",
            "fraction",
            "24h (288)",
            "vibration_above_warning",
            "L1",
            "vibration_above_warning_ratio_24h",
        ),
        (
            "days_since_maintenance",
            "maintenance",
            "Days since last maintenance",
            "days",
            "past date",
            "maintenance",
            "L2",
            "days_since_maintenance",
        ),
        (
            "days_since_inspection",
            "maintenance",
            "Days since last inspection",
            "days",
            "past date",
            "maintenance",
            "L2",
            "days_since_inspection",
        ),
        (
            "days_since_preventive",
            "maintenance",
            "Days since last PM",
            "days",
            "past date",
            "maintenance",
            "L2",
            "days_since_preventive",
        ),
        (
            "operating_hours_since_inspection",
            "maintenance",
            "Operating hours since inspection",
            "hours",
            "past date",
            "operating_hours",
            "L2",
            "operating_hours_since_inspection",
        ),
        (
            "operating_hours_since_preventive",
            "maintenance",
            "Operating hours since PM",
            "hours",
            "past date",
            "operating_hours",
            "L2",
            "operating_hours_since_preventive",
        ),
        (
            "hours_since_estimated",
            "maintenance",
            "1 if hours since insp/PM estimated",
            "binary",
            "past date",
            "maintenance",
            "L2",
            "hours_since_estimated",
        ),
        (
            "inspection_overdue",
            "maintenance",
            "Operating hours since insp >= 500h",
            "boolean",
            "past date",
            "operating_hours",
            "L2",
            "inspection_overdue",
        ),
        (
            "preventive_overdue",
            "maintenance",
            "Operating hours since PM >= 2000h",
            "boolean",
            "past date",
            "operating_hours",
            "L2",
            "preventive_overdue",
        ),
        (
            "maintenance_count",
            "maintenance",
            "Count of past maintenance events",
            "count",
            "past date",
            "maintenance",
            "L2",
            "maintenance_count",
        ),
        (
            "maintenance_count_90d",
            "maintenance",
            "Events count trailing 90d",
            "count",
            "90d trailing",
            "maintenance",
            "L2",
            "maintenance_count_90d",
        ),
        (
            "maintenance_cost_avg",
            "maintenance",
            "Expanding mean cost of past events",
            "GBP",
            "past date",
            "maintenance",
            "L2",
            "maintenance_cost_avg",
        ),
        (
            "maintenance_cost_90d",
            "maintenance",
            "Sum of costs in trailing 90d",
            "GBP",
            "90d trailing",
            "maintenance",
            "L2",
            "maintenance_cost_90d",
        ),
        (
            "failure_count",
            "maintenance",
            "Count of past Corrective failures",
            "count",
            "past date",
            "maintenance",
            "L2",
            "failure_count",
        ),
        (
            "days_since_failure",
            "maintenance",
            "Days since last failure",
            "days",
            "past date",
            "maintenance",
            "L2",
            "days_since_failure",
        ),
        (
            "previous_failure",
            "maintenance",
            "1 if failure in trailing 90d",
            "binary",
            "90d trailing",
            "maintenance",
            "L2",
            "previous_failure",
        ),
    ]

    cols = [
        "name",
        "group",
        "definition",
        "unit",
        "window",
        "source_columns",
        "leakage_rule",
        "doc_name",
    ]
    df_cat = pd.DataFrame(specs, columns=cols)

    # Save CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "feature_catalog.csv"
    df_cat.to_csv(csv_path, index=False)

    # Generate Markdown documentation
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / "feature_catalog.md"

    md_lines = [
        "# Feature Catalog",
        "",
        "**Project:** AI Operations Intelligence Platform  ",
        "**Pipeline Phase:** Day 03 (Data Engineering & Feature Engineering)  ",
        "**Target Datasets:** `sensor_features.parquet` and `operations_dataset.parquet`  ",
        "",
        "## How to Read This Catalog",
        "",
        "Every engineered feature strictly complies with temporal leakage prevention rules:",
        "- **L1 (Trailing Only):** Features use strictly current and past data ($t' \\le t$).",
        "- **L2 (Strictly Past Events):** Maintenance history dated $D$ is visible from $D + 1$.",
        "- **L3 (No Global Statistics):** All statistics are from local trailing windows.",
        "- **L4 (Label Isolation):** Failure labels reside only in `anomaly_eval_labels.parquet`.",
        "- **L5 (Forward Imputation):** Short gaps (<= 15m) filled and flagged via `is_imputed`.",
        "",
        "## Feature Inventory",
        "",
        "| Feature Name | Group | Definition | Unit | Window | Leakage Rule | Doc Name |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for _, r in df_cat.iterrows():
        md_lines.append(
            f"| `{r['name']}` | {r['group']} | {r['definition']} | {r['unit']} | "
            f"{r['window']} | {r['leakage_rule']} | `{r['doc_name']}` |"
        )

    md_lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info("Saved feature catalog to %s and %s", csv_path, md_path)
    return df_cat
