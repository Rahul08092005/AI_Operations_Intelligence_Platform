"""Sensor data generator for the 12-machine fleet.

Generates 5-minute grid sensor readings:
- M-104: Time-shifted real MetroPT-3 sensor time series with real failure windows
         and exponential hero degradation over the final 35 days.
- Derived machines (M-101..M-103, M-105..M-112): Daily block resampling from healthy
         MetroPT-3 days, per-machine scaling, synthetic vibration/RPM, and scenario overlays.

Outputs:
- data/raw/unified/sensors_raw.csv
- data/raw/unified/timeline_metadata.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from ml.preprocessing.config import (
        RAW,
        REFERENCE,
        SEED,
        TIMELINE_END,
        UNIFIED,
    )
    from ml.preprocessing.fleet import FLEET, MachineConfig
except (ImportError, ModuleNotFoundError):
    from config import (  # type: ignore
        RAW,
        REFERENCE,
        SEED,
        TIMELINE_END,
        UNIFIED,
    )
    from fleet import FLEET, MachineConfig  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
I_OFF = 1.55  # Running threshold from profile
VOLTAGE_FACTOR = 0.59  # 0.59 kW/A for ~400V 3-phase motor


def generate_ar1_noise(n: int, phi: float, sigma_e: float, rng: np.random.Generator) -> np.ndarray:
    """Generate an AR(1) stationary noise series."""
    sigma_w = sigma_e * np.sqrt(1.0 - phi**2)
    w = rng.normal(0.0, sigma_w, size=n)
    e = np.zeros(n, dtype=np.float32)
    e[0] = rng.normal(0.0, sigma_e)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + w[t]
    return e


def load_and_resample_metropt(file_path: Path) -> pd.DataFrame:
    """Load MetroPT and resample to 5-minute grid."""
    logger.info("Loading MetroPT-3 from %s...", file_path)
    usecols = ["timestamp", "TP3", "Oil_temperature", "Motor_current", "COMP"]
    dtypes = {
        "TP3": "float32",
        "Oil_temperature": "float32",
        "Motor_current": "float32",
        "COMP": "float32",
    }
    df = pd.read_csv(file_path, usecols=usecols, dtype=dtypes, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Resample to 5-min intervals
    df_res = df.set_index("timestamp").resample("5min").mean()
    # COMP threshold >= 0.5 gives 1, else 0
    df_res["COMP"] = (df_res["COMP"] >= 0.5).astype(np.float32)
    return df_res.reset_index()


def extract_healthy_day_pool(
    df_5min: pd.DataFrame, failure_windows_path: Path
) -> tuple[dict[date, pd.DataFrame], pd.DataFrame]:
    """Find days with >= 90% coverage and no overlap with failure windows (+3d / -1d)."""
    logger.info("Extracting healthy-day pool from MetroPT...")
    fw = pd.read_csv(failure_windows_path)
    fw["start_time"] = pd.to_datetime(fw["start_time"]) - pd.Timedelta(days=3)
    fw["end_time"] = pd.to_datetime(fw["end_time"]) + pd.Timedelta(days=1)

    df_5min["date"] = df_5min["timestamp"].dt.date
    daily_counts = df_5min.groupby("date").size()
    full_days = daily_counts[daily_counts >= 259].index.tolist()

    healthy_pool: dict[date, pd.DataFrame] = {}
    for d in full_days:
        d_start = pd.Timestamp(d)
        d_end = d_start + pd.Timedelta(days=1)
        overlap = False
        for _, row in fw.iterrows():
            if not (d_end <= row["start_time"] or d_start >= row["end_time"]):
                overlap = True
                break
        if not overlap:
            day_data = df_5min[df_5min["date"] == d].copy()
            # If slightly fewer than 288 rows due to gaps, reindex to exact 288 grid
            expected_times = pd.date_range(d_start, periods=288, freq="5min")
            day_data = day_data.set_index("timestamp").reindex(expected_times)
            day_data["Motor_current"] = day_data["Motor_current"].ffill().bfill()
            day_data["COMP"] = day_data["COMP"].ffill().bfill()
            day_data["TP3"] = day_data["TP3"].ffill().bfill()
            day_data["Oil_temperature"] = day_data["Oil_temperature"].ffill().bfill()
            healthy_pool[d] = day_data.reset_index().rename(columns={"index": "timestamp"})

    logger.info("Found %d healthy days in source pool", len(healthy_pool))
    if len(healthy_pool) < 60:
        raise ValueError(f"Fewer than 60 healthy days found: {len(healthy_pool)}")

    return healthy_pool, fw


def apply_real_failure_windows_overlay(
    df: pd.DataFrame, shifted_windows: pd.DataFrame
) -> pd.DataFrame:
    """Apply real failure window ramps and outages to M-104."""
    for _, row in shifted_windows.iterrows():
        ws = row["start_time"]
        we = row["end_time"]
        precursor_start = ws - pd.Timedelta(hours=72)

        # Precursor: 72 hours before ws
        mask_pre = (df["timestamp"] >= precursor_start) & (df["timestamp"] < ws)
        if mask_pre.any():
            pre_ts = df.loc[mask_pre, "timestamp"]
            pre_ramp = (
                (pre_ts - precursor_start).dt.total_seconds() / (72 * 3600)
            ).to_numpy(dtype=np.float64)
            df.loc[mask_pre, "vibration"] = np.minimum(
                df.loc[mask_pre, "vibration"] * (1.0 + 0.6 * pre_ramp), 4.2
            )
            df.loc[mask_pre, "temperature"] = df.loc[mask_pre, "temperature"] + 3.0 * pre_ramp

        # Inside window
        mask_win = (df["timestamp"] >= ws) & (df["timestamp"] <= we)
        if mask_win.any():
            df.loc[mask_win, "vibration"] = np.minimum(
                df.loc[mask_win, "vibration"] * 1.8, 4.2
            )
            df.loc[mask_win, "rpm"] = df.loc[mask_win, "rpm"] * 0.97

    return df


def apply_hero_degradation(
    df: pd.DataFrame,
    end_date: date,
    degrade_days: int = 35,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Apply exponential degradation over final 35 days for M-104."""
    start_degrade = pd.Timestamp(end_date) - pd.Timedelta(days=degrade_days)
    mask = df["timestamp"] >= start_degrade
    if not mask.any():
        return df

    # Ensure prior to degradation vibration is strictly capped at 4.2 mm/s
    pre_mask = df["timestamp"] < start_degrade
    df.loc[pre_mask, "vibration"] = np.minimum(df.loc[pre_mask, "vibration"], 4.2)

    k = 3.0
    ts_degrade = df.loc[mask, "timestamp"]
    total_sec = degrade_days * 86400
    s = ((ts_degrade - start_degrade).dt.total_seconds() / total_sec).to_numpy(dtype=np.float64)
    s = np.clip(s, 0.0, 1.0)
    ramp = (np.exp(k * s) - 1.0) / (np.exp(k) - 1.0)

    # Calculate baseline healthy mean vibration before degradation
    v_healthy_mean = float(df.loc[~mask, "vibration"].mean())
    df.loc[mask, "vibration"] = df.loc[mask, "vibration"] + (7.8 - v_healthy_mean) * ramp

    # Temperature elevation: ensure last 24h exceeds healthy 95th percentile by >= 8°C
    t_healthy = df.loc[~mask, "temperature"].dropna()
    t95 = float(np.percentile(t_healthy, 95)) if len(t_healthy) > 0 else 75.0
    last_24h_mask = mask & (df["timestamp"] >= pd.Timestamp(end_date) - pd.Timedelta(days=1))
    raw_last_day_temp = float(df.loc[last_24h_mask, "temperature"].mean())
    # Average ramp over final 24h is ~0.88; divide target lift by 0.88 to achieve desired mean
    needed_lift = max(11.0, (t95 + 9.8) - raw_last_day_temp)
    temp_elevation = needed_lift / 0.88

    df.loc[mask, "temperature"] = np.minimum(
        df.loc[mask, "temperature"] + temp_elevation * ramp, 84.8
    )
    df.loc[mask, "energy_consumption"] = df.loc[mask, "energy_consumption"] * (1.0 + 0.12 * ramp)
    df.loc[mask, "pressure"] = df.loc[mask, "pressure"] - 0.15 * ramp
    df.loc[mask, "rpm"] = df.loc[mask, "rpm"] * (1.0 - 0.03 * ramp)

    # Additional noise growth on vibration
    if rng is not None:
        noise_extra = rng.normal(0, 0.25, size=len(s)) * ramp
        df.loc[mask, "vibration"] = df.loc[mask, "vibration"] + noise_extra

    # Before 13.0 days before end, ensure vibration is capped at 4.40 mm/s so first crossing
    # is strictly between 7 and 14 days before timeline end
    target_cross_start = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(days=13)
    pre_cross_mask = df["timestamp"] < target_cross_start
    df.loc[pre_cross_mask, "vibration"] = np.minimum(df.loc[pre_cross_mask, "vibration"], 4.40)

    return df


def apply_mild_drift(df: pd.DataFrame, end_date: date, drift_days: int = 21) -> pd.DataFrame:
    """Apply mild linear drift for M-103 over final 21 days."""
    start_drift = pd.Timestamp(end_date) - pd.Timedelta(days=drift_days)
    mask = df["timestamp"] >= start_drift
    if not mask.any():
        return df

    ts_drift = df.loc[mask, "timestamp"]
    total_sec = drift_days * 86400
    s = ((ts_drift - start_drift).dt.total_seconds() / total_sec).to_numpy(dtype=np.float64)
    s = np.clip(s, 0.0, 1.0)

    # Bring final weekly mean into [4.5, 5.8] band
    baseline_vib = float(df.loc[~mask, "vibration"].mean())
    df.loc[mask, "vibration"] = df.loc[mask, "vibration"] + (5.3 - baseline_vib) * s
    df.loc[mask, "temperature"] = df.loc[mask, "temperature"] + 4.0 * s
    return df


def apply_failure_event(
    df: pd.DataFrame,
    center_time: pd.Timestamp,
    precursor_days: int = 3,
    outage_hours: int = 24,
) -> pd.DataFrame:
    """Apply generic failure event: 3d precursor, 24h machine outage."""
    precursor_start = center_time - pd.Timedelta(days=precursor_days)
    outage_end = center_time + pd.Timedelta(hours=outage_hours)

    # Precursor
    mask_pre = (df["timestamp"] >= precursor_start) & (df["timestamp"] < center_time)
    if mask_pre.any():
        s = (
            (df.loc[mask_pre, "timestamp"] - precursor_start).dt.total_seconds()
            / (precursor_days * 86400)
        ).to_numpy(dtype=np.float64)
        df.loc[mask_pre, "vibration"] = df.loc[mask_pre, "vibration"] * (1.0 + 0.7 * s)
        df.loc[mask_pre, "temperature"] = df.loc[mask_pre, "temperature"] + 3.0 * s

    # Outage: machine turned off
    mask_out = (df["timestamp"] >= center_time) & (df["timestamp"] < outage_end)
    if mask_out.any():
        df.loc[mask_out, "motor_current"] = 0.02
        df.loc[mask_out, "comp_state"] = 0
        df.loc[mask_out, "vibration"] = 0.3
        df.loc[mask_out, "rpm"] = 0.0
        df.loc[mask_out, "energy_consumption"] = 0.0

    return df


def apply_recently_maintained(df: pd.DataFrame, maint_time: pd.Timestamp) -> pd.DataFrame:
    """M-106 drift up to ~4.0 mm/s over 30d, drop to baseline on maint_time."""
    start_drift = maint_time - pd.Timedelta(days=30)
    mask_pre = (df["timestamp"] >= start_drift) & (df["timestamp"] < maint_time)
    if mask_pre.any():
        s = (
            (df.loc[mask_pre, "timestamp"] - start_drift).dt.total_seconds() / (30 * 86400)
        ).to_numpy(dtype=np.float64)
        df.loc[mask_pre, "vibration"] = df.loc[mask_pre, "vibration"] + 1.5 * s
    return df


def apply_benign_spikes(
    df: pd.DataFrame, rng: np.random.Generator, num_spikes: int = 42
) -> pd.DataFrame:
    """M-109 ~6 short vibration spikes per month (1-3 intervals, multiplier 1.8)."""
    n_rows = len(df)
    spike_indices = rng.choice(n_rows - 5, size=num_spikes, replace=False)
    for idx in spike_indices:
        duration = rng.integers(1, 4)
        for step in range(duration):
            if idx + step < n_rows:
                df.at[idx + step, "vibration"] = min(df.at[idx + step, "vibration"] * 1.8, 4.45)
    return df


def apply_in_maintenance(df: pd.DataFrame, end_time: pd.Timestamp, hours: int = 36) -> pd.DataFrame:
    """M-112 offline for the last 36 hours."""
    start_maint = end_time - pd.Timedelta(hours=hours)
    mask = df["timestamp"] >= start_maint
    if mask.any():
        df.loc[mask, "motor_current"] = 0.02
        df.loc[mask, "comp_state"] = 0
        df.loc[mask, "vibration"] = 0.3
        df.loc[mask, "rpm"] = 0.0
        df.loc[mask, "energy_consumption"] = 0.0
    return df


def generate_machine_sensors(
    config: MachineConfig,
    machine_idx: int,
    df_metro_5min: pd.DataFrame,
    healthy_pool: dict[date, pd.DataFrame],
    shifted_windows: pd.DataFrame,
    target_dates: list[date],
    shift_days: int,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, dict]:
    """Generate the full sensor series for one machine."""
    logger.info(
        "Generating sensors for %s (%s, %s)...",
        config.machine_id,
        config.machine_type,
        config.scenario,
    )
    rng = np.random.default_rng(SEED + machine_idx * 1000)

    # Multipliers per type
    type_mult = {
        "Air Compressor": 1.0,
        "Hydraulic Press": 1.1,
        "Conveyor Drive": 0.9,
    }.get(config.machine_type, 1.0)

    nominal_rpm = {
        "Air Compressor": 1480.0,
        "Hydraulic Press": 1450.0,
        "Conveyor Drive": 960.0,
    }.get(config.machine_type, 1480.0)

    metadata_entry: dict = {}

    if config.data_origin == "metropt_real_signals":
        # M-104: Real MetroPT series
        df = df_metro_5min.copy()
        df["timestamp"] = df["timestamp"] + pd.Timedelta(days=shift_days)
        # Filter to target window bounds
        window_end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
        df = df[
            (df["timestamp"] >= pd.Timestamp(start_date)) & (df["timestamp"] <= window_end_ts)
        ].copy()
        df = df.rename(columns={"Oil_temperature": "temperature", "TP3": "pressure"})
        df["temperature"] = df["temperature"].ffill().bfill()
        df["pressure"] = df["pressure"].ffill().bfill()
        df["Motor_current"] = df["Motor_current"].ffill().bfill()
        df["COMP"] = df["COMP"].ffill().bfill()
    else:
        # Derived machine: daily block resampling from healthy pool
        current_scale = float(rng.uniform(0.93, 1.07))
        temp_scale = float(rng.uniform(0.95, 1.05))
        temp_offset = float(rng.uniform(-4.0, 4.0))
        pressure_scale = float(rng.uniform(0.97, 1.03))

        metadata_entry["transforms"] = {
            "current_scale": round(current_scale, 4),
            "temp_scale": round(temp_scale, 4),
            "temp_offset": round(temp_offset, 4),
            "pressure_scale": round(pressure_scale, 4),
        }

        # Index healthy pool by weekday
        pool_by_weekday: dict[int, list[pd.DataFrame]] = {w: [] for w in range(7)}
        for d, day_df in healthy_pool.items():
            pool_by_weekday[d.weekday()].append(day_df)

        daily_dfs: list[pd.DataFrame] = []
        for target_d in target_dates:
            target_ts = pd.Timestamp(target_d)
            wd = target_d.weekday()
            candidates = pool_by_weekday[wd] or list(healthy_pool.values())
            chosen_day = candidates[rng.integers(0, len(candidates))].copy()

            # Align timestamps to target date
            chosen_day["timestamp"] = pd.date_range(target_ts, periods=288, freq="5min")
            # Apply fixed scaling
            chosen_day["Motor_current"] *= current_scale
            chosen_day["Oil_temperature"] = chosen_day["Oil_temperature"] * temp_scale + temp_offset
            chosen_day["TP3"] *= pressure_scale
            daily_dfs.append(chosen_day)

        df = pd.concat(daily_dfs, ignore_index=True)
        df = df.rename(columns={"Oil_temperature": "temperature", "TP3": "pressure"})

    df = df.rename(columns={"Motor_current": "motor_current", "COMP": "comp_state"})
    df["machine_id"] = config.machine_id
    for col_name in ["temperature", "pressure", "motor_current", "comp_state"]:
        if col_name in df.columns:
            df[col_name] = df[col_name].astype(np.float64)

    # Step 3: Base synthetic signals
    running_mask = df["motor_current"] > I_OFF
    running_currents = df.loc[running_mask, "motor_current"]
    i_ref = float(np.percentile(running_currents, 95)) if len(running_currents) > 0 else 7.0
    t_ref = float(df["temperature"].median())

    l_t = np.clip(df["motor_current"] / i_ref, 0.0, 1.2)
    n_rows = len(df)
    ar1_noise = generate_ar1_noise(n_rows, phi=0.95, sigma_e=0.2, rng=rng)

    # Base vibration
    base_vib = 1.6 + 1.2 * l_t + 0.02 * (df["temperature"] - t_ref) + ar1_noise
    base_vib = np.maximum(base_vib, 0.3)
    # When off, vibration is near 0.3
    off_noise = np.abs(rng.normal(0.0, 0.02, size=n_rows))
    df["vibration"] = np.where(running_mask, base_vib * type_mult, 0.3 + off_noise)

    # Base RPM
    rpm_noise = rng.normal(0.0, 0.004 * nominal_rpm, size=n_rows)
    df["rpm"] = np.where(running_mask, nominal_rpm * (1.0 - 0.02 * l_t) + rpm_noise, 0.0)

    # Base Energy consumption (kWh)
    df["energy_consumption"] = VOLTAGE_FACTOR * df["motor_current"] * (5.0 / 60.0)

    # Initial operating hours
    util_m = float(rng.uniform(0.55, 0.80))
    window_start_dt = pd.Timestamp(start_date)
    inst_dt = pd.Timestamp(config.installation_date)
    prior_days = max((window_start_dt - inst_dt).days, 0)
    initial_hours = prior_days * 24.0 * util_m
    metadata_entry["initial_operating_hours"] = round(initial_hours, 2)
    metadata_entry["utilization_factor"] = round(util_m, 4)

    # Cumulative operating hours
    df["operating_hours"] = initial_hours + (df["motor_current"] > I_OFF).astype(int).cumsum() * (
        5.0 / 60.0
    )

    # Step 4: Scenario Overlays
    failure_events: list[str] = []
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)

    if config.scenario == "hero_degrading":
        df = apply_real_failure_windows_overlay(df, shifted_windows)
        df = apply_hero_degradation(df, end_date=end_date, degrade_days=35, rng=rng)
        metadata_entry["window_overlay_applied"] = True
    elif config.scenario == "mild_drift":
        df = apply_mild_drift(df, end_date=end_date, drift_days=21)
    elif config.scenario == "past_failure_old":
        f_center = end_ts - pd.Timedelta(days=120)
        df = apply_failure_event(df, f_center, precursor_days=3, outage_hours=24)
        failure_events.append(str(f_center.date()))
    elif config.scenario == "past_failure_mid":
        f_center = end_ts - pd.Timedelta(days=75)
        df = apply_failure_event(df, f_center, precursor_days=3, outage_hours=24)
        failure_events.append(str(f_center.date()))
    elif config.scenario == "past_failure_recent":
        f_center = end_ts - pd.Timedelta(days=58)
        df = apply_failure_event(df, f_center, precursor_days=3, outage_hours=24)
        failure_events.append(str(f_center.date()))
    elif config.scenario == "recently_maintained":
        m_date = end_ts - pd.Timedelta(days=5)
        df = apply_recently_maintained(df, maint_time=m_date)
    elif config.scenario == "benign_spikes":
        df = apply_benign_spikes(df, rng=rng)
    elif config.scenario == "in_maintenance":
        df = apply_in_maintenance(df, end_time=end_ts, hours=36)

    # After overlays: recompute dependent signals
    df["energy_consumption"] = VOLTAGE_FACTOR * df["motor_current"] * (5.0 / 60.0)
    df["operating_hours"] = initial_hours + (df["motor_current"] > I_OFF).astype(int).cumsum() * (
        5.0 / 60.0
    )

    metadata_entry["failure_events"] = failure_events
    return df, metadata_entry


def generate_all_sensors() -> None:
    """Generate sensors for the entire fleet and save outputs."""
    metro_csv = list((RAW / "metropt").glob("*.csv"))[0]
    fw_csv = REFERENCE / "metropt_failure_windows.csv"

    # Step 1: Load and reduce MetroPT
    df_metro = load_and_resample_metropt(metro_csv)

    # Determine date range & shift
    max_date = df_metro["timestamp"].dt.date.max()
    min_date = df_metro["timestamp"].dt.date.min()

    # Shift calculation per D20: whole weeks ending on TIMELINE_END
    # If MetroPT ends on 2020-09-01, shifting by 320 weeks lands its last full day (2020-08-31)
    # on TIMELINE_END = 2026-10-19.
    diff_days = (TIMELINE_END - max_date).days
    shift_days = 7 * (diff_days // 7)
    if (max_date + pd.Timedelta(days=shift_days)) < TIMELINE_END:
        # Align last full day to TIMELINE_END
        shift_days += 7

    shifted_max = max_date + pd.Timedelta(days=shift_days)
    shifted_min = min_date + pd.Timedelta(days=shift_days)

    logger.info("Timeline Shift: %d days (%d weeks)", shift_days, shift_days // 7)
    logger.info("Original Range: %s to %s", min_date, max_date)
    logger.info(
        "Shifted Range : %s to %s (Target End: %s)", shifted_min, shifted_max, TIMELINE_END
    )

    # Shifted failure windows
    fw = pd.read_csv(fw_csv)
    shifted_fw = fw.copy()
    shifted_fw["start_time"] = pd.to_datetime(shifted_fw["start_time"]) + pd.Timedelta(
        days=shift_days
    )
    shifted_fw["end_time"] = pd.to_datetime(shifted_fw["end_time"]) + pd.Timedelta(
        days=shift_days
    )

    # Extract healthy pool
    healthy_pool, _ = extract_healthy_day_pool(df_metro, fw_csv)

    # Target dates for derived machines: from shifted_min to TIMELINE_END
    cur = shifted_min
    target_dates: list[date] = []
    while cur <= TIMELINE_END:
        target_dates.append(cur)
        cur += timedelta(days=1)

    all_dfs: list[pd.DataFrame] = []
    metadata: dict = {
        "shift_days": shift_days,
        "shift_weeks": shift_days // 7,
        "timeline_start": str(shifted_min),
        "timeline_end": str(TIMELINE_END),
        "I_OFF": I_OFF,
        "machines": {},
    }

    for idx, machine in enumerate(FLEET):
        df_m, m_meta = generate_machine_sensors(
            config=machine,
            machine_idx=idx,
            df_metro_5min=df_metro,
            healthy_pool=healthy_pool,
            shifted_windows=shifted_fw,
            target_dates=target_dates,
            shift_days=shift_days,
            start_date=shifted_min,
            end_date=TIMELINE_END,
        )
        all_dfs.append(df_m)
        metadata["machines"][machine.machine_id] = m_meta

    full_sensors = pd.concat(all_dfs, ignore_index=True)

    # Output columns and ordering per contract
    contract_cols = [
        "timestamp",
        "machine_id",
        "temperature",
        "vibration",
        "pressure",
        "rpm",
        "energy_consumption",
        "operating_hours",
        "motor_current",
        "comp_state",
    ]
    full_sensors = full_sensors[contract_cols].sort_values(["machine_id", "timestamp"])

    # Rounding decimals per contract
    full_sensors["temperature"] = full_sensors["temperature"].round(2)
    full_sensors["vibration"] = full_sensors["vibration"].round(3)
    full_sensors["pressure"] = full_sensors["pressure"].round(3)
    full_sensors["rpm"] = full_sensors["rpm"].round(1)
    full_sensors["energy_consumption"] = full_sensors["energy_consumption"].round(4)
    full_sensors["operating_hours"] = full_sensors["operating_hours"].round(2)
    full_sensors["motor_current"] = full_sensors["motor_current"].round(3)
    full_sensors["comp_state"] = full_sensors["comp_state"].astype(int)

    # Format timestamp as ISO 8601 UTC string
    full_sensors["timestamp"] = full_sensors["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Save outputs
    out_csv = UNIFIED / "sensors_raw.csv"
    out_json = UNIFIED / "timeline_metadata.json"

    logger.info("Writing sensors to %s (%d rows)...", out_csv, len(full_sensors))
    full_sensors.to_csv(out_csv, index=False)

    logger.info("Writing metadata to %s...", out_json)
    out_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Sensor generation complete!")


if __name__ == "__main__":
    generate_all_sensors()
