"""Production Dataset Generator for AI Operations Intelligence Platform.

Generates data/raw/unified/production_raw.csv (2,556 machine-days)
grounded directly in sensors_raw.csv and maintenance_raw.csv.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ml.preprocessing.config import I_OFF, SEED, TIMELINE_END, TIMELINE_START, UNIFIED
from ml.preprocessing.fleet import FLEET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DAILY_TARGETS: dict[str, int] = {
    "Air Compressor": 600,
    "Hydraulic Press": 1200,
    "Conveyor Drive": 2400,
}


def _deterministic_uuid(key: str) -> str:
    """Generate deterministic UUIDv5."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def generate_production_records() -> pd.DataFrame:
    """Generate daily production records for each machine in the fleet."""
    rng = np.random.Generator(np.random.PCG64(SEED))

    sensors_path = UNIFIED / "sensors_raw.csv"
    maint_path = UNIFIED / "maintenance_raw.csv"

    if not sensors_path.exists():
        raise FileNotFoundError(f"Missing {sensors_path}. Run generate_sensors.py first.")
    if not maint_path.exists():
        raise FileNotFoundError(f"Missing {maint_path}. Run generate_maintenance.py first.")

    logger.info("Reading sensors_raw.csv for operating hours aggregation...")
    sensors_df = pd.read_csv(
        sensors_path,
        usecols=["timestamp", "machine_id", "motor_current"],
    )
    sensors_df["timestamp"] = pd.to_datetime(sensors_df["timestamp"], utc=True)
    sensors_df["date"] = sensors_df["timestamp"].dt.date
    sensors_df["is_running"] = sensors_df["motor_current"] > I_OFF

    # Daily operating hours per machine-day: sum of 5-min intervals
    daily_ops = (
        sensors_df.groupby(["machine_id", "date"])["is_running"]
        .sum()
        .reset_index(name="running_intervals")
    )
    daily_ops["operating_hours"] = (daily_ops["running_intervals"] * (5.0 / 60.0)).round(2)

    logger.info("Reading maintenance_raw.csv for downtime reason mapping...")
    maint_df = pd.read_csv(maint_path, usecols=["machine_id", "performed_date", "maintenance_type"])
    maint_df["date"] = pd.to_datetime(maint_df["performed_date"], utc=True).dt.date
    maint_map: dict[tuple[str, Any], str] = {}
    for _, row in maint_df.iterrows():
        key = (row["machine_id"], row["date"])
        maint_map[key] = str(row["maintenance_type"])

    machine_types = {m.machine_id: m.machine_type for m in FLEET}
    all_dates = [d.date() for d in pd.date_range(TIMELINE_START, TIMELINE_END, freq="D")]

    # Pre-index daily operating hours
    ops_lookup = {
        (row["machine_id"], row["date"]): row["operating_hours"]
        for _, row in daily_ops.iterrows()
    }

    records: list[dict[str, Any]] = []

    for m in FLEET:
        m_id = m.machine_id
        m_type = machine_types[m_id]
        target = DAILY_TARGETS[m_type]

        for cur_date in all_dates:
            op_h = float(ops_lookup.get((m_id, cur_date), 0.0))
            downtime_h = round(max(0.0, 24.0 - op_h), 2)

            # Determine downtime reason
            maint_event = maint_map.get((m_id, cur_date))
            if maint_event == "corrective" or maint_event == "emergency":
                downtime_reason = "Unscheduled Outage"
            elif maint_event in ["preventive", "inspection"]:
                downtime_reason = "Scheduled Maintenance"
            elif op_h >= 20.0:
                downtime_reason = "None"
            else:
                downtime_reason = "Idle"

            # Production units calculation
            if op_h <= 0.5:
                units_produced = 0
                scrap_units = 0
            else:
                # Stochastic efficiency factor [0.95, 1.02]
                efficiency = float(rng.uniform(0.96, 1.01))
                avail = op_h / 24.0
                raw_units = int(round(target * avail * efficiency))
                units_produced = max(1, min(target, raw_units))

                # Scrap calculation
                # Healthy baseline scrap rate ~ 1.5% to 2.5%
                base_scrap_rate = float(rng.uniform(0.015, 0.025))

                # M-104 Hero Degradation: Scrap elevation in final 14 days
                days_from_end = (TIMELINE_END - cur_date).days
                if m_id == "M-104" and days_from_end <= 14:
                    # Degradation climbs from baseline to ~10.5%
                    progress = (14 - days_from_end) / 14.0
                    elevated_scrap_rate = base_scrap_rate + 0.085 * (progress**1.4)
                    scrap_rate = min(0.12, elevated_scrap_rate)
                else:
                    scrap_rate = base_scrap_rate

                scrap_units = int(round(units_produced * scrap_rate))
                scrap_units = min(units_produced, scrap_units)

            # OEE Calculation
            availability = float(np.clip(op_h / 24.0, 0.0, 1.0))
            if op_h <= 0.5 or units_produced == 0:
                performance = 0.0
                quality = 1.0
                oee = 0.0
            else:
                expected_units = target * availability
                performance = float(np.clip(units_produced / max(expected_units, 1e-6), 0.0, 1.0))
                quality = float(np.clip((units_produced - scrap_units) / units_produced, 0.0, 1.0))
                oee = round(availability * performance * quality, 4)

            created_at = datetime(
                cur_date.year, cur_date.month, cur_date.day, 23, 55, tzinfo=UTC
            ).isoformat()

            records.append(
                {
                    "production_id": _deterministic_uuid(f"{m_id}-{cur_date.isoformat()}"),
                    "date": cur_date.isoformat(),
                    "machine_id": m_id,
                    "shift": "daily",
                    "units_produced": units_produced,
                    "units_target": target,
                    "scrap_units": scrap_units,
                    "operating_hours": op_h,
                    "downtime_hours": downtime_h,
                    "downtime_reason": downtime_reason,
                    "oee": oee,
                    "created_at": created_at,
                }
            )

    df_out = pd.DataFrame(records)
    # Sort by date, then machine_id
    df_out = df_out.sort_values(["date", "machine_id"]).reset_index(drop=True)
    return df_out


def run() -> None:
    """Generate and validate production_raw.csv."""
    UNIFIED.mkdir(parents=True, exist_ok=True)
    out_file = UNIFIED / "production_raw.csv"

    df = generate_production_records()
    logger.info("Generated %d production records.", len(df))

    # Acceptance Checks
    # 1. Exactly 2,556 rows (12 machines * 213 days)
    assert len(df) == 2556, f"Expected 2556 rows, got {len(df)}"

    # 2. Date range matches timeline
    dates = pd.to_datetime(df["date"]).dt.date
    assert dates.min() == TIMELINE_START, f"Min date mismatch: {dates.min()}"
    assert dates.max() == TIMELINE_END, f"Max date mismatch: {dates.max()}"

    # 3. OEE strictly in [0.0, 1.0]
    assert ((df["oee"] >= 0.0) & (df["oee"] <= 1.0)).all(), "OEE out of range [0, 1]"

    # 4. Units produced >= scrap units >= 0
    assert (df["units_produced"] >= df["scrap_units"]).all(), "scrap exceeds production"
    assert (df["scrap_units"] >= 0).all(), "negative scrap"

    # 5. Operating hours + downtime hours == 24.0 (within 0.05 rounding)
    hour_sum = df["operating_hours"] + df["downtime_hours"]
    assert (hour_sum - 24.0).abs().max() < 0.05, "Hour balance mismatch"

    # 6. M-104 scrap rate significantly elevated in final 14 days
    m104 = df[df["machine_id"] == "M-104"].copy()
    m104["scrap_rate"] = m104["scrap_units"] / m104["units_produced"]
    m104_early_rate = m104.iloc[:-14]["scrap_rate"].mean()
    m104_late_rate = m104.iloc[-14:]["scrap_rate"].mean()
    logger.info(
        "M-104 Scrap Rate: early=%.4f (%.2f%%), final 14d=%.4f (%.2f%%)",
        m104_early_rate,
        m104_early_rate * 100,
        m104_late_rate,
        m104_late_rate * 100,
    )
    assert m104_late_rate > 2.5 * m104_early_rate, (
        f"Hero degradation scrap not elevated: early={m104_early_rate}, late={m104_late_rate}"
    )

    df.to_csv(out_file, index=False)
    logger.info("Successfully wrote %d production records to %s", len(df), out_file)


if __name__ == "__main__":
    run()
