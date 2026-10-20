"""Maintenance History Generator for AI Operations Intelligence Platform.

Generates data/raw/unified/maintenance_raw.csv grounded in operating hours
and fleet scenario definitions.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ml.preprocessing.config import SEED, UNIFIED
from ml.preprocessing.fleet import FLEET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PARTS_CATALOG: dict[str, list[str]] = {
    "Air Compressor": [
        "Air Intake Filter",
        "Oil Separator Cartridge",
        "Discharge Valve Kit",
        "Check Valve Gasket",
        "Synthetic Lubricant ISO VG 68",
        "Thermostatic Valve Seal",
        "Pressure Relief Valve",
    ],
    "Hydraulic Press": [
        "Hydraulic Oil ISO VG 46",
        "Return Line Filter Cartridge",
        "Piston Rod Seal Kit",
        "Proportional Relief Valve",
        "Directional Control Valve",
        "High-Pressure Hose Assembly",
        "Pressure Transducer",
    ],
    "Conveyor Drive": [
        "Drive Roller Bearings",
        "Gearbox Oil Seal",
        "Heavy Duty Timing Belt",
        "VFD Inverter Module",
        "Motor Cooling Fan",
        "Coupling Polyurethane Insert",
        "Drive Sprocket Assembly",
    ],
}


def _deterministic_uuid(key: str) -> str:
    """Generate a deterministic UUIDv5 string."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def _sample_lognormal(
    rng: np.random.Generator, low: float, high: float, mean_target: float
) -> float:
    """Sample a lognormal value bounded approximately between low and high."""
    log_mean = np.log(mean_target)
    sigma = (np.log(high) - np.log(low)) / 3.29  # 90% CI roughly
    val = float(rng.lognormal(log_mean, sigma))
    return float(np.clip(val, low, high))


def generate_maintenance_records() -> pd.DataFrame:
    """Generate realistic maintenance records for all fleet machines."""
    rng = np.random.Generator(np.random.PCG64(SEED))

    sensors_path = UNIFIED / "sensors_raw.csv"
    if not sensors_path.exists():
        raise FileNotFoundError(f"Missing {sensors_path}. Run generate_sensors.py first.")

    logger.info("Reading operating hours and timestamps from sensors_raw.csv...")
    sensor_df = pd.read_csv(
        sensors_path,
        usecols=["timestamp", "machine_id", "operating_hours", "comp_state"],
    )
    sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"], utc=True)

    records: list[dict[str, Any]] = []

    # Map machine_id to machine_type
    machine_types = {m.machine_id: m.machine_type for m in FLEET}

    # Step 1: Specific Scenario Anchors
    logger.info("Generating scenario-specific maintenance records...")

    # M-104: Inspection on 2026-09-20 (Day 183)
    # Technician noted: "Minor air leak suspected on discharge manifold.
    # Recommended follow-up in 30 days. Operating parameters within tolerance."
    # Follow-up PM was NOT scheduled (smoking gun).
    m104_date = datetime(2026, 9, 20, 10, 15, tzinfo=UTC)
    records.append(
        {
            "maintenance_id": _deterministic_uuid("M-104-hero-inspection-2026-09-20"),
            "machine_id": "M-104",
            "maintenance_type": "inspection",
            "scheduled_date": datetime(2026, 9, 15, 8, 0, tzinfo=UTC).isoformat(),
            "performed_date": m104_date.isoformat(),
            "duration_hours": 2.50,
            "cost_gbp": 245.00,
            "parts_replaced": json.dumps([]),
            "technician_notes": (
                "Minor air leak suspected on discharge manifold. Recommended "
                "follow-up in 30 days. Operating parameters within tolerance."
            ),
            "root_cause": "",
            "failure_mode": "",
            "created_at": m104_date.isoformat(),
        }
    )

    # M-106: Recently had PM on 2026-10-05 (14 days before timeline end)
    m106_date = datetime(2026, 10, 5, 8, 30, tzinfo=UTC)
    m106_parts = ["Hydraulic Oil ISO VG 46", "Return Line Filter Cartridge", "Piston Rod Seal Kit"]
    records.append(
        {
            "maintenance_id": _deterministic_uuid("M-106-pm-2026-10-05"),
            "machine_id": "M-106",
            "maintenance_type": "preventive",
            "scheduled_date": datetime(2026, 9, 28, 8, 0, tzinfo=UTC).isoformat(),
            "performed_date": m106_date.isoformat(),
            "duration_hours": 6.50,
            "cost_gbp": 1450.00,
            "parts_replaced": json.dumps(m106_parts),
            "technician_notes": (
                "Full 2000h scheduled service completed. Replaced hydraulic fluid, "
                "main filters, and cylinder seals. Pressure and temperature within normal range."
            ),
            "root_cause": "",
            "failure_mode": "",
            "created_at": m106_date.isoformat(),
        }
    )

    # M-112: In maintenance on 2026-10-18 08:00
    m112_date = datetime(2026, 10, 18, 8, 0, tzinfo=UTC)
    m112_parts = ["Drive Roller Bearings", "Gearbox Oil Seal", "Heavy Duty Timing Belt"]
    records.append(
        {
            "maintenance_id": _deterministic_uuid("M-112-in-maintenance-2026-10-18"),
            "machine_id": "M-112",
            "maintenance_type": "preventive",
            "scheduled_date": datetime(2026, 10, 15, 8, 0, tzinfo=UTC).isoformat(),
            "performed_date": m112_date.isoformat(),
            "duration_hours": 36.00,
            "cost_gbp": 3200.00,
            "parts_replaced": json.dumps(m112_parts),
            "technician_notes": (
                "Ongoing overhaul of conveyor drive gearbox. Estimated completion 2026-10-19 20:00."
            ),
            "root_cause": "",
            "failure_mode": "",
            "created_at": m112_date.isoformat(),
        }
    )

    # Past failures aligned with simulated outages:
    # M-105: 2026-06-21
    m105_date = datetime(2026, 6, 21, 8, 0, tzinfo=UTC)
    m105_parts = ["Piston Rod Seal Kit", "Proportional Relief Valve"]
    records.append(
        {
            "maintenance_id": _deterministic_uuid("M-105-corrective-2026-06-21"),
            "machine_id": "M-105",
            "maintenance_type": "corrective",
            "scheduled_date": "",
            "performed_date": m105_date.isoformat(),
            "duration_hours": 24.00,
            "cost_gbp": 4250.00,
            "parts_replaced": json.dumps(m105_parts),
            "technician_notes": (
                "Emergency shutdown following hydraulic pressure drop. Replaced blown seals "
                "and faulty relief valve. Hydraulic pressure restored to nominal 180 bar."
            ),
            "root_cause": (
                "Loss of hydraulic pressure due to seal blowout and valve spool sticking."
            ),
            "failure_mode": "Pressure Drop",
            "created_at": m105_date.isoformat(),
        }
    )

    # M-110: 2026-08-05
    m110_date = datetime(2026, 8, 5, 9, 30, tzinfo=UTC)
    m110_parts = ["Drive Roller Bearings", "Coupling Polyurethane Insert"]
    records.append(
        {
            "maintenance_id": _deterministic_uuid("M-110-corrective-2026-08-05"),
            "machine_id": "M-110",
            "maintenance_type": "corrective",
            "scheduled_date": "",
            "performed_date": m110_date.isoformat(),
            "duration_hours": 24.00,
            "cost_gbp": 3800.00,
            "parts_replaced": json.dumps(m110_parts),
            "technician_notes": (
                "Drive motor tripped on overcurrent. Disassembly revealed bearings seized "
                "from mechanical overstrain. Replaced bearings and flexible coupling."
            ),
            "root_cause": "Drive motor bearing fatigue seizure leading to mechanical overstrain.",
            "failure_mode": "Overstrain Failure",
            "created_at": m110_date.isoformat(),
        }
    )

    # M-111: 2026-08-22
    m111_date = datetime(2026, 8, 22, 11, 0, tzinfo=UTC)
    m111_parts = ["VFD Inverter Module", "Motor Cooling Fan"]
    records.append(
        {
            "maintenance_id": _deterministic_uuid("M-111-corrective-2026-08-22"),
            "machine_id": "M-111",
            "maintenance_type": "corrective",
            "scheduled_date": "",
            "performed_date": m111_date.isoformat(),
            "duration_hours": 24.00,
            "cost_gbp": 5100.00,
            "parts_replaced": json.dumps(m111_parts),
            "technician_notes": (
                "Thermal shutdown alarm triggered. Found cooling fan motor seized, resulting in "
                "thermal overload and IGBT breakdown in drive inverter. Replaced fan and module."
            ),
            "root_cause": "Heat dissipation failure caused by cooling fan motor burnout.",
            "failure_mode": "Heat Dissipation Failure",
            "created_at": m111_date.isoformat(),
        }
    )

    # M-107: Overdue maintenance anchor
    # Last PM was done early in timeline at ~33,300 operating hours (2026-04-12)
    # Next PM was due at 35,300 hours, but current hours is ~35,786 (overdue by > 480 hours!)
    m107_date = datetime(2026, 4, 12, 8, 0, tzinfo=UTC)
    m107_parts = ["Return Line Filter Cartridge", "Hydraulic Oil ISO VG 46"]
    records.append(
        {
            "maintenance_id": _deterministic_uuid("M-107-pm-2026-04-12"),
            "machine_id": "M-107",
            "maintenance_type": "preventive",
            "scheduled_date": datetime(2026, 4, 7, 8, 0, tzinfo=UTC).isoformat(),
            "performed_date": m107_date.isoformat(),
            "duration_hours": 5.50,
            "cost_gbp": 1250.00,
            "parts_replaced": json.dumps(m107_parts),
            "technician_notes": (
                "Routine 2000h service performed at 33,350 operating hours. Hydraulic fluid filter "
                "replaced. Next full service due at 35,350 operating hours."
            ),
            "root_cause": "",
            "failure_mode": "",
            "created_at": m107_date.isoformat(),
        }
    )

    # Step 2: Realistic routine maintenance across fleet (inspections & PMs)
    logger.info("Generating routine fleet inspections and PMs...")

    for machine_id, group in sensor_df.groupby("machine_id"):
        m_id = str(machine_id)
        m_type = machine_types[m_id]
        parts_pool = PARTS_CATALOG[m_type]

        start_op_h = group["operating_hours"].iloc[0]
        end_op_h = group["operating_hours"].iloc[-1]

        # Schedule 2-3 inspection milestones roughly at 550h, 1300h, 2050h increments
        intervals = [550.0, 1300.0, 2050.0]
        for offset_h in intervals:
            target_h = start_op_h + offset_h
            if target_h >= end_op_h:
                continue

            # Find closest sensor row
            idx = (group["operating_hours"] - target_h).abs().idxmin()
            row = group.loc[idx]
            dt = row["timestamp"]

            # Avoid collision with special anchor dates
            if m_id == "M-104" and abs((dt - m104_date).total_seconds()) < 86400 * 10:
                continue
            if m_id == "M-106" and abs((dt - m106_date).total_seconds()) < 86400 * 10:
                continue
            if m_id == "M-112" and abs((dt - m112_date).total_seconds()) < 86400 * 5:
                continue
            if m_id == "M-105" and abs((dt - m105_date).total_seconds()) < 86400 * 5:
                continue
            if m_id == "M-110" and abs((dt - m110_date).total_seconds()) < 86400 * 5:
                continue
            if m_id == "M-111" and abs((dt - m111_date).total_seconds()) < 86400 * 5:
                continue

            # Decide whether this is inspection or PM
            is_pm = (offset_h > 1800.0) and (m_id not in ["M-104", "M-107"])

            if is_pm:
                dur = round(_sample_lognormal(rng, 4.0, 10.0, 6.0), 2)
                cost = round(_sample_lognormal(rng, 800.0, 2500.0, 1500.0), 2)
                num_parts = rng.integers(1, 3)
                chosen_parts = list(rng.choice(parts_pool, size=num_parts, replace=False))
                notes = (
                    f"Scheduled 2000h service performed at {round(target_h, 1)} operating hours. "
                    f"Replaced items: {', '.join(chosen_parts)}. System re-calibrated."
                )
                mtype = "preventive"
            else:
                dur = round(_sample_lognormal(rng, 1.5, 4.0, 2.5), 2)
                cost = round(_sample_lognormal(rng, 150.0, 500.0, 280.0), 2)
                chosen_parts = []
                notes = (
                    f"Routine 500h diagnostic inspection at {round(target_h, 1)} operating hours. "
                    f"Visual check, acoustic survey, and lubrication levels verified normal."
                )
                mtype = "inspection"

            sched_date = (dt - pd.Timedelta(days=int(rng.integers(3, 8)))).isoformat()
            records.append(
                {
                    "maintenance_id": _deterministic_uuid(f"{m_id}-{mtype}-{dt.isoformat()}"),
                    "machine_id": m_id,
                    "maintenance_type": mtype,
                    "scheduled_date": sched_date,
                    "performed_date": dt.isoformat(),
                    "duration_hours": dur,
                    "cost_gbp": cost,
                    "parts_replaced": json.dumps(chosen_parts),
                    "technician_notes": notes,
                    "root_cause": "",
                    "failure_mode": "",
                    "created_at": dt.isoformat(),
                }
            )

    df_out = pd.DataFrame(records)
    df_out = df_out.sort_values("performed_date").reset_index(drop=True)
    return df_out


def run() -> None:
    """Generate and write maintenance_raw.csv with acceptance assertions."""
    UNIFIED.mkdir(parents=True, exist_ok=True)
    out_file = UNIFIED / "maintenance_raw.csv"

    df = generate_maintenance_records()
    logger.info("Generated %d maintenance records across fleet.", len(df))

    # Acceptance Checks
    # 1. 30 to 60 total maintenance records across the fleet
    assert 30 <= len(df) <= 60, f"Expected 30-60 records, got {len(df)}"

    # 2. M-104 inspection record on 2026-09-20 with exact technician note
    m104_recs = df[
        (df["machine_id"] == "M-104") & (df["performed_date"].str.startswith("2026-09-20"))
    ]
    assert len(m104_recs) == 1, (
        f"Expected exactly 1 M-104 record on 2026-09-20, got {len(m104_recs)}"
    )
    expected_note = (
        "Minor air leak suspected on discharge manifold. Recommended follow-up in 30 days. "
        "Operating parameters within tolerance."
    )
    assert m104_recs.iloc[0]["technician_notes"] == expected_note, (
        f"M-104 note mismatch: {m104_recs.iloc[0]['technician_notes']}"
    )

    # 3. All machine_ids exist in machines.csv
    valid_ids = {m.machine_id for m in FLEET}
    assert set(df["machine_id"]).issubset(valid_ids), "Found invalid machine_ids"

    # 4. performed_date within timeline range for all records
    p_dates = pd.to_datetime(df["performed_date"], utc=True)
    t_min = pd.Timestamp("2026-03-21", tz="UTC")
    t_max = pd.Timestamp("2026-10-19 23:55:00", tz="UTC")
    assert (p_dates >= t_min).all() and (p_dates <= t_max).all(), "performed_date outside timeline"

    # 5. Costs and durations strictly positive
    assert (df["duration_hours"] > 0).all(), "Durations must be strictly positive"
    assert (df["cost_gbp"] > 0).all(), "Costs must be strictly positive"

    # 6. parts_replaced is valid JSON
    for p in df["parts_replaced"]:
        parsed = json.loads(p)
        assert isinstance(parsed, list), f"Expected JSON list in parts_replaced, got {type(parsed)}"

    # 7. failure_mode populated whenever maintenance_type IN ('corrective', 'emergency')
    for _, row in df.iterrows():
        if row["maintenance_type"] in ["corrective", "emergency"]:
            assert row["failure_mode"] != "", f"Missing failure_mode for corrective: {row}"
            assert row["root_cause"] != "", f"Missing root_cause for corrective: {row}"
        else:
            assert row["failure_mode"] == "", f"Unexpected failure_mode for non-corrective: {row}"

    df.to_csv(out_file, index=False)
    logger.info("Successfully wrote %d maintenance records to %s", len(df), out_file)


if __name__ == "__main__":
    run()
