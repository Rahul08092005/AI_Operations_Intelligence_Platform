"""Fleet definition and machine metadata for the simulated smart factory."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

try:
    from ml.preprocessing.config import UNIFIED
except (ImportError, ModuleNotFoundError):
    from config import UNIFIED


@dataclass(frozen=True)
class MachineConfig:
    machine_id: str
    machine_type: str
    location: str
    installation_date: date
    scenario: str
    ai4i_type: str
    status: str
    data_origin: str


FLEET: list[MachineConfig] = [
    MachineConfig(
        machine_id="M-101",
        machine_type="Air Compressor",
        location="Plant A, Line 1",
        installation_date=date(2021, 3, 15),
        scenario="healthy",
        ai4i_type="L",
        status="Running",
        data_origin="metropt_derived",
    ),
    MachineConfig(
        machine_id="M-102",
        machine_type="Air Compressor",
        location="Plant A, Line 1",
        installation_date=date(2021, 9, 1),
        scenario="healthy",
        ai4i_type="M",
        status="Running",
        data_origin="metropt_derived",
    ),
    MachineConfig(
        machine_id="M-103",
        machine_type="Air Compressor",
        location="Plant A, Line 2",
        installation_date=date(2020, 11, 10),
        scenario="mild_drift",
        ai4i_type="M",
        status="Warning",
        data_origin="metropt_derived",
    ),
    MachineConfig(
        machine_id="M-104",
        machine_type="Air Compressor",
        location="Plant A, Line 2",
        installation_date=date(2019, 6, 20),
        scenario="hero_degrading",
        ai4i_type="L",
        status="Critical",
        data_origin="metropt_real_signals",
    ),
    MachineConfig(
        machine_id="M-105",
        machine_type="Hydraulic Press",
        location="Plant B, Line 1",
        installation_date=date(2022, 2, 1),
        scenario="past_failure_old",
        ai4i_type="M",
        status="Running",
        data_origin="metropt_derived",
    ),
    MachineConfig(
        machine_id="M-106",
        machine_type="Hydraulic Press",
        location="Plant B, Line 1",
        installation_date=date(2021, 7, 12),
        scenario="recently_maintained",
        ai4i_type="H",
        status="Running",
        data_origin="metropt_derived",
    ),
    MachineConfig(
        machine_id="M-107",
        machine_type="Hydraulic Press",
        location="Plant B, Line 2",
        installation_date=date(2020, 5, 25),
        scenario="overdue_maintenance",
        ai4i_type="L",
        status="Running",
        data_origin="metropt_derived",
    ),
    MachineConfig(
        machine_id="M-108",
        machine_type="Hydraulic Press",
        location="Plant B, Line 2",
        installation_date=date(2022, 8, 30),
        scenario="healthy",
        ai4i_type="M",
        status="Running",
        data_origin="metropt_derived",
    ),
    MachineConfig(
        machine_id="M-109",
        machine_type="Conveyor Drive",
        location="Plant C, Line 1",
        installation_date=date(2021, 1, 18),
        scenario="benign_spikes",
        ai4i_type="L",
        status="Running",
        data_origin="metropt_derived",
    ),
    MachineConfig(
        machine_id="M-110",
        machine_type="Conveyor Drive",
        location="Plant C, Line 1",
        installation_date=date(2020, 9, 3),
        scenario="past_failure_mid",
        ai4i_type="M",
        status="Running",
        data_origin="metropt_derived",
    ),
    MachineConfig(
        machine_id="M-111",
        machine_type="Conveyor Drive",
        location="Plant C, Line 2",
        installation_date=date(2021, 10, 14),
        scenario="past_failure_recent",
        ai4i_type="H",
        status="Running",
        data_origin="metropt_derived",
    ),
    MachineConfig(
        machine_id="M-112",
        machine_type="Conveyor Drive",
        location="Plant C, Line 2",
        installation_date=date(2022, 4, 5),
        scenario="in_maintenance",
        ai4i_type="M",
        status="Maintenance",
        data_origin="metropt_derived",
    ),
]


def build_machines_df(output_path: Path | None = None) -> pd.DataFrame:
    """Build the DataFrame for the 12 machines and optionally write to CSV."""
    records = [asdict(m) for m in FLEET]
    df = pd.DataFrame(records)
    # Output columns ordering per contract
    cols = [
        "machine_id",
        "machine_type",
        "location",
        "installation_date",
        "status",
        "data_origin",
        "scenario",
        "ai4i_type",
    ]
    df = df[cols]

    target = output_path or (UNIFIED / "machines.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return df


if __name__ == "__main__":
    out_df = build_machines_df()
    print(f"Generated {len(out_df)} machines in {UNIFIED / 'machines.csv'}")
