"""Ingestion module for raw unified datasets.

Loads raw CSV and JSON datasets into typed pandas DataFrames with strict schema
validation, memory tracking, and optional sample mode.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

import pandas as pd

from ml.preprocessing.config import REFERENCE, TIMELINE_END, UNIFIED

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestReport:
    """Report metrics for an ingested dataset."""

    table: str
    rows: int
    columns: int
    memory_mb: float
    seconds: float


# Column contracts
MACHINES_COLS = {
    "machine_id",
    "machine_type",
    "location",
    "installation_date",
    "status",
    "data_origin",
    "scenario",
    "ai4i_type",
}

SENSORS_COLS = {
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
}

MAINTENANCE_COLS = {
    "maintenance_id",
    "machine_id",
    "maintenance_type",
    "scheduled_date",
    "performed_date",
    "duration_hours",
    "cost_gbp",
    "parts_replaced",
    "technician_notes",
    "root_cause",
    "failure_mode",
    "created_at",
}

PRODUCTION_COLS = {
    "production_id",
    "date",
    "machine_id",
    "shift",
    "units_produced",
    "units_target",
    "scrap_units",
    "operating_hours",
    "downtime_hours",
    "downtime_reason",
    "oee",
    "created_at",
}

BUSINESS_DAILY_COLS = {
    "date",
    "plant_id",
    "total_production_units",
    "total_scrap_units",
    "total_operating_hours",
    "total_downtime_hours",
    "total_energy_kwh",
    "energy_cost_gbp",
    "maintenance_cost_gbp",
    "revenue_gbp",
    "inventory_level",
    "backorder_units",
    "created_at",
}

TRANSACTIONS_COLS = {
    "transaction_id",
    "invoice_no",
    "stock_code",
    "description",
    "quantity",
    "unit_price",
    "customer_id",
    "country",
    "transaction_date",
    "is_cancellation",
    "line_total",
}

FAILURE_WINDOWS_COLS = {
    "failure_id",
    "start_time",
    "end_time",
    "failure_type",
    "source",
}


def _validate_columns(df: pd.DataFrame, expected: set[str], table_name: str) -> None:
    actual = set(df.columns)
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        raise ValueError(f"[{table_name}] Missing required columns: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"[{table_name}] Unexpected columns found: {sorted(unexpected)}")


def _log_and_report(
    table: str, file_path: Path, df: pd.DataFrame, start_time: float
) -> IngestReport:
    seconds = time.perf_counter() - start_time
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(
        "Loaded %s from %s: %d rows, %d columns, %.2f MB in %.3fs",
        table,
        file_path.name,
        len(df),
        len(df.columns),
        mem_mb,
        seconds,
    )
    return IngestReport(
        table=table,
        rows=len(df),
        columns=len(df.columns),
        memory_mb=round(mem_mb, 2),
        seconds=round(seconds, 3),
    )


def load_machines(
    path: Path | None = None, sample: bool = False
) -> tuple[pd.DataFrame, IngestReport]:
    t0 = time.perf_counter()
    p = path or (UNIFIED / "machines.csv")
    if not p.exists():
        raise FileNotFoundError(f"Machines file not found: {p}")

    dtypes = {
        "machine_id": "category",
        "machine_type": "string",
        "location": "string",
        "status": "string",
        "data_origin": "string",
        "scenario": "string",
        "ai4i_type": "string",
    }
    df = pd.read_csv(p, dtype=dtypes)
    _validate_columns(df, MACHINES_COLS, "machines")

    df["installation_date"] = pd.to_datetime(df["installation_date"], utc=True).dt.floor("D")

    if sample:
        df = df[df["machine_id"].isin(["M-103", "M-104"])].copy()
        df["machine_id"] = df["machine_id"].cat.remove_unused_categories()

    report = _log_and_report("machines", p, df, t0)
    return df, report


def load_sensors(
    path: Path | None = None, sample: bool = False
) -> tuple[pd.DataFrame, IngestReport]:
    t0 = time.perf_counter()
    p = path or (UNIFIED / "sensors_raw.csv")
    if not p.exists():
        raise FileNotFoundError(f"Sensors raw file not found: {p}")

    dtypes = {
        "machine_id": "category",
        "temperature": "float32",
        "vibration": "float32",
        "pressure": "float32",
        "rpm": "float32",
        "energy_consumption": "float32",
        "operating_hours": "float32",
        "motor_current": "float32",
        "comp_state": "float32",
    }
    df = pd.read_csv(p, dtype=dtypes)
    _validate_columns(df, SENSORS_COLS, "sensors")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if sample:
        sample_cutoff = pd.Timestamp(TIMELINE_END, tz=UTC) - pd.Timedelta(days=44)
        df = df[
            df["machine_id"].isin(["M-103", "M-104"]) & (df["timestamp"] >= sample_cutoff)
        ].copy()
        df["machine_id"] = df["machine_id"].cat.remove_unused_categories()

    report = _log_and_report("sensors", p, df, t0)
    return df, report


def load_maintenance(
    path: Path | None = None, sample: bool = False
) -> tuple[pd.DataFrame, IngestReport]:
    t0 = time.perf_counter()
    p = path or (UNIFIED / "maintenance_raw.csv")
    if not p.exists():
        raise FileNotFoundError(f"Maintenance raw file not found: {p}")

    dtypes = {
        "maintenance_id": "string",
        "machine_id": "category",
        "maintenance_type": "string",
        "duration_hours": "float32",
        "cost_gbp": "float32",
        "parts_replaced": "string",
        "technician_notes": "string",
        "root_cause": "string",
        "failure_mode": "string",
    }
    df = pd.read_csv(p, dtype=dtypes)
    _validate_columns(df, MAINTENANCE_COLS, "maintenance")

    df["scheduled_date"] = pd.to_datetime(df["scheduled_date"], utc=True)
    df["performed_date"] = pd.to_datetime(df["performed_date"], utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    if sample:
        df = df[df["machine_id"].isin(["M-103", "M-104"])].copy()
        df["machine_id"] = df["machine_id"].cat.remove_unused_categories()

    report = _log_and_report("maintenance", p, df, t0)
    return df, report


def load_production(
    path: Path | None = None, sample: bool = False
) -> tuple[pd.DataFrame, IngestReport]:
    t0 = time.perf_counter()
    p = path or (UNIFIED / "production_raw.csv")
    if not p.exists():
        raise FileNotFoundError(f"Production raw file not found: {p}")

    dtypes = {
        "production_id": "string",
        "machine_id": "category",
        "shift": "string",
        "units_produced": "int32",
        "units_target": "int32",
        "scrap_units": "int32",
        "operating_hours": "float32",
        "downtime_hours": "float32",
        "downtime_reason": "string",
        "oee": "float32",
    }
    df = pd.read_csv(p, dtype=dtypes)
    _validate_columns(df, PRODUCTION_COLS, "production")

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.floor("D")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    if sample:
        sample_cutoff = pd.Timestamp(TIMELINE_END, tz=UTC) - pd.Timedelta(days=44)
        df = df[df["machine_id"].isin(["M-103", "M-104"]) & (df["date"] >= sample_cutoff)].copy()
        df["machine_id"] = df["machine_id"].cat.remove_unused_categories()

    report = _log_and_report("production", p, df, t0)
    return df, report


def load_business_daily(
    path: Path | None = None, sample: bool = False
) -> tuple[pd.DataFrame, IngestReport]:
    t0 = time.perf_counter()
    p = path or (UNIFIED / "business_daily_raw.csv")
    if not p.exists():
        raise FileNotFoundError(f"Business daily raw file not found: {p}")

    dtypes = {
        "plant_id": "string",
        "total_production_units": "int32",
        "total_scrap_units": "int32",
        "total_operating_hours": "float32",
        "total_downtime_hours": "float32",
        "total_energy_kwh": "float32",
        "energy_cost_gbp": "float32",
        "maintenance_cost_gbp": "float32",
        "revenue_gbp": "float32",
        "inventory_level": "int32",
        "backorder_units": "int32",
    }
    df = pd.read_csv(p, dtype=dtypes)
    _validate_columns(df, BUSINESS_DAILY_COLS, "business_daily")

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.floor("D")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    if sample:
        sample_cutoff = pd.Timestamp(TIMELINE_END, tz=UTC) - pd.Timedelta(days=44)
        df = df[df["date"] >= sample_cutoff].copy()

    report = _log_and_report("business_daily", p, df, t0)
    return df, report


def load_transactions(
    path: Path | None = None, sample: bool = False
) -> tuple[pd.DataFrame, IngestReport]:
    t0 = time.perf_counter()
    p = path or (UNIFIED / "business_transactions_raw.csv")
    if not p.exists():
        raise FileNotFoundError(f"Business transactions raw file not found: {p}")

    dtypes = {
        "transaction_id": "string",
        "invoice_no": "string",
        "stock_code": "string",
        "description": "string",
        "quantity": "int32",
        "unit_price": "float32",
        "customer_id": "string",
        "country": "string",
        "is_cancellation": "bool",
        "line_total": "float32",
    }
    df = pd.read_csv(p, dtype=dtypes)
    _validate_columns(df, TRANSACTIONS_COLS, "business_transactions")

    df["transaction_date"] = pd.to_datetime(df["transaction_date"], utc=True)

    if sample:
        sample_cutoff = pd.Timestamp(TIMELINE_END, tz=UTC) - pd.Timedelta(days=44)
        df = df[df["transaction_date"] >= sample_cutoff].copy()

    report = _log_and_report("business_transactions", p, df, t0)
    return df, report


def load_failure_windows(path: Path | None = None) -> tuple[pd.DataFrame, IngestReport]:
    t0 = time.perf_counter()
    p = path or (REFERENCE / "metropt_failure_windows.csv")
    if not p.exists():
        raise FileNotFoundError(f"Failure windows reference file not found: {p}")

    dtypes = {
        "failure_id": "string",
        "failure_type": "string",
        "source": "string",
    }
    df = pd.read_csv(p, dtype=dtypes)
    _validate_columns(df, FAILURE_WINDOWS_COLS, "metropt_failure_windows")

    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)

    report = _log_and_report("failure_windows", p, df, t0)
    return df, report


def load_metadata(path: Path | None = None) -> dict[str, Any]:
    p = path or (UNIFIED / "timeline_metadata.json")
    if not p.exists():
        raise FileNotFoundError(f"Timeline metadata not found: {p}")
    with open(p, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data
