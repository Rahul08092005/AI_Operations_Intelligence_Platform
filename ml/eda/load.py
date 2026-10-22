"""Data loading utilities for EDA with pyarrow column selection and caching."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow.parquet as pq

from ml.eda.config import AI4I_CSV, AI4I_SPLIT_INDEX, PROCESSED

if TYPE_CHECKING:
    from collections.abc import Sequence


AI4I_RENAME = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}


def _read_parquet_selected(
    path: Path,
    columns: Sequence[str] | None = None,
    filters: list[Any] | None = None,
) -> pd.DataFrame:
    """Read a parquet file with column pruning via pyarrow."""
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")

    cols = list(columns) if columns is not None else None
    table = pq.read_table(path, columns=cols, filters=filters)
    return table.to_pandas()


@functools.lru_cache(maxsize=16)
def _load_ops_cached(columns_tuple: tuple[str, ...] | None) -> pd.DataFrame:
    cols = list(columns_tuple) if columns_tuple is not None else None
    df = _read_parquet_selected(PROCESSED / "operations_dataset.parquet", columns=cols)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_ops(columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load operations_dataset.parquet (daily machine-level data)."""
    cols_tup = tuple(columns) if columns is not None else None
    return _load_ops_cached(cols_tup).copy()


def load_sensors(
    columns: Sequence[str] | None = None,
    machines: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load cleaned 5-minute sensor data with optional column/machine filtering."""
    path = PROCESSED / "sensors.parquet"
    filters = None
    if machines is not None:
        filters = [("machine_id", "in", list(machines))]

    # Ensure required filtering columns are loaded if not present
    cols_to_load = list(columns) if columns is not None else None
    if cols_to_load is not None and "timestamp" not in cols_to_load:
        cols_to_load.append("timestamp")
    if cols_to_load is not None and machines is not None and "machine_id" not in cols_to_load:
        cols_to_load.append("machine_id")

    df = _read_parquet_selected(path, columns=cols_to_load, filters=filters)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_sensor_features(
    columns: Sequence[str] | None = None,
    machines: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load sensor_features.parquet (rolling, trend, and maintenance features)."""
    path = PROCESSED / "sensor_features.parquet"
    filters = None
    if machines is not None:
        filters = [("machine_id", "in", list(machines))]

    cols_to_load = list(columns) if columns is not None else None
    if cols_to_load is not None and "timestamp" not in cols_to_load:
        cols_to_load.append("timestamp")
    if cols_to_load is not None and machines is not None and "machine_id" not in cols_to_load:
        cols_to_load.append("machine_id")

    df = _read_parquet_selected(path, columns=cols_to_load, filters=filters)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@functools.lru_cache(maxsize=4)
def load_maintenance() -> pd.DataFrame:
    """Load maintenance logs."""
    df = _read_parquet_selected(PROCESSED / "maintenance.parquet")
    if "scheduled_date" in df.columns:
        df["scheduled_date"] = pd.to_datetime(df["scheduled_date"]).dt.date
    if "completed_date" in df.columns:
        df["completed_date"] = pd.to_datetime(df["completed_date"]).dt.date
    return df.copy()


@functools.lru_cache(maxsize=4)
def load_machines() -> pd.DataFrame:
    """Load machines metadata table."""
    return _read_parquet_selected(PROCESSED / "machines.parquet").copy()


@functools.lru_cache(maxsize=4)
def load_labels() -> pd.DataFrame:
    """Load ground-truth anomaly evaluation labels (for interpretation only)."""
    df = _read_parquet_selected(PROCESSED / "anomaly_eval_labels.parquet")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.copy()


def load_transactions(columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load cleaned business retail transactions."""
    path = PROCESSED / "business_transactions.parquet"
    df = _read_parquet_selected(path, columns=columns)
    if "invoice_date" in df.columns:
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], utc=True)
    return df


@functools.lru_cache(maxsize=4)
def load_business_daily() -> pd.DataFrame:
    """Load business_daily.parquet (daily factory production/cost aggregates)."""
    df = _read_parquet_selected(PROCESSED / "business_daily.parquet")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.copy()


@functools.lru_cache(maxsize=4)
def load_business_features_daily() -> pd.DataFrame:
    """Load business_features_daily.parquet (retail demand KPIs)."""
    df = _read_parquet_selected(PROCESSED / "business_features_daily.parquet")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.copy()


@functools.lru_cache(maxsize=4)
def load_product_demand() -> pd.DataFrame:
    """Load product_demand_daily.parquet (top 50 SKUs daily demand)."""
    df = _read_parquet_selected(PROCESSED / "product_demand_daily.parquet")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.copy()


def load_ai4i(partition: str | None = None) -> pd.DataFrame:
    """Load AI4I 2020 dataset, renaming columns to snake_case.

    Args:
        partition: If "train", loads strictly the training partition.
                   If None, loads the raw full dataset.
                   (Note: "test" should not be loaded during EDA per R2/D46).
    """
    csv_candidates = list(AI4I_CSV.glob("*.csv")) if AI4I_CSV.is_dir() else [AI4I_CSV]
    if not csv_candidates or not csv_candidates[0].exists():
        raise FileNotFoundError(f"AI4I raw CSV not found in {AI4I_CSV}")

    df = pd.read_csv(csv_candidates[0])
    df = df.rename(columns=AI4I_RENAME)

    if partition is not None:
        if not AI4I_SPLIT_INDEX.exists():
            from ml.eda.splits import save_ai4i_split

            save_ai4i_split()
        split_df = pd.read_csv(AI4I_SPLIT_INDEX)
        valid_udis = split_df.loc[split_df["partition"] == partition, "udi"]
        df = df[df["udi"].isin(valid_udis)].reset_index(drop=True)

    return df


__all__ = [
    "AI4I_RENAME",
    "load_ai4i",
    "load_business_daily",
    "load_business_features_daily",
    "load_labels",
    "load_machines",
    "load_maintenance",
    "load_ops",
    "load_product_demand",
    "load_sensor_features",
    "load_sensors",
    "load_transactions",
]
