"""Cleaning module for non-sensor operational datasets.

Cleans and normalizes maintenance records, machine daily production records,
and company-level business daily metrics without silent modifications.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ml.preprocessing.validation_rules import (
    MAINTENANCE_LIMITS,
    PRODUCTION_LIMITS,
)

logger = logging.getLogger(__name__)


@dataclass
class OtherCleaningLog:
    """Summary log for non-sensor table cleaning."""

    maintenance_rows_in: int = 0
    maintenance_rows_out: int = 0
    maintenance_normalized_failures: int = 0
    production_rows_in: int = 0
    production_rows_out: int = 0
    production_nullified_values: dict[str, int] = field(default_factory=dict)
    business_daily_rows_in: int = 0
    business_daily_rows_out: int = 0
    business_consistency_passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_maintenance(df_maint: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean and normalize maintenance dataset."""
    df = df_maint.copy()
    rows_in = len(df)

    # 1. Deduplication on maintenance_id
    if df["maintenance_id"].duplicated().any():
        logger.warning("Duplicate maintenance_id records found! Keeping first occurrence.")
        df = df.drop_duplicates(subset=["maintenance_id"], keep="first")

    # 2. Capitalize maintenance_type
    df["maintenance_type"] = df["maintenance_type"].astype(str).str.capitalize()

    # 3. Normalize failure_type / failure_mode: string 'None' for non-corrective rows
    fail_col = "failure_mode" if "failure_mode" in df.columns else "failure_type"
    normalized_failures = 0
    if fail_col in df.columns:
        is_non_corrective = df["maintenance_type"] != "Corrective"
        normalized_failures = int(is_non_corrective.sum())
        df.loc[is_non_corrective, fail_col] = "None"
        df.loc[df[fail_col].isna(), fail_col] = "None"
        # Standardize column name as failure_type for Day 3 consistency
        df["failure_type"] = df[fail_col]

    # 4. Out-of-range costs -> set to NaN with logging
    cost_col = "cost_gbp" if "cost_gbp" in df.columns else "maintenance_cost"
    if cost_col in df.columns:
        cost_lim = MAINTENANCE_LIMITS["maintenance_cost"]
        cost_viol = (df[cost_col] < cost_lim.hard_min) | (df[cost_col] > cost_lim.hard_max)
        if cost_viol.any():
            logger.warning("Nullifying %d out-of-range maintenance costs", cost_viol.sum())
            df.loc[cost_viol, cost_col] = np.nan

    log = {
        "rows_in": rows_in,
        "rows_out": len(df),
        "normalized_failures": normalized_failures,
    }
    return df, log


def clean_production(df_prod: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean daily machine production metrics."""
    df = df_prod.copy()
    rows_in = len(df)
    nullified: dict[str, int] = {}

    # Check duplicates on (machine_id, date)
    if df.duplicated(subset=["machine_id", "date"]).any():
        logger.warning("Duplicate (machine_id, date) rows in production! Keeping first.")
        df = df.drop_duplicates(subset=["machine_id", "date"], keep="first")

    # Map column names if needed
    if "units_produced" in df.columns and "production_quantity" not in df.columns:
        df["production_quantity"] = df["units_produced"]
    if "units_target" in df.columns and "target_quantity" not in df.columns:
        df["target_quantity"] = df["units_target"]
    if "downtime_hours" in df.columns and "downtime_minutes" not in df.columns:
        df["downtime_minutes"] = df["downtime_hours"] * 60.0

    # Physical limits checks
    for col in ["production_quantity", "target_quantity", "downtime_minutes"]:
        if col in df.columns:
            lim = PRODUCTION_LIMITS[col]
            viol = (df[col] < lim.hard_min) | (df[col] > lim.hard_max)
            cnt = int(viol.sum())
            if cnt > 0:
                logger.warning("Nullifying %d out-of-range values in %s", cnt, col)
                df.loc[viol, col] = np.nan
                nullified[col] = cnt

    log = {
        "rows_in": rows_in,
        "rows_out": len(df),
        "nullified_values": nullified,
    }
    return df, log


def clean_business_daily(
    df_biz: pd.DataFrame, df_prod: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and clean business daily fleet metrics."""
    df = df_biz.copy()
    rows_in = len(df)

    if df.duplicated(subset=["date"]).any():
        logger.warning("Duplicate date rows in business_daily! Keeping first.")
        df = df.drop_duplicates(subset=["date"], keep="first")

    log = {
        "rows_in": rows_in,
        "rows_out": len(df),
    }
    return df, log
