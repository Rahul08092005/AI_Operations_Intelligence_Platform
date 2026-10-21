"""Data validation module for raw and cleaned operational datasets.

Validation detects, checks, and reports issues without modifying data.
Provides before/after quality checks across sensors, production, maintenance,
business daily, and retail transactions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from ml.preprocessing.config import (
    RETAIL_EXTREME_QTY,
    SENSOR_FREQ,
)
from ml.preprocessing.validation_rules import (
    ALLOWED_MAINTENANCE_TYPES,
    MAINTENANCE_LIMITS,
    NON_PRODUCT_CODE_REGEX,
    PRODUCTION_LIMITS,
    SENSOR_LIMITS,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Individual validation check result."""

    name: str
    table: str
    severity: str  # "error" | "warning"
    count: int
    details: dict[str, Any]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_sensors(df_sensors: pd.DataFrame, df_machines: pd.DataFrame) -> list[ValidationResult]:
    results: list[ValidationResult] = []

    # 1. Duplicates on (machine_id, timestamp)
    dup_mask = df_sensors.duplicated(subset=["machine_id", "timestamp"], keep=False)
    dup_count = int(dup_mask.sum())
    identical_dups = 0
    conflicting_dups = 0
    if dup_count > 0:
        dup_rows = df_sensors[dup_mask]
        all_col_dups = dup_rows.duplicated(keep=False).sum()
        identical_dups = int(all_col_dups)
        conflicting_dups = dup_count - identical_dups

    results.append(
        ValidationResult(
            name="duplicate_machine_timestamp",
            table="sensors",
            severity="error" if dup_count > 0 else "warning",
            count=dup_count,
            details={
                "identical_duplicates": identical_dups,
                "conflicting_duplicates": conflicting_dups,
            },
            passed=dup_count == 0,
        )
    )

    # 2. Timestamps strictly on 5-minute grid & gap detection
    off_grid = (
        (df_sensors["timestamp"].dt.minute % 5 != 0) | (df_sensors["timestamp"].dt.second != 0)
    ).sum()
    results.append(
        ValidationResult(
            name="off_5min_grid",
            table="sensors",
            severity="error" if off_grid > 0 else "warning",
            count=int(off_grid),
            details={"off_grid_count": int(off_grid)},
            passed=off_grid == 0,
        )
    )

    # Missing interval / gap distribution per machine
    gaps_by_machine: dict[str, Any] = {}
    total_gaps = 0
    for m_id, group in df_sensors.groupby("machine_id", observed=True):
        sorted_ts = group["timestamp"].sort_values()
        diffs = sorted_ts.diff()
        gap_intervals = (
            diffs[diffs > pd.Timedelta(SENSOR_FREQ)] / pd.Timedelta(SENSOR_FREQ) - 1
        ).astype(int)
        if len(gap_intervals) > 0:
            m_gaps = int(gap_intervals.sum())
            total_gaps += m_gaps
            gaps_by_machine[str(m_id)] = {
                "num_gaps": int(len(gap_intervals)),
                "total_missing_intervals": m_gaps,
                "max_gap_intervals": int(gap_intervals.max()),
            }

    results.append(
        ValidationResult(
            name="sensor_data_gaps",
            table="sensors",
            severity="warning",
            count=total_gaps,
            details=gaps_by_machine,
            passed=True,  # Informational check
        )
    )

    # 3. Monotonic timestamp order per machine
    out_of_order = 0
    for _m_id, group in df_sensors.groupby("machine_id", observed=True):
        if not group["timestamp"].is_monotonic_increasing:
            out_of_order += 1

    results.append(
        ValidationResult(
            name="timestamp_ordering",
            table="sensors",
            severity="error" if out_of_order > 0 else "warning",
            count=out_of_order,
            details={"machines_out_of_order": out_of_order},
            passed=out_of_order == 0,
        )
    )

    # 4. Hard physical range violations
    limit_violations: dict[str, int] = {}
    for col, lim in SENSOR_LIMITS.items():
        if col in df_sensors.columns:
            s = df_sensors[col].dropna()
            v_count = int(((s < lim.hard_min) | (s > lim.hard_max)).sum())
            if v_count > 0:
                limit_violations[col] = v_count

    total_sensor_violations = sum(limit_violations.values())
    results.append(
        ValidationResult(
            name="sensor_hard_limit_violations",
            table="sensors",
            severity="error" if total_sensor_violations > 0 else "warning",
            count=total_sensor_violations,
            details=limit_violations,
            passed=total_sensor_violations == 0,
        )
    )

    # 5. Missing values per column
    missing_counts = {
        col: int(df_sensors[col].isna().sum())
        for col in df_sensors.columns
        if df_sensors[col].isna().sum() > 0
    }
    results.append(
        ValidationResult(
            name="sensor_missing_values",
            table="sensors",
            severity="warning",
            count=sum(missing_counts.values()),
            details=missing_counts,
            passed=True,
        )
    )

    # 6. Monotonic non-decreasing operating hours
    op_hours_decreases = 0
    for _m_id, group in df_sensors.groupby("machine_id", observed=True):
        op_h = group["operating_hours"].dropna()
        dec = (op_h.diff() < -1e-4).sum()
        op_hours_decreases += int(dec)

    results.append(
        ValidationResult(
            name="operating_hours_monotonic",
            table="sensors",
            severity="error" if op_hours_decreases > 0 else "warning",
            count=op_hours_decreases,
            details={"decreases_count": op_hours_decreases},
            passed=op_hours_decreases == 0,
        )
    )

    # 7. Foreign key check
    valid_machines = set(df_machines["machine_id"])
    sensor_machines = set(df_sensors["machine_id"].unique())
    unknown_m = sensor_machines - valid_machines
    results.append(
        ValidationResult(
            name="sensor_foreign_keys",
            table="sensors",
            severity="error" if len(unknown_m) > 0 else "warning",
            count=len(unknown_m),
            details={"unknown_machines": list(unknown_m)},
            passed=len(unknown_m) == 0,
        )
    )

    return results


def check_production_and_business(
    df_prod: pd.DataFrame, df_biz: pd.DataFrame, df_machines: pd.DataFrame
) -> list[ValidationResult]:
    results: list[ValidationResult] = []

    # 1. One row per (machine_id, date), no missing dates
    prod_dups = int(df_prod.duplicated(subset=["machine_id", "date"]).sum())
    results.append(
        ValidationResult(
            name="production_unique_machine_date",
            table="production",
            severity="error" if prod_dups > 0 else "warning",
            count=prod_dups,
            details={"duplicate_rows": prod_dups},
            passed=prod_dups == 0,
        )
    )

    # Missing dates per machine
    all_dates = pd.date_range(df_prod["date"].min(), df_prod["date"].max(), freq="D")
    expected_count = len(df_machines) * len(all_dates)
    missing_prod_dates = max(0, expected_count - len(df_prod))
    results.append(
        ValidationResult(
            name="production_missing_dates",
            table="production",
            severity="warning" if missing_prod_dates > 0 else "warning",
            count=missing_prod_dates,
            details={"expected": expected_count, "actual": len(df_prod)},
            passed=missing_prod_dates == 0,
        )
    )

    # 2. Hard range violations
    qty_col = "units_produced" if "units_produced" in df_prod.columns else "production_quantity"
    tgt_col = "units_target" if "units_target" in df_prod.columns else "target_quantity"
    dt_col = "downtime_hours" if "downtime_hours" in df_prod.columns else "downtime_minutes"

    violations = 0
    if qty_col in df_prod:
        violations += int(
            (
                (df_prod[qty_col] < PRODUCTION_LIMITS["production_quantity"].hard_min)
                | (df_prod[qty_col] > PRODUCTION_LIMITS["production_quantity"].hard_max)
            ).sum()
        )
    if tgt_col in df_prod:
        violations += int(
            (
                (df_prod[tgt_col] < PRODUCTION_LIMITS["target_quantity"].hard_min)
                | (df_prod[tgt_col] > PRODUCTION_LIMITS["target_quantity"].hard_max)
            ).sum()
        )
    if dt_col in df_prod:
        dt_mins = df_prod[dt_col] * 60.0 if dt_col == "downtime_hours" else df_prod[dt_col]
        violations += int(
            (
                (dt_mins < PRODUCTION_LIMITS["downtime_minutes"].hard_min)
                | (dt_mins > PRODUCTION_LIMITS["downtime_minutes"].hard_max)
            ).sum()
        )

    results.append(
        ValidationResult(
            name="production_hard_limits",
            table="production",
            severity="error" if violations > 0 else "warning",
            count=violations,
            details={"violations": violations},
            passed=violations == 0,
        )
    )

    # 3. Foreign key
    valid_machines = set(df_machines["machine_id"])
    prod_machines = set(df_prod["machine_id"].unique())
    unknown_m = prod_machines - valid_machines
    results.append(
        ValidationResult(
            name="production_foreign_keys",
            table="production",
            severity="error" if len(unknown_m) > 0 else "warning",
            count=len(unknown_m),
            details={"unknown_machines": list(unknown_m)},
            passed=len(unknown_m) == 0,
        )
    )

    # 4. Consistency: business_daily vs aggregated production
    mismatch_count = 0
    is_sampled = len(df_prod["machine_id"].unique()) < len(df_machines)
    if not is_sampled:
        if "total_production_units" in df_biz.columns and qty_col in df_prod.columns:
            p_agg = df_prod.groupby("date")[qty_col].sum()
            b_agg = df_biz.set_index("date")["total_production_units"]
            common_dates = p_agg.index.intersection(b_agg.index)
            diff = (p_agg.loc[common_dates] - b_agg.loc[common_dates]).abs()
            mismatch_count += int((diff > 1e-3).sum())

        if "total_downtime_hours" in df_biz.columns and dt_col in df_prod.columns:
            p_dt = (
                df_prod.groupby("date")[dt_col].sum()
                if dt_col == "downtime_hours"
                else df_prod.groupby("date")[dt_col].sum() / 60.0
            )
            b_dt = df_biz.set_index("date")["total_downtime_hours"]
            common_dates = p_dt.index.intersection(b_dt.index)
            diff_dt = (p_dt.loc[common_dates] - b_dt.loc[common_dates]).abs()
            mismatch_count += int((diff_dt > 1e-2).sum())

    results.append(
        ValidationResult(
            name="production_business_consistency",
            table="production_and_business",
            severity="error" if (mismatch_count > 0 and not is_sampled) else "warning",
            count=mismatch_count,
            details={
                "mismatched_days": mismatch_count,
                "is_sampled": is_sampled,
            },
            passed=mismatch_count == 0 or is_sampled,
        )
    )

    return results


def check_maintenance(df_maint: pd.DataFrame, df_machines: pd.DataFrame) -> list[ValidationResult]:
    results: list[ValidationResult] = []

    # 1. Unique maintenance_id
    dup_id = int(df_maint["maintenance_id"].duplicated().sum())
    results.append(
        ValidationResult(
            name="maintenance_id_unique",
            table="maintenance",
            severity="error" if dup_id > 0 else "warning",
            count=dup_id,
            details={"duplicate_ids": dup_id},
            passed=dup_id == 0,
        )
    )

    # 2. Allowed maintenance_type values
    normalized_types = df_maint["maintenance_type"].astype(str).str.capitalize()
    invalid_types = (~normalized_types.isin(ALLOWED_MAINTENANCE_TYPES)).sum()
    results.append(
        ValidationResult(
            name="maintenance_allowed_types",
            table="maintenance",
            severity="error" if invalid_types > 0 else "warning",
            count=int(invalid_types),
            details={
                "allowed": sorted(ALLOWED_MAINTENANCE_TYPES),
                "invalid_count": int(invalid_types),
            },
            passed=invalid_types == 0,
        )
    )

    # 3. failure_type is None/null unless Corrective
    fail_col = "failure_mode" if "failure_mode" in df_maint.columns else "failure_type"
    non_corrective = normalized_types != "Corrective"
    unexpected_failures = 0
    if fail_col in df_maint.columns:
        has_failure = df_maint[fail_col].notna() & (
            df_maint[fail_col].astype(str).str.lower() != "none"
        )
        unexpected_failures = int((non_corrective & has_failure).sum())

    results.append(
        ValidationResult(
            name="maintenance_failure_type_consistency",
            table="maintenance",
            severity="warning" if unexpected_failures > 0 else "warning",
            count=unexpected_failures,
            details={"non_corrective_with_failure_type": unexpected_failures},
            passed=unexpected_failures == 0,
        )
    )

    # 4. Cost range
    cost_col = "cost_gbp" if "cost_gbp" in df_maint.columns else "maintenance_cost"
    cost_violations = 0
    if cost_col in df_maint.columns:
        cost_s = df_maint[cost_col].dropna()
        cost_violations = int(
            (
                (cost_s < MAINTENANCE_LIMITS["maintenance_cost"].hard_min)
                | (cost_s > MAINTENANCE_LIMITS["maintenance_cost"].hard_max)
            ).sum()
        )

    results.append(
        ValidationResult(
            name="maintenance_cost_range",
            table="maintenance",
            severity="error" if cost_violations > 0 else "warning",
            count=cost_violations,
            details={"cost_violations": cost_violations},
            passed=cost_violations == 0,
        )
    )

    # 5. Foreign key
    valid_machines = set(df_machines["machine_id"])
    maint_machines = set(df_maint["machine_id"].unique())
    unknown_m = maint_machines - valid_machines
    results.append(
        ValidationResult(
            name="maintenance_foreign_keys",
            table="maintenance",
            severity="error" if len(unknown_m) > 0 else "warning",
            count=len(unknown_m),
            details={"unknown_machines": list(unknown_m)},
            passed=len(unknown_m) == 0,
        )
    )

    return results


def check_transactions(df_trans: pd.DataFrame) -> list[ValidationResult]:
    results: list[ValidationResult] = []

    # 1. Missing customer_id, missing description
    missing_cust = int(df_trans["customer_id"].isna().sum())
    missing_desc = int(df_trans["description"].isna().sum())
    results.append(
        ValidationResult(
            name="transactions_missing_required",
            table="business_transactions",
            severity="warning",
            count=missing_cust + missing_desc,
            details={"missing_customer_id": missing_cust, "missing_description": missing_desc},
            passed=True,
        )
    )

    # 2. Unit price <= 0 rows
    non_pos_price = int((df_trans["unit_price"] <= 0).sum())
    results.append(
        ValidationResult(
            name="transactions_non_positive_price",
            table="business_transactions",
            severity="warning",
            count=non_pos_price,
            details={"non_positive_price_count": non_pos_price},
            passed=non_pos_price == 0,
        )
    )

    # 3. Quantity < 0 vs cancellation rows
    is_neg_qty = df_trans["quantity"] < 0
    is_canc = df_trans["invoice_no"].astype(str).str.startswith("C")
    agreement = (is_neg_qty == is_canc).sum()
    disagreement = int(len(df_trans) - agreement)

    results.append(
        ValidationResult(
            name="transactions_returns_cancellations",
            table="business_transactions",
            severity="warning",
            count=disagreement,
            details={
                "negative_qty_count": int(is_neg_qty.sum()),
                "cancellation_invoice_count": int(is_canc.sum()),
                "disagreement_count": disagreement,
            },
            passed=True,
        )
    )

    # 4. Revenue calculation
    calc_line = df_trans["quantity"] * df_trans["unit_price"]
    rev_col = "line_total" if "line_total" in df_trans.columns else "revenue"
    if rev_col in df_trans.columns:
        diff = (df_trans[rev_col] - calc_line).abs()
        mismatch_rev = int((diff > 0.05).sum())
    else:
        mismatch_rev = 0

    results.append(
        ValidationResult(
            name="transactions_revenue_calculation",
            table="business_transactions",
            severity="error" if mismatch_rev > 0 else "warning",
            count=mismatch_rev,
            details={"calculation_mismatches": mismatch_rev},
            passed=mismatch_rev == 0,
        )
    )

    # 5. Non-product stock codes
    non_prod_pattern = re.compile(NON_PRODUCT_CODE_REGEX)
    codes = df_trans["stock_code"].astype(str)
    non_prod_mask = ~codes.str.match(non_prod_pattern)
    non_prod_count = int(non_prod_mask.sum())
    non_prod_samples = sorted(codes[non_prod_mask].unique().tolist())[:20]

    results.append(
        ValidationResult(
            name="transactions_non_product_codes",
            table="business_transactions",
            severity="warning",
            count=non_prod_count,
            details={
                "distinct_non_product_count": len(codes[non_prod_mask].unique()),
                "samples": non_prod_samples,
            },
            passed=True,
        )
    )

    # 6. Extreme quantities
    extreme_mask = df_trans["quantity"].abs() >= RETAIL_EXTREME_QTY
    extreme_count = int(extreme_mask.sum())
    top_10 = (
        df_trans.assign(abs_qty=df_trans["quantity"].abs())
        .sort_values(by="abs_qty", ascending=False)[
            ["transaction_id", "invoice_no", "stock_code", "quantity"]
        ]
        .head(10)
        .to_dict(orient="records")
    )

    results.append(
        ValidationResult(
            name="transactions_extreme_quantities",
            table="business_transactions",
            severity="warning",
            count=extreme_count,
            details={"extreme_threshold": RETAIL_EXTREME_QTY, "top_10": top_10},
            passed=True,
        )
    )

    # 7. Duplicate transaction_id
    dup_t_id = int(df_trans["transaction_id"].duplicated().sum())
    results.append(
        ValidationResult(
            name="transactions_duplicate_id",
            table="business_transactions",
            severity="error" if dup_t_id > 0 else "warning",
            count=dup_t_id,
            details={"duplicate_transaction_ids": dup_t_id},
            passed=dup_t_id == 0,
        )
    )

    return results


def run_all_validations(tables: dict[str, pd.DataFrame]) -> list[ValidationResult]:
    """Execute all validation checks across all provided tables."""
    results: list[ValidationResult] = []

    machines = tables.get("machines")
    sensors = tables.get("sensors")
    production = tables.get("production")
    business_daily = tables.get("business_daily")
    maintenance = tables.get("maintenance")
    transactions = tables.get("business_transactions", tables.get("transactions"))

    if sensors is not None and machines is not None:
        results.extend(check_sensors(sensors, machines))

    if production is not None and business_daily is not None and machines is not None:
        results.extend(check_production_and_business(production, business_daily, machines))

    if maintenance is not None and machines is not None:
        results.extend(check_maintenance(maintenance, machines))

    if transactions is not None:
        results.extend(check_transactions(transactions))

    return results


def summarize_validation(results: list[ValidationResult]) -> dict[str, Any]:
    """Produce a high-level JSON-serializable summary of validation results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    errors = sum(1 for r in results if not r.passed and r.severity == "error")
    warnings = sum(1 for r in results if not r.passed and r.severity == "warning")

    return {
        "total_checks": total,
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "status": "PASSED" if errors == 0 else "FAILED",
        "results": [r.to_dict() for r in results],
    }
