"""Data Engineering Pipeline Orchestrator for AI Operations Intelligence Platform.

Executes the Day 3 end-to-end data processing workflow:
1. Ingest raw datasets (sample or full fleet)
2. Validate before cleaning
3. Clean sensors (grid reindexing, fault nullification, bounded imputation, warmup flags)
4. Clean other tables (maintenance, production, business daily, machines)
5. Outliers (Tier 2 rolling robust z-scores and anomaly preservation checks)
6. Clean transactions (kept vs. rejected, KPI flags, revenue_clean)
7. Generate anomaly evaluation labels (separate from feature tables)
8. Engineer sensor features (rolling statistics, trends, LS slopes, maintenance as-of joins)
9. Daily integration & business features (company KPIs, product demand, operations dataset)
10. Validate after cleaning (fail on any remaining errors)
11. Atomic writing of all Parquet and CSV outputs
12. Quality report generation (JSON and Markdown)
13. Optional database loading into PostgreSQL / SQLite
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml.preprocessing.clean_other import (
    clean_business_daily,
    clean_machines,
    clean_maintenance,
    clean_production,
)
from ml.preprocessing.clean_sensors import clean_sensors
from ml.preprocessing.clean_transactions import clean_transactions
from ml.preprocessing.config import PROCESSED
from ml.preprocessing.features_business import (
    compute_daily_retail_features,
    compute_product_demand_daily,
)
from ml.preprocessing.features_sensor import (
    engineer_sensor_features,
    generate_feature_catalog,
)
from ml.preprocessing.ingest import (
    load_business_daily,
    load_failure_windows,
    load_machines,
    load_maintenance,
    load_metadata,
    load_production,
    load_sensors,
    load_transactions,
)
from ml.preprocessing.integrate import (
    build_operations_dataset,
    compute_daily_sensor_aggregates,
)
from ml.preprocessing.labels import generate_anomaly_eval_labels
from ml.preprocessing.outliers import flag_outliers, run_preservation_checks
from ml.preprocessing.report import generate_quality_report
from ml.preprocessing.validate import run_all_validations, summarize_validation

logger = logging.getLogger(__name__)


@dataclass
class StepTiming:
    """Execution timing and row statistics for a pipeline step."""

    step: str
    table: str
    rows_in: int
    rows_out: int
    seconds: float


@dataclass
class PipelineResult:
    """End-to-end pipeline execution result."""

    success: bool
    sample_mode: bool
    timings: list[StepTiming]
    validation_before: dict[str, Any]
    validation_after: dict[str, Any]
    quality_report_path: Path
    markdown_report_path: Path
    db_load_result: dict[str, Any] | None


def _atomic_write_parquet(df: pd.DataFrame, target_path: Path) -> None:
    """Atomically write a DataFrame to Parquet by writing to .tmp then replacing."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.stem}.tmp{target_path.suffix}")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(target_path)


def _atomic_write_csv(df: pd.DataFrame, target_path: Path, **kwargs: Any) -> None:
    """Atomically write a DataFrame to CSV by writing to .tmp then replacing."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.stem}.tmp{target_path.suffix}")
    df.to_csv(tmp_path, index=False, **kwargs)
    tmp_path.replace(target_path)


def audit_temporal_leakage(
    sensor_features: pd.DataFrame,
    operations_dataset: pd.DataFrame,
    anomaly_labels: pd.DataFrame,
) -> dict[str, Any]:
    """Audit all outputs for compliance with temporal leakage policies L1 - L5."""
    leakage_results: dict[str, Any] = {}

    # Check 1: Label isolation (Policy L4)
    label_cols = {
        "in_failure_window",
        "in_precursor_window",
        "failure_window_id",
        "episode_type",
        "episode_id",
    }
    sf_leaks = label_cols.intersection(sensor_features.columns)
    leakage_results["labels_in_sensor_features"] = {
        "policy": "L4 (Label Isolation)",
        "passed": len(sf_leaks) == 0,
        "leaked_columns": sorted(sf_leaks),
    }

    ops_leaks = label_cols.intersection(operations_dataset.columns)
    leakage_results["labels_in_operations_dataset"] = {
        "policy": "L4 (Label Isolation)",
        "passed": len(ops_leaks) == 0,
        "leaked_columns": sorted(ops_leaks),
    }

    # Check 2: Forbidden columns in operations dataset
    forbidden = {"status", "scenario", "data_origin", "ai4i_type"}
    forbidden_in_ops = forbidden.intersection(operations_dataset.columns)
    leakage_results["forbidden_metadata_in_operations"] = {
        "policy": "L4 (Metadata Isolation)",
        "passed": len(forbidden_in_ops) == 0,
        "leaked_columns": sorted(forbidden_in_ops),
    }

    # Check 3: Unique keys in operations dataset
    dup_keys = int(operations_dataset.duplicated(subset=["machine_id", "date"]).sum())
    leakage_results["operations_dataset_unique_keys"] = {
        "policy": "Grain Integrity (Machine x Day)",
        "passed": dup_keys == 0,
        "duplicate_count": dup_keys,
    }

    # Check 4: Maintenance strictly past as-of join (Policy L2)
    # days_since_maintenance should be >= 1.0 everywhere if computed
    valid_maint_days = sensor_features["days_since_maintenance"].dropna()
    min_days_maint = float(valid_maint_days.min()) if len(valid_maint_days) > 0 else 1.0
    leakage_results["maintenance_strictly_past"] = {
        "policy": "L2 (Strictly Past Maintenance)",
        "passed": bool((valid_maint_days >= 1.0).all()) if len(valid_maint_days) > 0 else True,
        "min_days_since_maintenance": min_days_maint,
    }

    return leakage_results


def run_pipeline(
    sample: bool = False,
    skip_db: bool = False,
    only_step: str | None = None,
    log_level: str = "INFO",
    output_dir: Path | None = None,
) -> PipelineResult:
    """Execute end-to-end data engineering pipeline."""
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))
    out_dir = output_dir or PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: list[StepTiming] = []

    logger.info("================================================================================")
    logger.info(
        "STARTING DATA PIPELINE (mode=%s, skip_db=%s, only_step=%s)",
        "SAMPLE" if sample else "FULL FLEET",
        skip_db,
        only_step or "ALL",
    )
    logger.info("================================================================================")

    # -------------------------------------------------------------------------
    # Step 1: Ingestion
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 1/11] Ingesting unified raw datasets...")
    df_raw_machines, _ = load_machines(sample=sample)
    df_raw_sensors, _ = load_sensors(sample=sample)
    df_raw_maintenance, _ = load_maintenance(sample=sample)
    df_raw_production, _ = load_production(sample=sample)
    df_raw_biz_daily, _ = load_business_daily(sample=sample)
    df_raw_trans, _ = load_transactions(sample=sample)
    df_failure_windows, _ = load_failure_windows()
    metadata = load_metadata()

    timings.append(
        StepTiming(
            step="1_ingest",
            table="sensors_raw",
            rows_in=len(df_raw_sensors),
            rows_out=len(df_raw_sensors),
            seconds=round(time.perf_counter() - t0, 3),
        )
    )

    if only_step == "ingest":
        logger.info("Exiting after step: ingest")
        return PipelineResult(
            True,
            sample,
            timings,
            {},
            {},
            out_dir / "quality_report.json",
            out_dir / "data_quality_report.md",
            None,
        )

    # -------------------------------------------------------------------------
    # Step 2: Validate Before
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 2/11] Running pre-cleaning validation suite...")
    val_before_results = run_all_validations(
        {
            "machines": df_raw_machines,
            "sensors": df_raw_sensors,
            "maintenance": df_raw_maintenance,
            "production": df_raw_production,
            "business_daily": df_raw_biz_daily,
            "business_transactions": df_raw_trans,
        }
    )
    val_before_summary = summarize_validation(val_before_results)
    logger.info(
        "Pre-cleaning validation: %d checks, %d passed, %d errors, %d warnings",
        val_before_summary["total_checks"],
        val_before_summary["passed"],
        val_before_summary["errors"],
        val_before_summary["warnings"],
    )
    timings.append(
        StepTiming(
            step="2_validate_before",
            table="all_tables",
            rows_in=len(df_raw_sensors),
            rows_out=len(df_raw_sensors),
            seconds=round(time.perf_counter() - t0, 3),
        )
    )

    if only_step == "validate_before":
        logger.info("Exiting after step: validate_before")
        return PipelineResult(
            True,
            sample,
            timings,
            val_before_summary,
            {},
            out_dir / "quality_report.json",
            out_dir / "data_quality_report.md",
            None,
        )

    # -------------------------------------------------------------------------
    # Step 3: Clean Sensors & Non-Sensor Tables
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 3/11] Cleaning sensors (grid, faults, imputation, warmup)...")
    df_clean_sensors, sensor_clean_log = clean_sensors(df_raw_sensors)
    timings.append(
        StepTiming(
            step="3_clean_sensors",
            table="sensors",
            rows_in=len(df_raw_sensors),
            rows_out=len(df_clean_sensors),
            seconds=round(time.perf_counter() - t0, 3),
        )
    )

    t0 = time.perf_counter()
    logger.info("[Step 3b/11] Cleaning machines, maintenance, production, business daily...")
    df_clean_machines, _ = clean_machines(df_raw_machines)
    df_clean_maintenance, _ = clean_maintenance(df_raw_maintenance)
    df_clean_production, _ = clean_production(df_raw_production)
    df_clean_biz_daily, _ = clean_business_daily(df_raw_biz_daily, df_clean_production)
    timings.append(
        StepTiming(
            step="3b_clean_other",
            table="other_tables",
            rows_in=len(df_raw_production),
            rows_out=len(df_clean_production),
            seconds=round(time.perf_counter() - t0, 3),
        )
    )

    # -------------------------------------------------------------------------
    # Step 4: Tier 2 Outlier Flagging & Anomaly Preservation Checks
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 4/11] Flagging Tier 2 outliers and verifying anomaly preservation...")
    df_sensors_flagged = flag_outliers(df_clean_sensors)
    shift_days = metadata.get("shift_days", 2240)
    preservation_report = run_preservation_checks(
        df_raw_sensors=df_raw_sensors,
        df_clean_sensors=df_sensors_flagged,
        df_failure_windows=df_failure_windows,
        shift_days=shift_days,
    )
    timings.append(
        StepTiming(
            step="4_outliers",
            table="sensors_flagged",
            rows_in=len(df_clean_sensors),
            rows_out=len(df_sensors_flagged),
            seconds=round(time.perf_counter() - t0, 3),
        )
    )

    # -------------------------------------------------------------------------
    # Step 5: Clean Retail Transactions
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 5/11] Cleaning and flagging retail transactions...")
    df_kept_trans, df_rejected_trans, retail_log = clean_transactions(df_raw_trans)
    timings.append(
        StepTiming(
            step="5_transactions",
            table="business_transactions",
            rows_in=len(df_raw_trans),
            rows_out=len(df_kept_trans),
            seconds=round(time.perf_counter() - t0, 3),
        )
    )

    # -------------------------------------------------------------------------
    # Step 6: Generate Anomaly Evaluation Labels
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 6/11] Generating ground-truth anomaly evaluation labels...")
    df_labels = generate_anomaly_eval_labels(
        df_clean_sensors=df_sensors_flagged,
        df_failure_windows=df_failure_windows,
        metadata=metadata,
    )
    timings.append(
        StepTiming(
            step="6_labels",
            table="anomaly_eval_labels",
            rows_in=len(df_sensors_flagged),
            rows_out=len(df_labels),
            seconds=round(time.perf_counter() - t0, 3),
        )
    )

    # -------------------------------------------------------------------------
    # Step 7: Engineer Sensor Features & Feature Catalog
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 7/11] Engineering sensor rolling, trend, and maintenance features...")
    df_sensor_features = engineer_sensor_features(df_sensors_flagged, df_clean_maintenance)
    generate_feature_catalog(output_dir=out_dir)
    timings.append(
        StepTiming(
            step="7_sensor_features",
            table="sensor_features",
            rows_in=len(df_sensors_flagged),
            rows_out=len(df_sensor_features),
            seconds=round(time.perf_counter() - t0, 3),
        )
    )

    # -------------------------------------------------------------------------
    # Step 8: Daily Integration & Business Features
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 8/11] Computing daily business features and product demand...")
    df_biz_features, _ = compute_daily_retail_features(df_kept_trans)
    df_prod_demand = compute_product_demand_daily(df_kept_trans)

    logger.info("[Step 8b/11] Integrating daily operations dataset...")
    df_sensor_aggregates = compute_daily_sensor_aggregates(df_sensor_features)
    df_operations, _ = build_operations_dataset(
        df_machines=df_clean_machines,
        df_production=df_clean_production,
        df_business_daily=df_clean_biz_daily,
        df_sensor_aggregates=df_sensor_aggregates,
        df_sensor_features=df_sensor_features,
        df_maintenance=df_clean_maintenance,
        df_retail_daily=df_biz_features,
        output_dir=out_dir,
    )
    timings.append(
        StepTiming(
            step="8_operations_dataset",
            table="operations_dataset",
            rows_in=len(df_sensor_features),
            rows_out=len(df_operations),
            seconds=round(time.perf_counter() - t0, 3),
        )
    )

    # -------------------------------------------------------------------------
    # Step 9: Validate After Cleaning & Audit Leakage
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 9/11] Running post-cleaning validation and temporal leakage audit...")
    val_after_results = run_all_validations(
        {
            "machines": df_clean_machines,
            "sensors": df_sensors_flagged,
            "maintenance": df_clean_maintenance,
            "production": df_clean_production,
            "business_daily": df_clean_biz_daily,
            "business_transactions": df_kept_trans,
        }
    )
    val_after_summary = summarize_validation(val_after_results)
    leakage_results = audit_temporal_leakage(
        sensor_features=df_sensor_features,
        operations_dataset=df_operations,
        anomaly_labels=df_labels,
    )
    timings.append(
        StepTiming(
            step="9_validate_after",
            table="all_tables",
            rows_in=len(df_sensors_flagged),
            rows_out=len(df_sensors_flagged),
            seconds=round(time.perf_counter() - t0, 3),
        )
    )

    # Check for remaining validation errors
    if val_after_summary["errors"] > 0:
        err_details = [
            r for r in val_after_summary["results"] if not r["passed"] and r["severity"] == "error"
        ]
        logger.error(
            "CRITICAL: %d validation errors remain in post-cleaning checks! Details: %s",
            val_after_summary["errors"],
            err_details,
        )
        raise RuntimeError(
            f"Pipeline validation failed: {val_after_summary['errors']} errors remaining."
        )

    # -------------------------------------------------------------------------
    # Step 10: Atomic Writing of Outputs
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 10/11] Atomically writing all processed datasets to %s...", out_dir)

    _atomic_write_parquet(df_clean_machines, out_dir / "machines.parquet")
    _atomic_write_parquet(df_sensors_flagged, out_dir / "sensors.parquet")
    _atomic_write_parquet(df_sensor_features, out_dir / "sensor_features.parquet")
    _atomic_write_parquet(df_clean_maintenance, out_dir / "maintenance.parquet")
    _atomic_write_parquet(df_clean_production, out_dir / "production.parquet")
    _atomic_write_parquet(df_clean_biz_daily, out_dir / "business_daily.parquet")
    _atomic_write_parquet(df_kept_trans, out_dir / "business_transactions.parquet")
    _atomic_write_parquet(
        df_rejected_trans, out_dir / "business_transactions_rejected.parquet"
    )
    _atomic_write_parquet(df_biz_features, out_dir / "business_features_daily.parquet")
    _atomic_write_parquet(df_prod_demand, out_dir / "product_demand_daily.parquet")
    _atomic_write_parquet(df_operations, out_dir / "operations_dataset.parquet")
    _atomic_write_csv(
        df_operations, out_dir / "operations_dataset.csv", date_format="%Y-%m-%d"
    )
    _atomic_write_parquet(df_labels, out_dir / "anomaly_eval_labels.parquet")

    timings.append(
        StepTiming(
            step="10_atomic_writes",
            table="12_parquet_csv_files",
            rows_in=len(df_sensors_flagged),
            rows_out=len(df_sensors_flagged),
            seconds=round(time.perf_counter() - t0, 3),
        )
    )

    # -------------------------------------------------------------------------
    # Step 11: Quality Report & Optional Database Load
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 11/11] Generating quality report and executing database loader...")

    cleaned_dict = {
        "machines": df_clean_machines,
        "sensors": df_sensors_flagged,
        "sensor_features": df_sensor_features,
        "maintenance": df_clean_maintenance,
        "production": df_clean_production,
        "business_daily": df_clean_biz_daily,
        "business_transactions": df_kept_trans,
        "business_transactions_rejected": df_rejected_trans,
        "business_features_daily": df_biz_features,
        "product_demand_daily": df_prod_demand,
        "operations_dataset": df_operations,
        "anomaly_eval_labels": df_labels,
    }

    raw_dict = {
        "machines": df_raw_machines,
        "sensors": df_raw_sensors,
        "maintenance": df_raw_maintenance,
        "production": df_raw_production,
        "business_daily": df_raw_biz_daily,
        "business_transactions": df_raw_trans,
    }

    db_load_results: dict[str, Any] | None = None
    if not skip_db:
        try:
            # Import dynamically to prevent hard dependency on backend if db not installed
            from scripts.load_to_db import run_database_loader

            logger.info("Executing database loader into PostgreSQL...")
            db_load_results = run_database_loader(processed_dir=out_dir)
        except Exception as exc:
            logger.warning("Database loading skipped or failed: %s", exc)
            db_load_results = {"status": "FAILED_OR_UNAVAILABLE", "error": str(exc)}

    _, json_rep_path, md_rep_path = generate_quality_report(
        raw_tables=raw_dict,
        cleaned_tables=cleaned_dict,
        sensor_cleaning_log=sensor_clean_log,
        preservation_report=preservation_report,
        retail_log=retail_log,
        validation_before=val_before_summary,
        validation_after=val_after_summary,
        leakage_results=leakage_results,
        db_load_results=db_load_results,
        sample_mode=sample,
        output_dir=out_dir,
    )

    timings.append(
        StepTiming(
            step="11_reports",
            table="quality_report",
            rows_in=len(df_operations),
            rows_out=len(df_operations),
            seconds=round(time.perf_counter() - t0, 3),
        )
    )

    # -------------------------------------------------------------------------
    # Final Summary Table
    # -------------------------------------------------------------------------
    total_seconds = sum(t.seconds for t in timings)
    print("\n" + "=" * 80)
    print(f"PIPELINE SUMMARY ({'SAMPLE MODE' if sample else 'FULL FLEET'})")
    print("=" * 80)
    header_fmt = "{:<22} | {:<25} | {:<10} | {:<10} | {:<8}"
    print(header_fmt.format("Step", "Table / Target", "Rows In", "Rows Out", "Time (s)"))
    print("-" * 80)
    row_fmt = "{:<22} | {:<25} | {:<10,d} | {:<10,d} | {:<8.2f}"
    for t in timings:
        print(row_fmt.format(t.step, t.table, t.rows_in, t.rows_out, t.seconds))
    print("-" * 80)
    mins = total_seconds / 60.0
    print(f"TOTAL PIPELINE EXECUTION TIME: {total_seconds:.2f} seconds ({mins:.2f} minutes)")
    print(f"QUALITY REPORT SAVED TO: {json_rep_path} and {md_rep_path}")
    print("=" * 80 + "\n")

    return PipelineResult(
        success=True,
        sample_mode=sample,
        timings=timings,
        validation_before=val_before_summary,
        validation_after=val_after_summary,
        quality_report_path=json_rep_path,
        markdown_report_path=md_rep_path,
        db_load_result=db_load_results,
    )
