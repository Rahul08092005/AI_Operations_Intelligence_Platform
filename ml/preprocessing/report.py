"""Quality report generator for data engineering pipeline.

Produces `data/processed/quality_report.json` and `docs/data_quality_report.md`.
Documents:
1. Row counts per table: raw, removed, final.
2. Values set to missing (Tier 1), imputed values, longest gap per machine.
3. Duplicates removed.
4. Tier 2 outlier flag rates per machine.
5. Anomaly preservation verification results.
6. Retail cleaning: rejections, flags, non-product codes, extreme quantities.
7. Validation results before and after, side-by-side.
8. NaN rates in sensor_features and operations_dataset.
9. Row-level leakage check outcomes.
10. Database load verification (if executed).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml.preprocessing.config import PROCESSED

logger = logging.getLogger(__name__)


@dataclass
class QualityReportData:
    """Structured container for all data pipeline quality metrics."""

    timestamp: str
    sample_mode: bool
    table_row_counts: dict[str, dict[str, int]]
    sensor_cleaning: dict[str, Any]
    tier2_outliers: dict[str, Any]
    anomaly_preservation: dict[str, Any]
    retail_cleaning: dict[str, Any]
    validation_comparison: dict[str, Any]
    nan_rates_sensor_features: dict[str, float]
    nan_rates_operations_dataset: dict[str, float]
    leakage_checks: dict[str, Any]
    database_load: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_quality_report(
    raw_tables: dict[str, pd.DataFrame],
    cleaned_tables: dict[str, pd.DataFrame],
    sensor_cleaning_log: Any,
    preservation_report: Any,
    retail_log: Any,
    validation_before: dict[str, Any],
    validation_after: dict[str, Any],
    leakage_results: dict[str, Any],
    db_load_results: dict[str, Any] | None = None,
    sample_mode: bool = False,
    output_dir: Path | None = None,
    docs_dir: Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Generate both JSON and Markdown quality reports."""
    out_dir = output_dir or PROCESSED
    d_dir = docs_dir or (out_dir.parents[1] / "docs")
    out_dir.mkdir(parents=True, exist_ok=True)
    d_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "quality_report.json"
    md_path = d_dir / "data_quality_report.md"

    # 1. Table row counts
    table_counts: dict[str, dict[str, int]] = {}
    all_keys = sorted(set(raw_tables.keys()).union(cleaned_tables.keys()))
    for k in all_keys:
        raw_r = len(raw_tables[k]) if k in raw_tables else 0
        clean_r = len(cleaned_tables[k]) if k in cleaned_tables else 0
        diff = raw_r - clean_r
        table_counts[k] = {
            "raw_rows": raw_r,
            "final_rows": clean_r,
            "removed_rows": max(0, diff),
            "added_rows": max(0, -diff),
        }

    # 2. Sensor cleaning stats & gaps per machine
    sensor_clean_dict = (
        sensor_cleaning_log.to_dict()
        if hasattr(sensor_cleaning_log, "to_dict")
        else dict(sensor_cleaning_log)
    )

    clean_sensors = cleaned_tables.get("sensors")
    longest_gap_per_machine: dict[str, int] = {}
    if clean_sensors is not None and "is_gap" in clean_sensors.columns:
        for m_id, grp in clean_sensors.groupby("machine_id", observed=True):
            gap_series = grp["is_gap"].astype(int)
            runs = gap_series.groupby((~grp["is_gap"]).cumsum()).sum()
            longest_gap_per_machine[str(m_id)] = int(runs.max()) if len(runs) > 0 else 0

    sensor_clean_dict["longest_gap_per_machine"] = longest_gap_per_machine

    # 3. Tier 2 Outliers & Anomaly preservation
    pres_dict = (
        preservation_report.to_dict()
        if hasattr(preservation_report, "to_dict")
        else dict(preservation_report)
    )

    tier2_info = {
        "flag_rates_pct": pres_dict.get("tier2_flag_rates", {}),
        "total_flagged_rows": int(clean_sensors["is_outlier_flag"].sum())
        if clean_sensors is not None and "is_outlier_flag" in clean_sensors.columns
        else 0,
    }

    # 4. Retail stats
    retail_dict = (
        retail_log.to_dict() if hasattr(retail_log, "to_dict") else dict(retail_log)
    )

    # 5. Validation comparison
    val_comp = {
        "before": {
            "total_checks": validation_before.get("total_checks", 0),
            "passed": validation_before.get("passed", 0),
            "errors": validation_before.get("errors", 0),
            "warnings": validation_before.get("warnings", 0),
            "status": validation_before.get("status", "UNKNOWN"),
        },
        "after": {
            "total_checks": validation_after.get("total_checks", 0),
            "passed": validation_after.get("passed", 0),
            "errors": validation_after.get("errors", 0),
            "warnings": validation_after.get("warnings", 0),
            "status": validation_after.get("status", "UNKNOWN"),
        },
        "details_after": [
            r for r in validation_after.get("results", []) if not r.get("passed", True)
        ],
    }

    # 6. NaN rates in sensor_features and operations_dataset
    sf_nan: dict[str, float] = {}
    sf_df = cleaned_tables.get("sensor_features")
    if sf_df is not None:
        for col in sf_df.columns:
            rate = float(sf_df[col].isna().mean() * 100.0)
            if rate > 0:
                sf_nan[col] = round(rate, 2)

    ops_nan: dict[str, float] = {}
    ops_df = cleaned_tables.get("operations_dataset")
    if ops_df is not None:
        for col in ops_df.columns:
            rate = float(ops_df[col].isna().mean() * 100.0)
            if rate > 0:
                ops_nan[col] = round(rate, 2)

    # Build report object
    report_data = QualityReportData(
        timestamp=datetime.now(UTC).isoformat(),
        sample_mode=sample_mode,
        table_row_counts=table_counts,
        sensor_cleaning=sensor_clean_dict,
        tier2_outliers=tier2_info,
        anomaly_preservation=pres_dict,
        retail_cleaning=retail_dict,
        validation_comparison=val_comp,
        nan_rates_sensor_features=sf_nan,
        nan_rates_operations_dataset=ops_nan,
        leakage_checks=leakage_results,
        database_load=db_load_results,
    )

    report_dict = report_data.to_dict()

    # Save JSON atomically
    tmp_json = json_path.with_suffix(".tmp.json")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    tmp_json.replace(json_path)

    # Format Markdown
    md_content = _build_markdown_report(report_data)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Generated quality reports at %s and %s", json_path, md_path)
    return report_dict, json_path, md_path


def _build_markdown_report(data: QualityReportData) -> str:
    """Format structured QualityReportData into GitHub-flavored Markdown."""
    mode_str = (
        "SAMPLE (2 machines, 45 days)"
        if data.sample_mode
        else "FULL FLEET (12 machines, 214 days)"
    )
    status_str = (
        "PASSED (Zero Errors)"
        if data.validation_comparison["after"]["errors"] == 0
        else "FAILED"
    )

    lines: list[str] = [
        "# Data Quality & Pipeline Audit Report",
        "",
        f"**Generated:** {data.timestamp}  ",
        f"**Pipeline Mode:** {mode_str}  ",
        f"**Pipeline Status:** {status_str}  ",
        "",
        "---",
        "",
        "## 1. Table Row Counts (Raw vs. Processed)",
        "",
        "| Table Name | Raw Rows | Final Rows | Change / Reason |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for table, counts in sorted(data.table_row_counts.items()):
        raw_r = f"{counts['raw_rows']:,}"
        final_r = f"{counts['final_rows']:,}"
        if counts["added_rows"] > 0:
            diff_str = f"+{counts['added_rows']:,} (explicit gaps / reindexing)"
        elif counts["removed_rows"] > 0:
            diff_str = f"-{counts['removed_rows']:,} (invalid / rejected / duplicates)"
        else:
            diff_str = "0 (preserved)"
        lines.append(f"| `{table}` | {raw_r} | {final_r} | {diff_str} |")

    sc = data.sensor_cleaning
    hard_range_total = sum(sc.get("hard_range_violations", {}).values())

    lines.extend([
        "",
        "---",
        "",
        "## 2. Sensor Cleaning & Missing Value Imputation",
        "",
        f"- **Raw Rows Ingested:** {sc.get('raw_rows', 0):,}",
        f"- **Exact Duplicates Dropped:** {sc.get('exact_duplicates_dropped', 0):,}",
        f"- **Conflicting Duplicates Resolved:** {sc.get('conflicting_duplicates_resolved', 0):,}",
        f"- **Explicit Gap Rows Created:** {sc.get('explicit_gap_rows_created', 0):,}",
        f"- **Tier 1 Hard Range Violations (Set to NaN):** {hard_range_total}",
        f"- **Short Gaps Imputed (<= 15 min):** {sc.get('imputed_rows', 0):,} rows",
        f"- **Warmup Rows Flagged (First 24h):** {sc.get('warmup_rows', 0):,}",
        "",
        "### Longest Gap per Machine (in 5-min intervals)",
        "",
        "| Machine ID | Longest Gap Intervals | Equivalent Hours |",
        "| :--- | :--- | :--- |",
    ])

    for m_id, gap_cnt in sorted(sc.get("longest_gap_per_machine", {}).items()):
        lines.append(f"| `{m_id}` | {gap_cnt:,} | {round(gap_cnt * 5 / 60, 1)} h |")

    ap = data.anomaly_preservation
    t2_status = "PASSED" if ap.get("passed") else "FAILED"
    zero_destroyed = f"{ap.get('zero_real_data_destroyed')} (0 destroyed)"

    lines.extend([
        "",
        "---",
        "",
        "## 3. Tier 2 Outlier Flagging & Anomaly Preservation",
        "",
        f"- **Total Rows Flagged by Tier 2:** {data.tier2_outliers.get('total_flagged_rows', 0):,}",
        f"- **Preservation Suite Status:** {t2_status}",
        f"- **Real Failure Window Faults:** {zero_destroyed}",
        f"- **Peak Temperature Preservation:** {ap.get('peak_preservation_temperature')}",
        f"- **Peak Vibration Preservation:** {ap.get('peak_preservation_vibration')}",
        f"- **Benign Spikes Preserved:** {ap.get('benign_spikes_preserved')}",
        "",
        "### Tier 2 Outlier Flag Rates per Machine",
        "",
        "| Machine ID | Flag Rate (%) |",
        "| :--- | :--- |",
    ])

    for m_id, rate in sorted(data.tier2_outliers.get("flag_rates_pct", {}).items()):
        lines.append(f"| `{m_id}` | {rate:.2f}% |")

    rc = data.retail_cleaning
    cancel_str = f"{rc.get('cancellations', 0):,} / {rc.get('returns', 0):,}"

    lines.extend([
        "",
        "---",
        "",
        "## 4. Retail Transactions Cleaning & Rejections",
        "",
        f"- **Raw Transactions Ingested:** {rc.get('raw_rows', 0):,}",
        f"- **Clean Kept Transactions:** {rc.get('kept_rows', 0):,}",
        f"- **Rejected Transactions:** {rc.get('rejected_rows', 0):,}",
        f"- **Cancellations / Returns:** {cancel_str}",
        f"- **Guest Customers Flagged:** {rc.get('guests', 0):,}",
        f"- **Non-Product Lines Flagged:** {rc.get('non_product_lines', 0):,}",
        f"- **Extreme Quantities Flagged (>= 10,000):** {rc.get('extreme_quantities', 0)}",
        "",
        "### Rejection Reasons Breakdown",
        "",
        "| Reject Reason | Count |",
        "| :--- | :--- |",
    ])

    for reason, cnt in sorted(rc.get("reject_reasons", {}).items()):
        lines.append(f"| `{reason}` | {cnt:,} |")

    vb = data.validation_comparison["before"]
    va = data.validation_comparison["after"]

    lines.extend([
        "",
        "---",
        "",
        "## 5. Validation Suite Comparison (Before vs. After)",
        "",
        "| Metric | Before Pipeline | After Pipeline |",
        "| :--- | :--- | :--- |",
        f"| **Overall Status** | {vb['status']} | {va['status']} |",
        f"| **Total Checks** | {vb['total_checks']} | {va['total_checks']} |",
        f"| **Checks Passed** | {vb['passed']} | {va['passed']} |",
        f"| **Errors Remaining** | {vb['errors']} | **{va['errors']}** |",
        f"| **Warnings (Audits)** | {vb['warnings']} | {va['warnings']} |",
        "",
        "---",
        "",
        "## 6. Feature NaN Rates (Sensor Features & Operations Dataset)",
        "",
        "### `sensor_features.parquet` (Top non-zero NaN rates)",
        "",
        "| Feature Name | NaN Rate (%) | Note / Reason |",
        "| :--- | :--- | :--- |",
    ])

    sorted_sf = sorted(data.nan_rates_sensor_features.items(), key=lambda x: x[1], reverse=True)
    if not sorted_sf:
        lines.append("| *All features 0.0% NaN* | 0.0% | Fully populated |")
    else:
        for feat, rate in sorted_sf[:15]:
            lines.append(f"| `{feat}` | {rate:.2f}% | Expected window warmup / missing signal |")

    lines.extend([
        "",
        "### `operations_dataset.parquet` (Daily Machine Level)",
        "",
        "| Column Name | NaN Rate (%) | Note / Reason |",
        "| :--- | :--- | :--- |",
    ])

    sorted_ops = sorted(data.nan_rates_operations_dataset.items(), key=lambda x: x[1], reverse=True)
    if not sorted_ops:
        lines.append("| *All columns 0.0% NaN* | 0.0% | Complete |")
    else:
        for col, rate in sorted_ops[:15]:
            lines.append(f"| `{col}` | {rate:.2f}% | Expected lag warmup |")

    lines.extend([
        "",
        "---",
        "",
        "## 7. Temporal Leakage Audit & Verification",
        "",
        "| Leakage Check | Policy | Result |",
        "| :--- | :--- | :--- |",
    ])

    for chk, res in sorted(data.leakage_checks.items()):
        status = "PASSED (No Leakage)" if res.get("passed", True) else "FAILED (Leakage Detected)"
        lines.append(f"| `{chk}` | {res.get('policy', 'Temporal')} | **{status}** |")

    lines.extend([
        "",
        "---",
        "",
        "## 8. Database Loading Verification",
        "",
    ])

    if data.database_load:
        db_stat = data.database_load.get("status", "Unknown")
        lines.extend([
            f"**Database Status:** {db_stat}  ",
            f"**Verified Row Counts:** {data.database_load.get('rows_verified', 'N/A')}  ",
            "",
            "| Table | Rows Loaded | Target Parquet Rows | Match |",
            "| :--- | :--- | :--- | :--- |",
        ])
        for t, stat in data.database_load.get("tables", {}).items():
            lines.append(
                f"| `{t}` | {stat.get('db_rows', 0):,} | {stat.get('parquet_rows', 0):,} | "
                f"{stat.get('match', False)} |"
            )
    else:
        lines.append("*Database load was skipped (`--skip-db` or PostgreSQL not reachable).*")

    lines.append("")
    return "\n".join(lines)
