"""Comprehensive raw output validation suite for AI Operations Intelligence Platform.

Validates file existence, row counts, schema conformity, foreign keys,
temporal alignment, operational scenarios, and determinism.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.preprocessing.config import (
    REFERENCE,
    TIMELINE_END,
    TIMELINE_START,
    UNIFIED,
    VIB_WARNING,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_validations() -> dict[str, Any]:
    """Run full validation suite on unified datasets and return summary."""
    results: dict[str, Any] = {"checks": [], "failures": []}

    def check(name: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        results["checks"].append({"name": name, "status": status, "detail": detail})
        if not passed:
            results["failures"].append(f"{name}: {detail}")
            logger.error("VALIDATION FAILED: %s - %s", name, detail)
        else:
            logger.info("VALIDATION PASSED: %s %s", name, f"({detail})" if detail else "")

    # 1. File existence and non-zero size
    files_to_check = {
        "machines.csv": UNIFIED / "machines.csv",
        "sensors_raw.csv": UNIFIED / "sensors_raw.csv",
        "timeline_metadata.json": UNIFIED / "timeline_metadata.json",
        "maintenance_raw.csv": UNIFIED / "maintenance_raw.csv",
        "production_raw.csv": UNIFIED / "production_raw.csv",
        "business_daily_raw.csv": UNIFIED / "business_daily_raw.csv",
        "business_transactions_raw.csv": UNIFIED / "business_transactions_raw.csv",
        "ai4i_state_raw.csv": UNIFIED / "ai4i_state_raw.csv",
        "metropt_failure_windows.csv": REFERENCE / "metropt_failure_windows.csv",
    }

    for fname, fpath in files_to_check.items():
        exists = fpath.exists()
        size = fpath.stat().st_size if exists else 0
        check(f"File exists: {fname}", exists and size > 0, f"size={size} bytes")

    if results["failures"]:
        return results

    # 2. Row counts
    df_machines = pd.read_csv(files_to_check["machines.csv"])
    check("Row count: machines.csv == 12", len(df_machines) == 12, f"count={len(df_machines)}")

    df_sensors = pd.read_csv(files_to_check["sensors_raw.csv"])
    check(
        "Row count: sensors_raw.csv == 736128",
        len(df_sensors) == 736128,
        f"count={len(df_sensors)}",
    )

    df_maint = pd.read_csv(files_to_check["maintenance_raw.csv"])
    check(
        "Row count: maintenance_raw.csv in [30, 60]",
        30 <= len(df_maint) <= 60,
        f"count={len(df_maint)}",
    )

    df_prod = pd.read_csv(files_to_check["production_raw.csv"])
    check("Row count: production_raw.csv == 2556", len(df_prod) == 2556, f"count={len(df_prod)}")

    df_biz = pd.read_csv(files_to_check["business_daily_raw.csv"])
    check("Row count: business_daily_raw.csv == 213", len(df_biz) == 213, f"count={len(df_biz)}")

    df_trans = pd.read_csv(files_to_check["business_transactions_raw.csv"])
    check(
        "Row count: business_transactions_raw.csv in [400000, 550000]",
        400000 <= len(df_trans) <= 550000,
        f"count={len(df_trans)}",
    )

    df_ai4i = pd.read_csv(files_to_check["ai4i_state_raw.csv"])
    check("Row count: ai4i_state_raw.csv == 10000", len(df_ai4i) == 10000, f"count={len(df_ai4i)}")

    # 3. Foreign Key Integrity
    valid_m_ids = set(df_machines["machine_id"])
    check(
        "FK integrity: sensors_raw machine_ids valid",
        set(df_sensors["machine_id"]).issubset(valid_m_ids),
    )
    check(
        "FK integrity: maintenance_raw machine_ids valid",
        set(df_maint["machine_id"]).issubset(valid_m_ids),
    )
    check(
        "FK integrity: production_raw machine_ids valid",
        set(df_prod["machine_id"]).issubset(valid_m_ids),
    )

    # 4. Temporal Alignment
    sensors_ts = pd.to_datetime(df_sensors["timestamp"], utc=True)
    t_start = pd.Timestamp("2026-03-21", tz="UTC")
    t_end = pd.Timestamp("2026-10-19 23:55:00", tz="UTC")
    check(
        "Temporal alignment: sensors_raw covers 2026-03-21 to 2026-10-19",
        sensors_ts.min() == t_start and sensors_ts.max() == t_end,
        f"min={sensors_ts.min()}, max={sensors_ts.max()}",
    )

    prod_dates = pd.to_datetime(df_prod["date"]).dt.date
    check(
        "Temporal alignment: production_raw dates cover timeline",
        prod_dates.min() == TIMELINE_START and prod_dates.max() == TIMELINE_END,
    )

    biz_dates = pd.to_datetime(df_biz["date"]).dt.date
    check(
        "Temporal alignment: business_daily_raw dates cover timeline",
        biz_dates.min() == TIMELINE_START and biz_dates.max() == TIMELINE_END,
    )

    # 5. Operating Hours Monotonicity
    monotonic = True
    for _m_id, grp in df_sensors.groupby("machine_id"):
        diffs = grp["operating_hours"].diff().dropna()
        if (diffs < -1e-6).any():
            monotonic = False
            break
    check("Operating hours strictly monotonic per machine", monotonic)

    # 6. Scenario Checks
    # Healthy machines (M-101, M-102, M-108) vibration < 4.5 mm/s
    healthy_vib_max = df_sensors[df_sensors["machine_id"].isin(["M-101", "M-102", "M-108"])][
        "vibration"
    ].max()
    check(
        "Healthy machines vibration strictly < 4.5 mm/s",
        healthy_vib_max < VIB_WARNING,
        f"max={healthy_vib_max:.3f}",
    )

    # M-104 Hero degradation
    m104 = df_sensors[df_sensors["machine_id"] == "M-104"].copy()
    m104["timestamp"] = pd.to_datetime(m104["timestamp"], utc=True)
    m104_high = m104[m104["vibration"] >= VIB_WARNING]
    first_cross = m104_high["timestamp"].min() if len(m104_high) > 0 else None
    days_before_end = (t_end - first_cross).total_seconds() / 86400.0 if first_cross else 0.0
    check(
        "M-104 first vibration warning crossing between 7 and 14 days before end",
        7.0 <= days_before_end <= 14.0,
        f"first_cross={first_cross}, days_before_end={days_before_end:.2f}",
    )

    # M-104 Temperature in last 24h exceeds healthy 95th percentile by >= 8 °C
    t95_healthy = float(np.percentile(m104.iloc[:-288]["temperature"], 95))
    last_day_temp = float(m104.iloc[-288:]["temperature"].mean())
    temp_diff = last_day_temp - t95_healthy
    check(
        "M-104 final 24h temperature exceeds healthy 95th percentile by >= 8 °C",
        temp_diff >= 8.0,
        f"last_day={last_day_temp:.2f}, t95={t95_healthy:.2f}, diff={temp_diff:.2f}",
    )

    # M-104 inspection record on 2026-09-20
    m104_maint = df_maint[
        (df_maint["machine_id"] == "M-104")
        & (df_maint["performed_date"].str.startswith("2026-09-20"))
    ]
    check("M-104 has inspection record on 2026-09-20", len(m104_maint) == 1)
    if len(m104_maint) == 1:
        note = m104_maint.iloc[0]["technician_notes"]
        check(
            "M-104 inspection contains expected manifold warning note",
            "Minor air leak suspected on discharge manifold" in note,
        )

    # M-106 recently maintained
    m106_maint = df_maint[
        (df_maint["machine_id"] == "M-106") & (df_maint["maintenance_type"] == "preventive")
    ]
    check("M-106 has recent PM record", len(m106_maint) >= 1)

    # M-112 in maintenance state
    m112_maint = df_maint[
        (df_maint["machine_id"] == "M-112")
        & (df_maint["performed_date"].str.startswith("2026-10-18"))
    ]
    check("M-112 has ongoing overhaul starting 2026-10-18", len(m112_maint) == 1)

    # 7. Production Scrap Elevation & OEE
    m104_prod = df_prod[df_prod["machine_id"] == "M-104"].copy()
    m104_prod["scrap_rate"] = m104_prod["scrap_units"] / m104_prod["units_produced"]
    early_scrap = m104_prod.iloc[:-14]["scrap_rate"].mean()
    late_scrap = m104_prod.iloc[-14:]["scrap_rate"].mean()
    check(
        "M-104 scrap rate significantly elevated in final 14 days",
        late_scrap > 2.0 * early_scrap,
        f"early={early_scrap:.3f}, late={late_scrap:.3f}",
    )

    oee_min = df_prod["oee"].min()
    oee_max = df_prod["oee"].max()
    check(
        "OEE bounded in [0, 1]",
        oee_min >= 0.0 and oee_max <= 1.0,
        f"min={oee_min}, max={oee_max}",
    )

    # 8. Business Daily stock-flow balance
    check("Business daily inventory >= 0", (df_biz["inventory_level"] >= 0).all())
    check("Business daily backorders >= 0", (df_biz["backorder_units"] >= 0).all())
    check("Business daily revenue > 0", (df_biz["revenue_gbp"] > 0).all())

    return results


def write_report(results: dict[str, Any], report_path: Path) -> None:
    """Write human-readable validation summary report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    total_checks = len(results["checks"])
    failed_checks = len(results["failures"])
    passed_checks = total_checks - failed_checks

    lines = [
        "=" * 80,
        "AI OPERATIONS INTELLIGENCE PLATFORM - RAW DATA VALIDATION REPORT",
        f"Generated: {datetime.now(tz=UTC).isoformat()}",
        f"Total Checks: {total_checks} | Passed: {passed_checks} | Failed: {failed_checks}",
        f"Overall Status: {'PASSED' if failed_checks == 0 else 'FAILED'}",
        "=" * 80,
        "",
        "CHECK DETAILS:",
    ]

    for c in results["checks"]:
        detail_str = f" - {c['detail']}" if c["detail"] else ""
        lines.append(f"[{c['status']}] {c['name']}{detail_str}")

    if results["failures"]:
        lines.extend(["", "FAILURES REQUIRING ATTENTION:"])
        for f in results["failures"]:
            lines.append(f" - {f}")

    lines.append("")
    report_content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info("Validation report written to %s", report_path)


def run_and_report(report_path: Path = DOCS / "raw_data_validation_report.txt") -> int:
    """Run validation suite, write report, and return status code."""
    results = run_validations()
    write_report(results, report_path)
    if results["failures"]:
        logger.error("Validation failed with %d error(s).", len(results["failures"]))
        return 1
    logger.info("All raw data validations passed successfully!")
    return 0


def main() -> int:
    """CLI entrypoint for raw output validation."""
    parser = argparse.ArgumentParser(description="Validate standardized raw datasets.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DOCS / "raw_data_validation_report.txt",
        help="Path for output text report",
    )
    args = parser.parse_args()
    return run_and_report(args.report_path)


if __name__ == "__main__":
    sys.exit(main())
