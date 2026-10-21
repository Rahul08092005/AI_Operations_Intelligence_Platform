"""Database loader for cleaned operational datasets.

Loads processed/cleaned operational data into PostgreSQL / SQLAlchemy tables
using PostgreSQL COPY inside a single atomic transaction, with idempotent replace mode,
foreign-key-safe ordering, gap-row skipping, and post-load verification.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

# Add backend directory to sys.path
ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app.models  # noqa: E402, F401 (registers models on Base.metadata)
from app.database.session import check_db, engine  # noqa: E402

from ml.preprocessing.config import PROCESSED, UNIFIED  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Foreign-key dependency order (load machines first, child tables after)
LOAD_ORDER = [
    "machines",
    "sensors_raw",
    "maintenance_records",
    "production_daily",
    "business_daily",
    "business_transactions",
]

# Truncation order (reverse of load order)
TRUNCATE_ORDER = list(reversed(LOAD_ORDER))


def get_table_dataframe(table_name: str, processed_dir: Path, unified_dir: Path) -> pd.DataFrame:
    """Load dataframe for database ingestion with gap filtering and column mapping."""
    if table_name == "machines":
        p = processed_dir / "machines.parquet"
        if not p.exists():
            p = unified_dir / "machines.csv"
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
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
        df = df[[c for c in cols if c in df.columns]].copy()
        df["installation_date"] = pd.to_datetime(df["installation_date"]).dt.date
        return df

    elif table_name == "sensors_raw":
        p = processed_dir / "sensors.parquet"
        if not p.exists():
            p = unified_dir / "sensors_raw.csv"
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Skip rows where all 5 main signals are missing (pure gap rows)
        main_signals = ["temperature", "vibration", "pressure", "rpm", "energy_consumption"]
        has_signal = df[[c for c in main_signals if c in df.columns]].notna().any(axis=1)
        df_filtered = df[has_signal].copy()

        cols = [
            "timestamp",
            "machine_id",
            "temperature",
            "vibration",
            "pressure",
            "motor_current",
            "energy_consumption",
            "rpm",
            "comp_state",
            "operating_hours",
        ]
        return df_filtered[[c for c in cols if c in df_filtered.columns]].copy()

    elif table_name == "maintenance_records":
        p = processed_dir / "maintenance.parquet"
        if not p.exists():
            p = unified_dir / "maintenance_raw.csv"
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        cols = [
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
        ]
        df = df[[c for c in cols if c in df.columns]].copy()
        for dt_c in ["scheduled_date", "performed_date"]:
            if dt_c in df.columns:
                df[dt_c] = pd.to_datetime(df[dt_c], utc=True)
        return df

    elif table_name == "production_daily":
        p = processed_dir / "production.parquet"
        if not p.exists():
            p = unified_dir / "production_raw.csv"
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        cols = [
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
        ]
        return df[[c for c in cols if c in df.columns]].copy()

    elif table_name == "business_daily":
        p = processed_dir / "business_daily.parquet"
        if not p.exists():
            p = unified_dir / "business_daily_raw.csv"
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        cols = [
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
        ]
        return df[[c for c in cols if c in df.columns]].copy()

    elif table_name == "business_transactions":
        p = processed_dir / "business_transactions.parquet"
        if not p.exists():
            p = unified_dir / "business_transactions_raw.csv"
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], utc=True)
        cols = [
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
        ]
        return df[[c for c in cols if c in df.columns]].copy()

    raise ValueError(f"Unknown table: {table_name}")


def load_dataframe_copy(
    cursor: Any, table_name: str, df: pd.DataFrame, chunk_size: int = 100000
) -> None:
    """Stream DataFrame into PostgreSQL via psycopg cursor.copy in chunks."""
    cols = list(df.columns)
    col_str = ", ".join(f'"{c}"' for c in cols)
    copy_sql = (
        f'COPY "{table_name}" ({col_str}) FROM STDIN WITH (FORMAT CSV, HEADER FALSE, NULL "")'
    )

    total_rows = len(df)
    logger.info("Streaming %d rows into %s using PostgreSQL COPY...", total_rows, table_name)

    for i in range(0, total_rows, chunk_size):
        chunk = df.iloc[i : i + chunk_size]
        buf = io.StringIO()
        chunk.to_csv(
            buf,
            index=False,
            header=False,
            na_rep="",
            date_format="%Y-%m-%d %H:%M:%S%z",
        )
        buf.seek(0)
        with cursor.copy(copy_sql) as copy:
            copy.write(buf.getvalue())


def run_database_loader(
    mode: str = "replace",
    dry_run: bool = False,
    target_tables: list[str] | None = None,
    processed_dir: Path | None = None,
    unified_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute atomic database loading with rollback protection and verification."""
    p_dir = processed_dir or PROCESSED
    u_dir = unified_dir or UNIFIED
    tables = [t for t in LOAD_ORDER if (target_tables is None or t in target_tables)]

    logger.info("Starting database loader (mode=%s, dry_run=%s)", mode, dry_run)
    counts: dict[str, int] = {}

    # Check database reachability
    is_connected = check_db()
    if not is_connected and not dry_run:
        msg = "database not reachable, check DATABASE_URL and that PostgreSQL is running"
        logger.error(msg)
        raise ConnectionError(msg)

    # 1. Prepare DataFrames and calculate expected counts
    dfs: dict[str, pd.DataFrame] = {}
    for tbl in tables:
        df = get_table_dataframe(tbl, p_dir, u_dir)
        dfs[tbl] = df
        counts[tbl] = len(df)
        logger.info("[Dry Run Plan] Table %s: %d rows to load", tbl, len(df))

    if dry_run:
        logger.info("Dry-run completed successfully. No changes applied.")
        return {"status": "DRY_RUN", "planned_counts": counts}

    # 2. Atomic loading in a single transaction
    is_postgres = "postgresql" in str(engine.url)

    if is_postgres:
        raw_conn = engine.raw_connection()
        try:
            cursor = raw_conn.cursor()
            # Truncate tables in reverse dependency order
            trunc_list = ", ".join(f'"{t}"' for t in TRUNCATE_ORDER if t in tables)
            logger.info("Truncating tables: %s", trunc_list)
            cursor.execute(f"TRUNCATE TABLE {trunc_list} CASCADE;")

            # Load tables in dependency order
            for tbl in tables:
                df = dfs[tbl]
                load_dataframe_copy(cursor, tbl, df)

            raw_conn.commit()
            logger.info("Transaction committed successfully.")

            # Run ANALYZE on loaded tables
            with engine.connect() as conn:
                for tbl in tables:
                    conn.execute(text(f'ANALYZE "{tbl}";'))
                conn.commit()
            logger.info("ANALYZE completed on loaded tables.")

        except Exception as exc:
            raw_conn.rollback()
            logger.error("Error during COPY load: %s. Rolling back transaction.", exc)
            raise
        finally:
            raw_conn.close()
    else:
        # SQLite fallback for unit testing / development
        logger.info("Using SQLAlchemy fallback engine: %s", engine.url)
        with engine.begin() as conn:
            for tbl in TRUNCATE_ORDER:
                if tbl in tables:
                    conn.execute(text(f'DELETE FROM "{tbl}";'))
            for tbl in tables:
                dfs[tbl].to_sql(tbl, conn, if_exists="append", index=False)
        logger.info("Fallback load completed successfully.")

    # 3. Post-load verification
    verification: dict[str, Any] = {}
    with engine.connect() as conn:
        for tbl in tables:
            res = conn.execute(text(f'SELECT COUNT(*) FROM "{tbl}";')).scalar()
            verification[tbl] = {
                "expected": counts[tbl],
                "actual": res,
                "matches": counts[tbl] == res,
            }
            logger.info("Verified %s: expected=%d, actual=%d", tbl, counts[tbl], res)

    return {"status": "SUCCESS", "loaded_counts": counts, "verification": verification}


def main() -> int:
    parser = argparse.ArgumentParser(description="Load cleaned datasets into database.")
    parser.add_argument(
        "--mode", default="replace", choices=["replace", "append"], help="Load mode"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print row counts and exit without loading"
    )
    parser.add_argument("--tables", nargs="*", default=None, help="Specific tables to load")
    args = parser.parse_args()

    try:
        run_database_loader(mode=args.mode, dry_run=args.dry_run, target_tables=args.tables)
        return 0
    except Exception as exc:
        logger.error("Database load failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
