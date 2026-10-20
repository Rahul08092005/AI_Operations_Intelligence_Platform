"""Database initialization and schema migration script for AI Operations Intelligence Platform.

Creates all required relational tables in PostgreSQL / SQLAlchemy Base,
with optional bulk-loading of standardized raw datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect

# Allow importing backend app when run from root
backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import app.models  # noqa: E402, F401 (registers models on Base.metadata)
from app.config import get_settings  # noqa: E402
from app.database.session import Base, check_db, engine  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
UNIFIED = ROOT / "data" / "raw" / "unified"
REFERENCE = ROOT / "ml" / "preprocessing" / "reference"


def init_schema(target_engine=None) -> bool:
    """Create all tables defined on Base metadata."""
    eng = target_engine or engine
    logger.info("Creating all tables in target database...")
    Base.metadata.create_all(bind=eng)
    inspector = inspect(eng)
    tables = inspector.get_table_names()
    logger.info("Database schema initialized. Tables present: %s", tables)
    return True


def load_unified_data(target_engine=None) -> None:
    """Load unified CSV data into database tables."""
    eng = target_engine or engine
    logger.info("Loading unified datasets into database...")

    # 1. Machines
    machines_file = UNIFIED / "machines.csv"
    if machines_file.exists():
        logger.info("Loading %s -> machines...", machines_file)
        df_m = pd.read_csv(machines_file)
        df_m["installation_date"] = pd.to_datetime(df_m["installation_date"]).dt.date
        df_m.to_sql("machines", eng, if_exists="append", index=False)

    # 2. Reference failure windows
    fw_file = REFERENCE / "metropt_failure_windows.csv"
    if fw_file.exists():
        logger.info("Loading %s -> metropt_failure_windows...", fw_file)
        df_fw = pd.read_csv(fw_file)
        df_fw["start_time"] = pd.to_datetime(df_fw["start_time"], utc=True)
        df_fw["end_time"] = pd.to_datetime(df_fw["end_time"], utc=True)
        df_fw.to_sql("metropt_failure_windows", eng, if_exists="append", index=False)

    # 3. Maintenance Records
    maint_file = UNIFIED / "maintenance_raw.csv"
    if maint_file.exists():
        logger.info("Loading %s -> maintenance_records...", maint_file)
        df_maint = pd.read_csv(maint_file)
        df_maint["scheduled_date"] = pd.to_datetime(df_maint["scheduled_date"], utc=True)
        df_maint["performed_date"] = pd.to_datetime(df_maint["performed_date"], utc=True)
        df_maint["created_at"] = pd.to_datetime(df_maint["created_at"], utc=True)
        df_maint["parts_replaced"] = df_maint["parts_replaced"].apply(
            lambda x: json.loads(x) if isinstance(x, str) and x else None
        )
        df_maint.to_sql("maintenance_records", eng, if_exists="append", index=False)

    # 4. Production Daily
    prod_file = UNIFIED / "production_raw.csv"
    if prod_file.exists():
        logger.info("Loading %s -> production_daily...", prod_file)
        df_prod = pd.read_csv(prod_file)
        df_prod["date"] = pd.to_datetime(df_prod["date"]).dt.date
        df_prod["created_at"] = pd.to_datetime(df_prod["created_at"], utc=True)
        df_prod.to_sql("production_daily", eng, if_exists="append", index=False)

    # 5. Business Daily
    biz_file = UNIFIED / "business_daily_raw.csv"
    if biz_file.exists():
        logger.info("Loading %s -> business_daily...", biz_file)
        df_biz = pd.read_csv(biz_file)
        df_biz["date"] = pd.to_datetime(df_biz["date"]).dt.date
        df_biz["created_at"] = pd.to_datetime(df_biz["created_at"], utc=True)
        df_biz.to_sql("business_daily", eng, if_exists="append", index=False)

    # 6. AI4I State
    ai4i_file = UNIFIED / "ai4i_state_raw.csv"
    if ai4i_file.exists():
        logger.info("Loading %s -> ai4i_state...", ai4i_file)
        df_ai4i = pd.read_csv(ai4i_file)
        df_ai4i["created_at"] = pd.to_datetime(df_ai4i["created_at"], utc=True)
        df_ai4i.to_sql("ai4i_state", eng, if_exists="append", index=False)

    # 7. Sensors Raw (chunked for memory efficiency)
    sensors_file = UNIFIED / "sensors_raw.csv"
    if sensors_file.exists():
        logger.info("Loading %s -> sensors_raw (chunked)...", sensors_file)
        for chunk in pd.read_csv(sensors_file, chunksize=50000):
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True)
            chunk.to_sql("sensors_raw", eng, if_exists="append", index=False)

    # 8. Business Transactions (chunked)
    trans_file = UNIFIED / "business_transactions_raw.csv"
    if trans_file.exists():
        logger.info("Loading %s -> business_transactions (chunked)...", trans_file)
        for chunk in pd.read_csv(trans_file, chunksize=50000):
            chunk["transaction_date"] = pd.to_datetime(chunk["transaction_date"], utc=True)
            chunk.to_sql("business_transactions", eng, if_exists="append", index=False)

    logger.info("All available unified data loaded successfully!")


def main() -> int:
    """CLI entrypoint for database initialization."""
    parser = argparse.ArgumentParser(
        description="Initialize PostgreSQL schema for AI Operations Platform."
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Only check database connectivity"
    )
    parser.add_argument(
        "--load-data", action="store_true", help="Bulk load unified CSV datasets into tables"
    )
    parser.add_argument(
        "--sqlite-fallback", action="store_true", help="Use SQLite if PostgreSQL is unavailable"
    )
    parser.add_argument(
        "--sqlite-path", type=str, default="", help="Optional SQLite database path"
    )
    args = parser.parse_args()

    settings = get_settings()

    if args.sqlite_path:
        target_eng = create_engine(f"sqlite:///{args.sqlite_path}")
        logger.info("Using explicit SQLite database: %s", args.sqlite_path)
        init_schema(target_eng)
        if args.load_data:
            load_unified_data(target_eng)
        return 0

    # PostgreSQL check
    db_available = check_db()
    if not db_available:
        logger.warning("PostgreSQL database is UNREACHABLE at %s", settings.database_url)
        logger.warning("To start PostgreSQL container: docker compose up -d db")
        if args.sqlite_fallback:
            logger.info("Falling back to local SQLite database: data/aiops.db")
            sqlite_file = ROOT / "data" / "aiops.db"
            target_eng = create_engine(f"sqlite:///{sqlite_file}")
            init_schema(target_eng)
            if args.load_data:
                load_unified_data(target_eng)
            return 0
        if args.check_only:
            return 1
        logger.info("Skipping initialization until PostgreSQL is started.")
        return 0

    logger.info("Successfully connected to PostgreSQL at %s", settings.database_url)
    if args.check_only:
        return 0

    init_schema(engine)
    if args.load_data:
        load_unified_data(engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
