"""Master data preparation orchestrator for AI Operations Intelligence Platform.

Executes all raw data extraction, generation, transformation, and validation
steps in strict dependency order, with optional determinism checking.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.preprocessing.config import SEED, UNIFIED  # noqa: E402
from ml.preprocessing.fleet import build_machines_df  # noqa: E402
from ml.preprocessing.generate_ai4i_state import run as run_ai4i  # noqa: E402
from ml.preprocessing.generate_business import run as run_business  # noqa: E402
from ml.preprocessing.generate_maintenance import run as run_maintenance  # noqa: E402
from ml.preprocessing.generate_production import run as run_production  # noqa: E402
from ml.preprocessing.generate_sensors import generate_all_sensors  # noqa: E402
from ml.preprocessing.standardize_retail import run as run_retail  # noqa: E402
from scripts.validate_raw_outputs import run_and_report as run_validation  # noqa: E402
from scripts.verify_sources import verify_sources as run_verification  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STEPS = [
    ("verify_sources", "Verifying raw downloaded archives and files", run_verification),
    ("machines", "Generating 12-machine fleet catalog", build_machines_df),
    ("sensors", "Generating high-frequency sensors telemetry", generate_all_sensors),
    ("maintenance", "Generating grounded maintenance records", run_maintenance),
    ("production", "Generating daily production dataset", run_production),
    ("retail", "Standardizing online retail transactions", run_retail),
    ("business", "Generating daily business metrics", run_business),
    ("ai4i", "Standardizing AI4I synthetic state", run_ai4i),
    ("validate", "Running comprehensive validation suite", run_validation),
]


def file_sha256(path: Path) -> str:
    """Calculate SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_determinism() -> bool:
    """Re-run pipeline generation and verify byte-level SHA256 reproducibility."""
    logger.info("=" * 70)
    logger.info("VERIFYING DETERMINISM & REPRODUCIBILITY (SEED=%d)...", SEED)
    logger.info("=" * 70)

    target_files = [
        "machines.csv",
        "maintenance_raw.csv",
        "production_raw.csv",
        "business_daily_raw.csv",
        "business_transactions_raw.csv",
        "ai4i_state_raw.csv",
    ]

    initial_hashes: dict[str, str] = {}
    for fname in target_files:
        fpath = UNIFIED / fname
        if not fpath.exists():
            logger.error("Missing file for determinism check: %s", fpath)
            return False
        initial_hashes[fname] = file_sha256(fpath)

    # Re-run deterministic generators in order
    build_machines_df()
    run_maintenance()
    run_production()
    run_retail()
    run_business()
    run_ai4i()

    matched = True
    for fname in target_files:
        fpath = UNIFIED / fname
        new_hash = file_sha256(fpath)
        if new_hash == initial_hashes[fname]:
            logger.info("[DETERMINISM PASS] %s -> %s", fname, new_hash[:16])
        else:
            logger.error(
                "[DETERMINISM FAIL] %s hash mismatch:\n  Original: %s\n  New:      %s",
                fname,
                initial_hashes[fname],
                new_hash,
            )
            matched = False

    return matched


def main() -> int:
    """CLI orchestrator for all data generation steps."""
    parser = argparse.ArgumentParser(description="AI Operations Platform - Data Orchestrator.")
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=[s[0] for s in STEPS] + ["all"],
        help="Target step to run (default: all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip steps whose output files already exist",
    )
    parser.add_argument(
        "--determinism",
        action="store_true",
        help="Verify byte-identical reproducibility across runs",
    )
    args = parser.parse_args()

    start_time = time.time()
    logger.info("Starting AI Operations Data Preparation Pipeline...")

    for step_id, desc, func in STEPS:
        if args.target != "all" and args.target != step_id:
            continue

        logger.info("-" * 60)
        logger.info("STEP [%s]: %s...", step_id.upper(), desc)
        logger.info("-" * 60)

        step_start = time.time()
        try:
            func()
            elapsed = time.time() - step_start
            logger.info("Completed [%s] in %.2f seconds.", step_id, elapsed)
        except Exception as e:
            logger.error("Failed during step [%s]: %s", step_id, e, exc_info=True)
            return 1

    if args.determinism:
        if not check_determinism():
            logger.error("Determinism check failed!")
            return 1
        logger.info("All outputs are 100% deterministic and reproducible!")

    total_elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Pipeline completed successfully in %.2f seconds!", total_elapsed)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
