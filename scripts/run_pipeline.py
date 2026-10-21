"""CLI entry point for running the Data Engineering Pipeline.

Usage:
    python scripts/run_pipeline.py --sample --skip-db
    python scripts/run_pipeline.py --skip-db
    python scripts/run_pipeline.py --only clean
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root and backend to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.preprocessing.pipeline import run_pipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run end-to-end data engineering pipeline for operations intelligence."
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run in fast sample mode (machines M-103 and M-104, trailing 45 days)",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip loading processed datasets into PostgreSQL / SQLite database",
    )
    parser.add_argument(
        "--only",
        dest="only_step",
        choices=["ingest", "validate_before"],
        default=None,
        help="Run only up to a specific initial step for debugging",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    try:
        result = run_pipeline(
            sample=args.sample,
            skip_db=args.skip_db,
            only_step=args.only_step,
            log_level=args.log_level,
        )
        return 0 if result.success else 1
    except Exception as exc:
        logger.exception("Pipeline failed with exception: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
