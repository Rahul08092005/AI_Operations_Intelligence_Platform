"""CLI utility to generate and verify the AI4I 80/20 stratified train/test split index."""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.eda.load import load_ai4i  # noqa: E402
from ml.eda.splits import save_ai4i_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    logger.info("Generating AI4I stratified 80/20 train/test split index...")
    split_path = save_ai4i_split()
    logger.info("Saved AI4I split index to %s", split_path)

    # Verify loading strictly the training partition
    train_df = load_ai4i(partition="train")
    full_df = load_ai4i(partition=None)

    n_train = len(train_df)
    n_full = len(full_df)
    train_failures = int(train_df["machine_failure"].sum())
    full_failures = int(full_df["machine_failure"].sum())

    train_rate = train_df["machine_failure"].mean()
    full_rate = full_df["machine_failure"].mean()

    logger.info("Full dataset: %d rows, %d failures (%.4f)", n_full, full_failures, full_rate)
    logger.info(
        "Train partition: %d rows, %d failures (%.4f)", n_train, train_failures, train_rate
    )

    assert n_train == 8000, f"Expected 8000 train rows, got {n_train}"
    diff = abs(train_rate - full_rate)
    assert diff < 0.005, f"Failure prevalence mismatch: {diff:.4f}"

    logger.info("AI4I split verification PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
