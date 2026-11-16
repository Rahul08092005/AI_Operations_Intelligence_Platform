"""Run failure model hyperparameter random search and ablations.

Executes 50 random draws from XGB_PARAM_GRID and 4 ablation/baseline runs.
Saves results to:
- data/processed/failure_experiments.csv
- data/processed/failure_oof_predictions.parquet
"""

from __future__ import annotations

import logging
import sys

from ml.prediction.search import run_experiments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_failure_search")


def main() -> None:
    logger.info("Starting Failure Model Hyperparameter Search...")
    experiments_df, primary_cfg, oof_df = run_experiments()

    logger.info("Search Complete!")
    logger.info("Total experiments recorded: %d", len(experiments_df))
    logger.info(
        "Primary Model Selected: %s (PR-AUC=%.4f, ROC-AUC=%.4f, Brier=%.4f)",
        primary_cfg["config_id"],
        primary_cfg["mean_pr_auc"],
        primary_cfg["mean_roc_auc"],
        primary_cfg["mean_brier"],
    )
    logger.info("OOF predictions shape: %s", oof_df.shape)


if __name__ == "__main__":
    main()

