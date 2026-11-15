"""Build and save demo machine operating states and risk tables.

Implements Task T11:
- Outputs:
  * data/processed/machine_ai4i_state.parquet (360 rows)
  * data/processed/machine_risk_daily.parquet (360 rows)
  * data/processed/machine_risk_latest.csv (12 rows, is_replay=True)
- Verifies that all sampled rows originate from the test partition only.
"""

from __future__ import annotations

import logging
import sys

from ml.prediction.demo_states import build_and_save_machine_risk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("build_machine_risk")


def main() -> None:
    logger.info("Executing Task T11: Building demo machine states and risk tables...")
    states_df, daily_df, latest_df = build_and_save_machine_risk()

    logger.info("Demo states rows: %d", len(states_df))
    logger.info("Daily risk rows: %d", len(daily_df))
    logger.info(
        "Latest risk rows: %d (machines: %d)",
        len(latest_df),
        latest_df["machine_id"].nunique(),
    )
    logger.info("\nLatest Risk Summary (Replay Data):\n%s", latest_df.to_string(index=False))
    logger.info("Task T11 successfully completed!")


if __name__ == "__main__":
    main()
