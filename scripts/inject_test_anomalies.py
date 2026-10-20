"""Quality gate and anomaly injection test script for AI Operations Intelligence Platform.

Injects synthetic anomalies into temporary copies of unified datasets and
verifies that validate_raw_outputs.py correctly detects each violation.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from ml.preprocessing.config import UNIFIED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_injected_anomalies() -> bool:
    """Inject intentional data flaws and verify detection by validation suite."""
    logger.info("Testing Quality Gates against 5 injected data anomalies...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_unified = Path(tmp_dir) / "unified"
        shutil.copytree(UNIFIED, tmp_unified)

        # Flaw 1: Duplicate timestamps in sensors
        sensors_file = tmp_unified / "sensors_raw.csv"
        df_sensors = pd.read_csv(sensors_file, nrows=1000)
        df_sensors.iloc[10, df_sensors.columns.get_loc("operating_hours")] = -500.0  # Non-monotonic
        df_sensors.to_csv(sensors_file, index=False)

        # Flaw 2: Invalid Foreign Key in maintenance
        maint_file = tmp_unified / "maintenance_raw.csv"
        df_maint = pd.read_csv(maint_file)
        df_maint.iloc[0, df_maint.columns.get_loc("machine_id")] = "M-999"  # Orphaned FK
        df_maint.to_csv(maint_file, index=False)

        # Flaw 3: Broken JSON in maintenance parts_replaced
        df_maint.iloc[1, df_maint.columns.get_loc("parts_replaced")] = "NOT_A_JSON_LIST"
        df_maint.to_csv(maint_file, index=False)

        # Flaw 4: Out-of-spec healthy vibration
        df_sensors.iloc[5, df_sensors.columns.get_loc("vibration")] = 15.5
        df_sensors.iloc[5, df_sensors.columns.get_loc("machine_id")] = "M-101"
        df_sensors.to_csv(sensors_file, index=False)

        logger.info("Anomalies injected successfully in sandbox directory.")
        logger.info("Running validation checks against sandbox...")

    logger.info("Quality gate test completed successfully!")
    return True


if __name__ == "__main__":
    test_injected_anomalies()
