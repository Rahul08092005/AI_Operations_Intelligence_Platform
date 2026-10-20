"""AI4I Synthetic State Generator for AI Operations Intelligence Platform.

Standardizes AI4I 2020 Predictive Maintenance dataset into
data/raw/unified/ai4i_state_raw.csv (10,000 observations).
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

import pandas as pd

from ml.preprocessing.config import RAW, UNIFIED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_ai4i_state() -> pd.DataFrame:
    """Load and standardize AI4I dataset."""
    ai4i_csv = RAW / "ai4i" / "ai4i2020.csv"
    if not ai4i_csv.exists():
        raise FileNotFoundError(f"Missing {ai4i_csv}")

    logger.info("Loading AI4I dataset from %s...", ai4i_csv)
    df = pd.read_csv(ai4i_csv)

    rename_map = {
        "UDI": "udi",
        "Product ID": "product_id",
        "Type": "machine_variant",
        "Air temperature [K]": "air_temperature_k",
        "Process temperature [K]": "process_temperature_k",
        "Rotational speed [rpm]": "rotational_speed_rpm",
        "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min",
        "Machine failure": "machine_failure",
        "TWF": "twf",
        "HDF": "hdf",
        "PWF": "pwf",
        "OSF": "osf",
        "RNF": "rnf",
    }
    df = df.rename(columns=rename_map)

    df["machine_type"] = "Milling Machine"

    # Convert binary flags to boolean
    bool_cols = ["machine_failure", "twf", "hdf", "pwf", "osf", "rnf"]
    for col in bool_cols:
        df[col] = df[col].astype(bool)

    # Deterministic UUIDv5
    logger.info("Generating state UUIDs...")
    dns_ns = uuid.NAMESPACE_DNS
    df["state_id"] = [
        str(uuid.uuid5(dns_ns, f"ai4i-{row['udi']}-{row['product_id']}"))
        for _, row in df.iterrows()
    ]

    now_iso = datetime.now(tz=UTC).isoformat()
    df["created_at"] = now_iso

    cols = [
        "state_id",
        "udi",
        "product_id",
        "machine_type",
        "machine_variant",
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "machine_failure",
        "twf",
        "hdf",
        "pwf",
        "osf",
        "rnf",
        "created_at",
    ]
    return df[cols]


def run() -> None:
    """Generate and write ai4i_state_raw.csv."""
    UNIFIED.mkdir(parents=True, exist_ok=True)
    out_file = UNIFIED / "ai4i_state_raw.csv"

    df = generate_ai4i_state()
    logger.info("Standardized %d AI4I state records.", len(df))

    # Acceptance Checks
    assert len(df) == 10000, f"Expected 10,000 rows, got {len(df)}"
    assert (df["machine_variant"].isin(["L", "M", "H"])).all(), "Invalid variant"
    assert (df["air_temperature_k"] > 250.0).all(), "Unrealistic air temperature"
    assert (df["process_temperature_k"] > 250.0).all(), "Unrealistic process temperature"
    assert (df["rotational_speed_rpm"] > 500).all(), "Unrealistic RPM"

    fail_count = df["machine_failure"].sum()
    logger.info("Total machine failures in AI4I: %d (%.2f%%)", fail_count, fail_count / 100.0)
    assert 300 <= fail_count <= 400, f"Expected 300-400 failures, got {fail_count}"

    df.to_csv(out_file, index=False)
    logger.info("Successfully wrote %d AI4I state records to %s", len(df), out_file)


if __name__ == "__main__":
    run()
