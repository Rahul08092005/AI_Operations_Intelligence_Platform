"""Guarded data access for AI4I predictive maintenance dataset.

Enforces Rule R2 and Decision D68:
- load_train() is always accessible.
- load_test() strictly raises unless models/failure_config_frozen.json exists
  and recorded artifact hashes match files on disk.
- load_demo_pool() requires frozen configuration and test log verification.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ml.eda.load import AI4I_RENAME
from ml.prediction.config import (
    FROZEN_CONFIG_PATH,
    RAW_AI4I_PATH,
    RAW_FEATURE_COLS,
    ROOT,
    SPLIT_INDEX_PATH,
)

logger = logging.getLogger(__name__)


def compute_sha256(path: Path | str) -> str:
    """Compute the SHA-256 hash of a file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _load_raw_ai4i() -> pd.DataFrame:
    """Load raw AI4I dataset and apply canonical column names."""
    if not RAW_AI4I_PATH.exists():
        raise FileNotFoundError(f"AI4I raw CSV not found at: {RAW_AI4I_PATH}")
    df = pd.read_csv(RAW_AI4I_PATH)
    df = df.rename(columns=AI4I_RENAME)
    return df


def _load_split_index() -> pd.DataFrame:
    """Load stratified train/test split index."""
    if not SPLIT_INDEX_PATH.exists():
        raise FileNotFoundError(f"AI4I split index not found at: {SPLIT_INDEX_PATH}")
    return pd.read_csv(SPLIT_INDEX_PATH)


def load_train() -> pd.DataFrame:
    """Load the AI4I training partition (8,000 rows, ~3.39% failure prevalence).

    Always accessible for model exploration, cross-validation, feature engineering,
    calibration, and threshold derivation.
    """
    raw_df = _load_raw_ai4i()
    split_df = _load_split_index()

    train_udis = set(split_df.loc[split_df["partition"] == "train", "udi"])
    train_df = raw_df[raw_df["udi"].isin(train_udis)].copy().reset_index(drop=True)

    if len(train_df) != 8000:
        raise ValueError(
            f"Expected 8,000 rows in AI4I training partition, found {len(train_df)}"
        )
    return train_df


def verify_frozen_config(frozen_config_path: Path | str | None = None) -> dict[str, Any]:
    """Verify that models/failure_config_frozen.json exists and hashes match."""
    config_path = (
        Path(frozen_config_path) if frozen_config_path is not None else FROZEN_CONFIG_PATH
    )

    if not config_path.exists():
        raise PermissionError(
            f"Guard violation: Cannot load test partition before configuration is frozen "
            f"in {config_path} (Rule R2, Decision D68)."
        )

    with open(config_path, encoding="utf-8") as f:
        frozen_data = json.load(f)

    if frozen_data.get("status") != "frozen":
        actual_status = frozen_data.get("status")
        raise PermissionError(
            f"Guard violation: Frozen config status is '{actual_status}', expected 'frozen'."
        )

    # Verify hashes of recorded artifacts
    artifact_hashes = frozen_data.get("artifact_hashes", {})
    if not artifact_hashes:
        raise PermissionError(
            "Guard violation: Frozen config contains no artifact hashes."
        )

    for rel_path, expected_hash in artifact_hashes.items():
        artifact_path = ROOT / rel_path
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Frozen artifact missing on disk: {artifact_path}"
            )
        actual_hash = compute_sha256(artifact_path)
        if actual_hash != expected_hash:
            raise PermissionError(
                f"Guard violation: Hash mismatch for {rel_path}. "
                f"Expected {expected_hash}, calculated {actual_hash}."
            )

    return frozen_data


def load_test(frozen_config_path: Path | str | None = None) -> pd.DataFrame:
    """Load the AI4I test partition (2,000 rows, ~3.40% failure prevalence).

    Guarded: strictly raises unless models/failure_config_frozen.json exists
    and all artifact hashes match files on disk (Rule R2).
    """
    verify_frozen_config(frozen_config_path)

    raw_df = _load_raw_ai4i()
    split_df = _load_split_index()

    test_udis = set(split_df.loc[split_df["partition"] == "test", "udi"])
    test_df = raw_df[raw_df["udi"].isin(test_udis)].copy().reset_index(drop=True)

    if len(test_df) != 2000:
        raise ValueError(
            f"Expected 2,000 rows in AI4I test partition, found {len(test_df)}"
        )
    return test_df


def load_demo_pool(frozen_config_path: Path | str | None = None) -> pd.DataFrame:
    """Load test partition rows classified into demo scenario pools (Decision D78).

    Guarded: requires verified frozen configuration.
    Pool boundaries (75th percentile of torque and tool wear) are computed strictly
    from the training partition to eliminate leakage into pool definitions.
    """
    verify_frozen_config(frozen_config_path)

    # Compute pool thresholds from training data only
    train_df = load_train()
    torque_p75 = float(train_df["torque_nm"].quantile(0.75))
    wear_p75 = float(train_df["tool_wear_min"].quantile(0.75))

    test_df = load_test(frozen_config_path)

    # Classify each test row into pools
    pools: list[str] = []
    for _, row in test_df.iterrows():
        is_fail = bool(row["machine_failure"] == 1)
        if is_fail:
            pools.append("high")
        elif (row["torque_nm"] > torque_p75) and (row["tool_wear_min"] > wear_p75):
            pools.append("stressed")
        elif (row["torque_nm"] <= torque_p75) and (row["tool_wear_min"] <= wear_p75):
            pools.append("low")
        else:
            # Intermediate non-failure rows
            pools.append("intermediate")

    test_df["pool"] = pools
    return test_df


def get_training_summary() -> dict[str, Any]:
    """Compute summary statistics and feature ranges for the training partition."""
    df = load_train()
    stats: dict[str, Any] = {
        "n_rows": len(df),
        "n_failures": int(df["machine_failure"].sum()),
        "failure_rate": float(df["machine_failure"].mean()),
        "feature_ranges": {},
    }

    numeric_features = [c for c in RAW_FEATURE_COLS if c != "type"]
    for col in numeric_features:
        s = df[col]
        min_val = float(s.min())
        max_val = float(s.max())
        span = max_val - min_val
        stats["feature_ranges"][col] = {
            "min": min_val,
            "max": max_val,
            "mean": float(s.mean()),
            "std": float(s.std()),
            "margin_min": min_val - 0.10 * span,
            "margin_max": max_val + 0.10 * span,
        }

    stats["type_counts"] = df["type"].value_counts().to_dict()
    return stats
