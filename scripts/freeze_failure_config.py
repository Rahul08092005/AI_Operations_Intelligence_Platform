"""Pre-register and freeze failure prediction model configuration.

Implements Task T9, Rule R2, R5, R6, Decision D68, D77:
- Reads the trained model artifact metadata, hyperparameters, calibrator, thresholds.
- Computes SHA-256 hashes of:
  * models/failure_model.pkl
  * data/processed/failure_experiments.csv
  * data/processed/failure_oof_predictions.parquet
  * data/processed/ai4i_split_index.csv
- Records git commit, timestamp, and explicit statement "test partition not yet evaluated".
- Serializes to models/failure_config_frozen.json with status="frozen".
"""

from __future__ import annotations

import datetime
import json
import logging
import subprocess
import sys

from ml.prediction.config import (
    EXPERIMENTS_LOG_PATH,
    FROZEN_CONFIG_PATH,
    MODEL_METADATA_PATH,
    MODEL_PKL_PATH,
    OOF_PREDICTIONS_PATH,
    ROOT,
    SPLIT_INDEX_PATH,
)
from ml.prediction.data import compute_sha256

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("freeze_failure_config")


def get_git_commit_hash() -> str:
    """Retrieve current git HEAD commit hash if available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    logger.info("Executing Task T9: Freezing Failure Model Configuration...")

    if not MODEL_PKL_PATH.exists() or not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError("Model artifacts not found. Run training (Task T7) first.")

    with open(MODEL_METADATA_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    # Required artifacts to hash
    tracked_artifacts = [
        MODEL_PKL_PATH,
        EXPERIMENTS_LOG_PATH,
        OOF_PREDICTIONS_PATH,
        SPLIT_INDEX_PATH,
    ]

    artifact_hashes: dict[str, str] = {}
    for p in tracked_artifacts:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact for freeze: {p}")
        # Store path relative to repository ROOT using forward slashes
        rel_path = p.relative_to(ROOT).as_posix()
        sha = compute_sha256(p)
        artifact_hashes[rel_path] = sha
        logger.info("Hashed %s -> %s", rel_path, sha)

    now_utc = datetime.datetime.now(datetime.UTC).isoformat()
    git_hash = get_git_commit_hash()

    frozen_record = {
        "status": "frozen",
        "frozen_at_utc": now_utc,
        "git_commit": git_hash,
        "model_version": meta.get("model_version", "failure-xgb-1.0.0"),
        "evaluation_status": "test partition not yet evaluated",
        "primary_config_id": meta.get("oof_summary", {}).get("primary_config_id", "XGB-PHYS-08"),
        "feature_set": meta.get("feature_set", "physics"),
        "feature_names": meta.get("feature_names", []),
        "hyperparameters": meta.get("hyperparameters", {}),
        "calibrator_type": meta.get("calibrator_type", "IsotonicCalibrator"),
        "thresholds": meta.get("thresholds", {}),
        "oof_summary": meta.get("oof_summary", {}),
        "artifact_hashes": artifact_hashes,
        "statement": (
            "Cryptographic pre-registration complete. All training hyperparameters, "
            "calibrator settings, and risk thresholds are frozen. The test partition "
            "has not yet been loaded or evaluated."
        ),
    }

    FROZEN_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FROZEN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(frozen_record, f, indent=2)

    logger.info("Successfully frozen configuration to %s", FROZEN_CONFIG_PATH)
    logger.info("SHA-256 of frozen config: %s", compute_sha256(FROZEN_CONFIG_PATH))


if __name__ == "__main__":
    main()
