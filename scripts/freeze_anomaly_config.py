"""Freeze anomaly detection model configurations and artifact hashes (Task T10).

Implements Decision D64:
- Freezes primary and real-only configurations before holdout evaluation.
- Records SHA-256 hashes of model artifacts and experiments log.
- Enforces strict single-run holdout discipline.

Usage:
    python scripts/freeze_anomaly_config.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.anomaly_detection.config import (  # noqa: E402
    MODELS_DIR,
    PROCESSED_DIR,
    STATUS_CRITICAL_THRESHOLD,
    STATUS_WARNING_THRESHOLD,
)
from ml.anomaly_detection.detector import AnomalyDetector  # noqa: E402
from ml.anomaly_detection.splits import get_periods  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("freeze_anomaly_config")


def compute_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    """Return current short git commit hash."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    primary_model_path = MODELS_DIR / "anomaly_model.pkl"
    real_only_model_path = MODELS_DIR / "anomaly_model_real_only.pkl"
    exp_log_path = PROCESSED_DIR / "anomaly_experiments.csv"
    frozen_config_path = MODELS_DIR / "anomaly_config_frozen.json"

    if not primary_model_path.exists() or not real_only_model_path.exists():
        logger.error("Model artifacts missing in %s. Run train_anomaly.py first.", MODELS_DIR)
        return 1

    if not exp_log_path.exists():
        logger.error("Experiments log missing in %s.", PROCESSED_DIR)
        return 1

    # Load detectors to extract exact configurations
    primary_detector = AnomalyDetector.load(primary_model_path)
    real_only_detector = AnomalyDetector.load(real_only_model_path)

    (exp_period, holdout_period) = get_periods()

    # Hashes
    primary_hash = compute_sha256(primary_model_path)
    real_only_hash = compute_sha256(real_only_model_path)
    exp_log_hash = compute_sha256(exp_log_path)

    frozen_dict = {
        "status": "frozen",
        "holdout_evaluated": False,
        "note": "holdout not yet evaluated",
        "frozen_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "exploration_period": {
            "start": str(exp_period[0]),
            "end": str(exp_period[1]),
        },
        "holdout_period": {
            "start": str(holdout_period[0]),
            "end": str(holdout_period[1]),
        },
        "artifact_hashes": {
            "primary_model_sha256": primary_hash,
            "real_only_model_sha256": real_only_hash,
            "experiments_log_sha256": exp_log_hash,
        },
        "primary_configuration": {
            "feature_set": primary_detector.feature_set_name,
            "feature_list": primary_detector.feature_list,
            "n_features": len(primary_detector.feature_list),
            "iforest_params": primary_detector.params,
            "post_params": primary_detector.post_params,
            "q_warn": primary_detector.post_params.get("q_warn"),
            "q_crit": primary_detector.post_params.get("q_crit"),
            "smoothing_window": primary_detector.smoothing_window,
            "status_thresholds": {
                "warning": STATUS_WARNING_THRESHOLD,
                "critical": STATUS_CRITICAL_THRESHOLD,
            },
        },
        "real_only_configuration": {
            "feature_set": real_only_detector.feature_set_name,
            "feature_list": real_only_detector.feature_list,
            "n_features": len(real_only_detector.feature_list),
            "iforest_params": real_only_detector.params,
            "post_params": real_only_detector.post_params,
            "q_warn": real_only_detector.post_params.get("q_warn"),
            "q_crit": real_only_detector.post_params.get("q_crit"),
            "smoothing_window": real_only_detector.smoothing_window,
            "status_thresholds": {
                "warning": STATUS_WARNING_THRESHOLD,
                "critical": STATUS_CRITICAL_THRESHOLD,
            },
        },
    }

    with open(frozen_config_path, "w", encoding="utf-8") as f:
        json.dump(frozen_dict, f, indent=2)

    logger.info("Saved frozen configuration to %s", frozen_config_path)

    print("\n" + "=" * 80)
    print("ANOMALY CONFIGURATION FROZEN (Task T10)")
    print(f"  Frozen Config File:      {frozen_config_path}")
    print(f"  Primary Model SHA-256:   {primary_hash}")
    print(f"  Real-Only Model SHA-256: {real_only_hash}")
    print(f"  Experiments Log SHA-256: {exp_log_hash}")
    print(f"  Git Commit:              {frozen_dict['git_commit']}")
    print(f"  Note:                    {frozen_dict['note']}")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
