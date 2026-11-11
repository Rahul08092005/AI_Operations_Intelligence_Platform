"""Train and save final anomaly detectors on exploration normal data (Task T9).

Usage:
    python scripts/train_anomaly.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from ml.anomaly_detection.config import MODELS_DIR, PROCESSED_DIR  # noqa: E402
from ml.anomaly_detection.detector import AnomalyDetector  # noqa: E402
from ml.anomaly_detection.experiments import select_primary, select_real_only  # noqa: E402
from ml.anomaly_detection.splits import get_periods  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_anomaly")


def main() -> int:
    features_path = PROCESSED_DIR / "sensor_features.parquet"
    labels_path = PROCESSED_DIR / "anomaly_eval_labels.parquet"
    machines_path = PROCESSED_DIR / "machines.parquet"
    exp_log_path = PROCESSED_DIR / "anomaly_experiments.csv"

    if not features_path.exists() or not labels_path.exists():
        logger.error("Missing input data files in %s", PROCESSED_DIR)
        return 1

    if not exp_log_path.exists():
        logger.error("Experiments log %s not found. Run CV first.", exp_log_path)
        return 1

    logger.info("Loading sensor features from %s", features_path)
    features_df = pd.read_parquet(features_path)
    logger.info("Loading anomaly evaluation labels from %s", labels_path)
    labels_df = pd.read_parquet(labels_path)
    experiments_df = pd.read_csv(exp_log_path)

    machines_df = None
    if machines_path.exists():
        machines_df = pd.read_parquet(machines_path)

    (exp_period, _) = get_periods()
    logger.info("Final fit exploration period: %s to %s", exp_period[0], exp_period[1])

    # 1. Select Primary & Real-only via pre-registered rules
    primary_sel = select_primary(experiments_df)
    real_only_sel = select_real_only(experiments_df)

    logger.info(
        "Primary Selected: %s (%s) | IF params: %s | Post: q_warn=%s, q_crit=%s, win=%s",
        primary_sel["config_id"],
        primary_sel["feature_set"],
        f"n_est={primary_sel['n_estimators']}, max_samp={primary_sel['max_samples']}",
        primary_sel["q_warn"],
        primary_sel["q_crit"],
        primary_sel["smoothing_window"],
    )

    logger.info(
        "Real-Only Selected: %s (%s) | Post: q_warn=%s, q_crit=%s, win=%s",
        real_only_sel["config_id"],
        real_only_sel["feature_set"],
        real_only_sel["q_warn"],
        real_only_sel["q_crit"],
        real_only_sel["smoothing_window"],
    )

    # 2. Fit Primary Detector on Exploration Normal Data
    logger.info("Fitting final Primary Anomaly Detector on exploration normal data...")
    primary_detector = AnomalyDetector().fit(
        features_df=features_df,
        labels_df=labels_df,
        period=exp_period,
        params={
            "n_estimators": int(primary_sel["n_estimators"]),
            "max_samples": int(primary_sel["max_samples"]),
            "max_features": float(primary_sel["max_features"]),
        },
        feature_set=str(primary_sel["feature_set"]),
        post={
            "q_warn": float(primary_sel["q_warn"]),
            "q_crit": float(primary_sel["q_crit"]),
            "smoothing_window": int(primary_sel["smoothing_window"]),
        },
        machines_df=machines_df,
    )

    primary_model_path = MODELS_DIR / "anomaly_model.pkl"
    primary_meta_path = MODELS_DIR / "anomaly_model_metadata.json"
    primary_detector.save(primary_model_path, primary_meta_path)
    logger.info(
        "Saved primary model to %s and metadata to %s",
        primary_model_path,
        primary_meta_path,
    )

    # 3. Fit Real-Only Comparator Detector on Exploration Normal Data
    logger.info("Fitting final Real-Only Comparator on exploration normal data...")
    real_only_detector = AnomalyDetector().fit(
        features_df=features_df,
        labels_df=labels_df,
        period=exp_period,
        params={
            "n_estimators": int(real_only_sel["n_estimators"]),
            "max_samples": int(real_only_sel["max_samples"]),
            "max_features": float(real_only_sel["max_features"]),
        },
        feature_set=str(real_only_sel["feature_set"]),
        post={
            "q_warn": float(real_only_sel["q_warn"]),
            "q_crit": float(real_only_sel["q_crit"]),
            "smoothing_window": int(real_only_sel["smoothing_window"]),
        },
        machines_df=machines_df,
    )

    real_only_model_path = MODELS_DIR / "anomaly_model_real_only.pkl"
    real_only_detector.save(real_only_model_path)
    logger.info("Saved real-only comparator to %s", real_only_model_path)

    # 4. Acceptance check: Load both and score a sample
    sample_df = features_df.head(1000).copy()
    loaded_primary = AnomalyDetector.load(primary_model_path)
    scored_primary = loaded_primary.score_frame(sample_df)
    assert "anomaly_score" in scored_primary.columns
    assert "anomaly_status" in scored_primary.columns
    assert "factor_1" in scored_primary.columns

    loaded_real = AnomalyDetector.load(real_only_model_path)
    scored_real = loaded_real.score_frame(sample_df)
    assert "anomaly_score" in scored_real.columns

    n_prim = len(primary_detector.feature_list)
    n_real = len(real_only_detector.feature_list)
    n_rows = primary_detector.n_training_rows

    print("\n" + "=" * 80)
    print("FINAL MODELS TRAINED AND SAVED SUCCESSFULLY (Task T9)")
    print(f"  Primary Model:    {primary_model_path} ({n_prim} features)")
    print(f"  Real-Only Model:  {real_only_model_path} ({n_real} features)")
    print(f"  Training Rows:    {n_rows:,} normal exploration intervals")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())