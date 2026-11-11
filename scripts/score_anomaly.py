"""Score all machines across the full timeline and export fleet status artifacts (Task T12).

Outputs:
- data/processed/anomaly_scores.parquet: full machine-interval table
- data/processed/anomaly_episodes.parquet: primary model alert episodes
- data/processed/anomaly_latest_status.csv: 12 rows (latest scored status per machine)

Usage:
    python scripts/score_anomaly.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.anomaly_detection.config import MODELS_DIR, PROCESSED_DIR  # noqa: E402
from ml.anomaly_detection.detector import AnomalyDetector  # noqa: E402
from ml.anomaly_detection.episodes import extract_alert_episodes  # noqa: E402
from ml.anomaly_detection.splits import cv_folds, get_periods  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("score_anomaly")


def main() -> int:
    primary_model_path = MODELS_DIR / "anomaly_model.pkl"
    real_only_model_path = MODELS_DIR / "anomaly_model_real_only.pkl"
    features_path = PROCESSED_DIR / "sensor_features.parquet"
    sensors_path = PROCESSED_DIR / "sensors.parquet"

    out_scores_path = PROCESSED_DIR / "anomaly_scores.parquet"
    out_episodes_path = PROCESSED_DIR / "anomaly_episodes.parquet"
    out_status_path = PROCESSED_DIR / "anomaly_latest_status.csv"

    if not primary_model_path.exists() or not real_only_model_path.exists():
        logger.error("Model artifacts missing in %s. Run train_anomaly.py first.", MODELS_DIR)
        return 1

    if not features_path.exists():
        logger.error("sensor_features.parquet missing in %s.", PROCESSED_DIR)
        return 1

    logger.info("Loading models and sensor features...")
    primary_detector = AnomalyDetector.load(primary_model_path)
    real_only_detector = AnomalyDetector.load(real_only_model_path)
    features_df = pd.read_parquet(features_path)

    n_rows = len(features_df)
    logger.info("Scoring full timeline (%d intervals) with Primary Model...", n_rows)
    primary_scored = primary_detector.score_frame(features_df)

    logger.info("Scoring full timeline with Real-Only Comparator Model...")
    real_only_scored = real_only_detector.score_frame(features_df)

    # Attach real-only score and status
    primary_scored["anomaly_score_real_only"] = real_only_scored["anomaly_score"]
    primary_scored["anomaly_status_real_only"] = real_only_scored["anomaly_status"]

    # Assign period column: train, validation, holdout
    folds = cv_folds()
    (_, (holdout_start, _)) = get_periods()

    period_col = np.full(n_rows, "train", dtype="object")

    # Mark validation folds
    for f in folds:
        v_idx = (features_df["timestamp"] >= f.val_start) & (features_df["timestamp"] < f.val_end)
        period_col[v_idx.to_numpy()] = "validation"

    # Mark holdout
    h_idx = features_df["timestamp"] >= holdout_start
    period_col[h_idx.to_numpy()] = "holdout"

    primary_scored["period"] = pd.Series(period_col, index=features_df.index, dtype="string")
    primary_scored["model_version"] = "1.0.0"

    # Check alignment with sensors.parquet
    if sensors_path.exists():
        sensors_df = pd.read_parquet(sensors_path)
        if len(sensors_df) != len(primary_scored):
            logger.warning(
                "Row count delta: sensors.parquet has %d rows, scores has %d rows.",
                len(sensors_df),
                len(primary_scored),
            )

    # 1. Save data/processed/anomaly_scores.parquet
    logger.info("Saving anomaly scores to %s...", out_scores_path)
    primary_scored.to_parquet(out_scores_path, index=False)

    # 2. Extract and save alert episodes
    logger.info("Extracting alert episodes across the fleet...")
    episodes_df = extract_alert_episodes(primary_scored)
    logger.info("Extracted %d alert episodes. Saving to %s...", len(episodes_df), out_episodes_path)
    episodes_df.to_parquet(out_episodes_path, index=False)

    # 3. Save anomaly_latest_status.csv (12 rows, one per machine at last scored timestamp)
    logger.info("Extracting latest scored status per machine...")
    status_rows: list[dict[str, object]] = []

    for m_id, grp in primary_scored.groupby("machine_id", observed=True):
        scored_grp = grp[grp["anomaly_score"].notna()]
        if len(scored_grp) > 0:
            last_scored = scored_grp.sort_values("timestamp").iloc[-1]
        else:
            last_scored = grp.sort_values("timestamp").iloc[-1]

        status_rows.append(
            {
                "machine_id": str(m_id),
                "timestamp": str(last_scored["timestamp"]),
                "state": str(last_scored["state"]),
                "anomaly_score": (
                    round(float(last_scored["anomaly_score"]), 4)
                    if pd.notna(last_scored["anomaly_score"])
                    else np.nan
                ),
                "anomaly_status": str(last_scored["anomaly_status"]),
                "factor_1": (
                    str(last_scored["factor_1"]) if pd.notna(last_scored["factor_1"]) else ""
                ),
                "factor_1_z": (
                    round(float(last_scored["factor_1_z"]), 2)
                    if pd.notna(last_scored["factor_1_z"])
                    else np.nan
                ),
                "factor_1_direction": (
                    str(last_scored["factor_1_direction"])
                    if pd.notna(last_scored["factor_1_direction"])
                    else ""
                ),
                "factor_2": (
                    str(last_scored["factor_2"]) if pd.notna(last_scored["factor_2"]) else ""
                ),
                "factor_2_z": (
                    round(float(last_scored["factor_2_z"]), 2)
                    if pd.notna(last_scored["factor_2_z"])
                    else np.nan
                ),
                "factor_2_direction": (
                    str(last_scored["factor_2_direction"])
                    if pd.notna(last_scored["factor_2_direction"])
                    else ""
                ),
                "factor_3": (
                    str(last_scored["factor_3"]) if pd.notna(last_scored["factor_3"]) else ""
                ),
                "factor_3_z": (
                    round(float(last_scored["factor_3_z"]), 2)
                    if pd.notna(last_scored["factor_3_z"])
                    else np.nan
                ),
                "factor_3_direction": (
                    str(last_scored["factor_3_direction"])
                    if pd.notna(last_scored["factor_3_direction"])
                    else ""
                ),
                "anomaly_score_real_only": (
                    round(float(last_scored["anomaly_score_real_only"]), 4)
                    if pd.notna(last_scored["anomaly_score_real_only"])
                    else np.nan
                ),
                "anomaly_status_real_only": str(last_scored["anomaly_status_real_only"]),
            }
        )

    df_latest = pd.DataFrame(status_rows).sort_values("machine_id")
    df_latest.to_csv(out_status_path, index=False)
    logger.info("Saved latest status (%d machines) to %s", len(df_latest), out_status_path)

    print("\n" + "=" * 80)
    print("ANOMALY SCORING AND FLEET STATUS EXPORT COMPLETE (Task T12)")
    print(f"  Anomaly Scores:  {out_scores_path} ({len(primary_scored):,} rows)")
    print(f"  Alert Episodes:  {out_episodes_path} ({len(episodes_df)} episodes)")
    print(f"  Latest Status:   {out_status_path} ({len(df_latest)} machines)")
    print("=" * 80 + "\n")
    cols = ["machine_id", "timestamp", "anomaly_score", "anomaly_status", "factor_1"]
    print(df_latest[cols].to_string(index=False))
    print("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
