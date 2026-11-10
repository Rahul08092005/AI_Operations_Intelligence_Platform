"""Run cross-validated Isolation Forest experiments and baseline comparisons (Task T7).

Usage:
    python scripts/run_anomaly_cv.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from ml.anomaly_detection.config import PROCESSED_DIR
from ml.anomaly_detection.experiments import (
    run_cv_experiments,
    select_primary,
    select_real_only,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_anomaly_cv")


def main() -> int:
    features_path = PROCESSED_DIR / "sensor_features.parquet"
    labels_path = PROCESSED_DIR / "anomaly_eval_labels.parquet"
    machines_path = PROCESSED_DIR / "machines.parquet"

    if not features_path.exists() or not labels_path.exists():
        logger.error("Missing input data files in %s", PROCESSED_DIR)
        return 1

    logger.info("Loading sensor features from %s", features_path)
    features_df = pd.read_parquet(features_path)
    logger.info("Loading anomaly evaluation labels from %s", labels_path)
    labels_df = pd.read_parquet(labels_path)

    machines_df = None
    if machines_path.exists():
        machines_df = pd.read_parquet(machines_path)

    # Run experiments
    out_csv = PROCESSED_DIR / "anomaly_experiments.csv"
    experiments_df = run_cv_experiments(
        features_df=features_df,
        labels_df=labels_df,
        machines_df=machines_df,
        output_csv=out_csv,
    )

    # Selection
    primary_sel = select_primary(experiments_df)
    real_only_sel = select_real_only(experiments_df)

    print("\n" + "=" * 80)
    print("ANOMALY DETECTION CV EXPERIMENTS SUMMARY (Task T7)")
    print("=" * 80)

    # Print best variant per feature set
    feature_sets = experiments_df["feature_set"].unique()
    summary_rows = []
    for fs in feature_sets:
        fs_df = experiments_df[experiments_df["feature_set"] == fs]
        pass_df = fs_df[fs_df["budget_passed"]]
        best_row = (
            pass_df.sort_values(
                ["episode_recall", "median_lead_time_hours"], ascending=[False, False]
            ).iloc[0]
            if len(pass_df) > 0
            else fs_df.sort_values(
                ["total_critical_fa", "pooled_warning_fa_per_30d"], ascending=[True, True]
            ).iloc[0]
        )
        summary_rows.append(
            {
                "Feature Set": fs,
                "Detector": best_row["detector"],
                "Config ID": best_row["config_id"],
                "Budget": "PASS" if best_row["budget_passed"] else "FAIL",
                "Warn FA/30d": best_row["pooled_warning_fa_per_30d"],
                "Crit FA": best_row["total_critical_fa"],
                "Recall": f"{best_row['episode_recall']:.2f}",
                "Episodes Det.": (
                    f"{best_row['episodes_detected']}/{best_row['episodes_total']}"
                ),
                "Med Lead (h)": best_row["median_lead_time_hours"],
                "q_warn": best_row["q_warn"],
                "q_crit": best_row["q_crit"],
                "Window": best_row["smoothing_window"],
            }
        )

    df_sum = pd.DataFrame(summary_rows)
    print(df_sum.to_string(index=False))

    print("\n" + "-" * 80)
    print("SELECTED PRIMARY CONFIGURATION:")
    print(f"  Config ID:     {primary_sel['config_id']}")
    print(
        f"  Feature Set:   {primary_sel['feature_set']} "
        f"({primary_sel['n_features']} features)"
    )
    print(
        f"  IF Params:     n_est={primary_sel['n_estimators']}, "
        f"max_samples={primary_sel['max_samples']}, "
        f"max_feat={primary_sel['max_features']}"
    )
    print(
        f"  Post-proc:     q_warn={primary_sel['q_warn']}, "
        f"q_crit={primary_sel['q_crit']}, "
        f"window={primary_sel['smoothing_window']} intervals"
    )
    print(
        f"  Performance:   Recall={primary_sel['episode_recall']:.2f} "
        f"({primary_sel['episodes_detected']}/{primary_sel['episodes_total']}), "
        f"Lead={primary_sel['median_lead_time_hours']:.1f}h"
    )
    print(
        f"  Budget:        PASS={primary_sel['budget_passed']} "
        f"({primary_sel['pooled_warning_fa_per_30d']:.2f} Warn FA/30d, "
        f"{primary_sel['total_critical_fa']} Crit FA)"
    )
    print(f"  Rationale:     {primary_sel['selection_reason']}")

    print("\nSELECTED REAL-ONLY COMPARATOR (Set D):")
    print(f"  Config ID:     {real_only_sel['config_id']}")
    print(
        f"  Feature Set:   {real_only_sel['feature_set']} "
        f"({real_only_sel['n_features']} features)"
    )
    print(
        f"  IF Params:     n_est={real_only_sel['n_estimators']}, "
        f"max_samples={real_only_sel['max_samples']}, "
        f"max_feat={real_only_sel['max_features']}"
    )
    print(
        f"  Post-proc:     q_warn={real_only_sel['q_warn']}, "
        f"q_crit={real_only_sel['q_crit']}, "
        f"window={real_only_sel['smoothing_window']} intervals"
    )
    print(
        f"  Performance:   Recall={real_only_sel['episode_recall']:.2f} "
        f"({real_only_sel['episodes_detected']}/{real_only_sel['episodes_total']}), "
        f"Lead={real_only_sel['median_lead_time_hours']:.1f}h"
    )
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
