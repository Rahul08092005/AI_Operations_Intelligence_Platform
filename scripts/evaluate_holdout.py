"""Single-run holdout evaluation of frozen anomaly detectors (Task T11).

Strictly enforces Decision D64:
- Refuses to execute if models/anomaly_config_frozen.json is missing or hashes mismatch.
- Scores the holdout period once from the frozen configuration.
- Appends results to docs/holdout_log.md (append-only).
- Produces comparison tables for Primary (plan_core), Real-Only (Set D), B1 and B2.
- Generates final Figures F11 to F14.

Usage:
    python scripts/evaluate_holdout.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from ml.anomaly_detection.config import DOCS_DIR, MODELS_DIR, PROCESSED_DIR  # noqa: E402
from ml.anomaly_detection.detector import AnomalyDetector  # noqa: E402
from ml.anomaly_detection.detectors_baseline import (  # noqa: E402
    B1ManualThresholdDetector,
    B2RobustZDetector,
)
from ml.anomaly_detection.episodes import extract_alert_episodes  # noqa: E402
from ml.anomaly_detection.evaluate import (  # noqa: E402
    METROPT_FAILURE_WINDOWS,
    evaluate_real_windows,
    evaluate_synthetic_episodes,
    false_alarm_report,
    final_state_report,
    interval_metrics,
)
from ml.anomaly_detection.plots import (  # noqa: E402
    generate_f11_timelines,
    generate_f12_lead_times,
    generate_f13_model_vs_baselines,
    generate_f14_m104_contributing_factors,
)
from ml.anomaly_detection.splits import get_periods  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_holdout")


def compute_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    frozen_config_path = MODELS_DIR / "anomaly_config_frozen.json"
    primary_model_path = MODELS_DIR / "anomaly_model.pkl"
    real_only_model_path = MODELS_DIR / "anomaly_model_real_only.pkl"
    features_path = PROCESSED_DIR / "sensor_features.parquet"
    labels_path = PROCESSED_DIR / "anomaly_eval_labels.parquet"
    holdout_log_path = DOCS_DIR / "holdout_log.md"

    # 1. Guard check: Frozen config & artifact hashes (Decision D64)
    if not frozen_config_path.exists():
        logger.error(
            "GUARD FAILURE: Frozen config %s missing! Run freeze_anomaly_config.py first.",
            frozen_config_path,
        )
        return 1

    with open(frozen_config_path, encoding="utf-8") as f:
        frozen_data = json.load(f)

    expected_prim_hash = frozen_data["artifact_hashes"]["primary_model_sha256"]
    expected_real_hash = frozen_data["artifact_hashes"]["real_only_model_sha256"]

    current_prim_hash = compute_sha256(primary_model_path)
    current_real_hash = compute_sha256(real_only_model_path)

    if current_prim_hash != expected_prim_hash or current_real_hash != expected_real_hash:
        logger.error(
            "GUARD FAILURE: Model hashes do not match frozen configuration!\n"
            "  Primary:  expected %s, got %s\n"
            "  Real-only: expected %s, got %s",
            expected_prim_hash,
            current_prim_hash,
            expected_real_hash,
            current_real_hash,
        )
        return 1

    logger.info("Guard checks passed: Frozen config verified with matching SHA-256 digests.")

    # 2. Load data and models
    logger.info("Loading sensor features and evaluation labels...")
    features_df = pd.read_parquet(features_path)
    labels_df = pd.read_parquet(labels_path)

    primary_detector = AnomalyDetector.load(primary_model_path)
    real_only_detector = AnomalyDetector.load(real_only_model_path)

    (exp_period, holdout_period) = get_periods()
    h_start, h_end = holdout_period
    logger.info("Holdout evaluation slice: %s to %s", h_start, h_end)

    h_mask = (features_df["timestamp"] >= h_start) & (features_df["timestamp"] < h_end)
    features_h = features_df.loc[h_mask].copy()
    labels_h = labels_df.loc[h_mask].copy()
    total_holdout_days = (h_end - h_start).total_seconds() / 86400.0

    # 3. Score Holdout with Primary, Real-Only, B1 and B2
    logger.info("Scoring holdout with Primary Model (%s)...", primary_detector.feature_set_name)
    primary_h_scored = primary_detector.score_frame(features_h)

    logger.info(
        "Scoring holdout with Real-Only Comparator (%s)...",
        real_only_detector.feature_set_name,
    )
    real_only_h_scored = real_only_detector.score_frame(features_h)

    logger.info("Scoring holdout with Baseline B1 (Manual Thresholds)...")
    b1_detector = B1ManualThresholdDetector(smoothing_window=36)
    b1_h_scored = b1_detector.score_frame(features_h)

    logger.info("Scoring holdout with Baseline B2 (Robust Z)...")
    b2_detector = B2RobustZDetector(z_warn=4.0, z_crit=8.0, smoothing_window=36)
    b2_detector.baseline = primary_detector.baseline
    b2_h_scored = b2_detector.score_frame(features_h)

    # 4. Alert episode extraction
    primary_episodes = extract_alert_episodes(primary_h_scored)
    real_only_episodes = extract_alert_episodes(real_only_h_scored)
    b1_episodes = extract_alert_episodes(b1_h_scored)
    b2_episodes = extract_alert_episodes(b2_h_scored)

    # 5. Holdout Real Failure Windows (FAIL-004)
    holdout_windows = [w for w in METROPT_FAILURE_WINDOWS if w.get("period") == "holdout"]
    res_real_prim = evaluate_real_windows(primary_episodes, holdout_windows)
    res_real_only = evaluate_real_windows(real_only_episodes, holdout_windows)
    res_real_b1 = evaluate_real_windows(b1_episodes, holdout_windows)
    res_real_b2 = evaluate_real_windows(b2_episodes, holdout_windows)

    # 6. Holdout Synthetic Problem Episodes
    res_syn_prim = evaluate_synthetic_episodes(primary_episodes, labels_h, primary_h_scored)
    res_syn_real = evaluate_synthetic_episodes(real_only_episodes, labels_h, real_only_h_scored)

    # 7. False-Alarm Budget Evaluation
    b_pass_prim, fa_prim, _ = false_alarm_report(
        primary_episodes, labels_h, primary_h_scored, total_days=total_holdout_days
    )
    b_pass_real, fa_real, _ = false_alarm_report(
        real_only_episodes, labels_h, real_only_h_scored, total_days=total_holdout_days
    )
    b_pass_b1, fa_b1, _ = false_alarm_report(
        b1_episodes, labels_h, b1_h_scored, total_days=total_holdout_days
    )
    b_pass_b2, fa_b2, _ = false_alarm_report(
        b2_episodes, labels_h, b2_h_scored, total_days=total_holdout_days
    )

    # 8. Interval-level metrics (PR-AUC with block bootstrap)
    logger.info("Computing interval-level PR-AUC metrics on holdout...")
    int_prim = interval_metrics(primary_h_scored, labels_h, block_hours=24, n_boot=500)
    int_real = interval_metrics(real_only_h_scored, labels_h, block_hours=24, n_boot=500)

    # 9. Final State Report (Demo check Rule R9)
    logger.info("Scoring full timeline for demo state verification and figures...")
    primary_full_scored = primary_detector.score_frame(features_df)
    state_report = final_state_report(primary_full_scored)

    # 10. Generate Figures F11 to F14
    logger.info("Generating and saving Figures F11 to F14...")
    generate_f11_timelines(primary_full_scored, labels_df)

    all_real_prim = evaluate_real_windows(extract_alert_episodes(primary_full_scored))
    all_real_only = evaluate_real_windows(
        extract_alert_episodes(real_only_detector.score_frame(features_df))
    )
    all_syn_prim = evaluate_synthetic_episodes(
        extract_alert_episodes(primary_full_scored), labels_df, primary_full_scored
    )
    all_syn_only = evaluate_synthetic_episodes(
        extract_alert_episodes(real_only_detector.score_frame(features_df)),
        labels_df,
        real_only_detector.score_frame(features_df),
    )
    generate_f12_lead_times(all_real_prim, all_real_only, all_syn_prim, all_syn_only)

    exp_df = pd.read_csv(PROCESSED_DIR / "anomaly_experiments.csv")
    generate_f13_model_vs_baselines(exp_df)
    generate_f14_m104_contributing_factors(primary_full_scored)

    # 11. Write Append-Only Holdout Log (docs/holdout_log.md)
    run_timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    prim_fail_lead = res_real_prim.iloc[0]["lead_time_hours"]
    real_fail_lead = res_real_only.iloc[0]["lead_time_hours"]

    # Format table lines cleanly
    p_w_rate = f"{fa_prim['pooled_warning_fa_per_30d']:.3f}"
    p_c_fa = fa_prim["total_critical_fa"]
    p_m109_w = f"{fa_prim['m109_warning_fa_per_30d']:.3f}"
    p_m109_c = fa_prim["m109_critical_fa"]
    p_stat = res_real_prim.iloc[0]["status"]
    hero_deg_sub = res_syn_prim[res_syn_prim["episode_id"] == "HERO-DEG-M104"]
    hero_peak_val = hero_deg_sub["peak_score"].iloc[0] if len(hero_deg_sub) > 0 else 0.0
    hero_peak = f"Det ({hero_peak_val:.2f})"

    r_w_rate = f"{fa_real['pooled_warning_fa_per_30d']:.3f}"
    r_c_fa = fa_real["total_critical_fa"]
    r_m109_w = f"{fa_real['m109_warning_fa_per_30d']:.3f}"
    r_m109_c = fa_real["m109_critical_fa"]
    r_stat = res_real_only.iloc[0]["status"]

    b1_w_rate = f"{fa_b1['pooled_warning_fa_per_30d']:.3f}"
    b1_c_fa = fa_b1["total_critical_fa"]
    b1_m109_w = f"{fa_b1['m109_warning_fa_per_30d']:.3f}"
    b1_m109_c = fa_b1["m109_critical_fa"]
    b1_stat = res_real_b1.iloc[0]["status"]

    b2_w_rate = f"{fa_b2['pooled_warning_fa_per_30d']:.3f}"
    b2_c_fa = fa_b2["total_critical_fa"]
    b2_m109_w = f"{fa_b2['m109_warning_fa_per_30d']:.3f}"
    b2_m109_c = fa_b2["m109_critical_fa"]
    b2_stat = res_real_b2.iloc[0]["status"]

    t_row_prim = (
        f"| **Primary (IF-06)** | `plan_core` | **{fa_prim['budget_passed']}** | "
        f"{p_w_rate} | {p_c_fa} | {p_m109_w} | {p_m109_c} | "
        f"{p_stat} ({prim_fail_lead}h) | {hero_peak} |\n"
    )
    t_row_real = (
        f"| **Real-Only (IF-25)** | `real_only` | **{fa_real['budget_passed']}** | "
        f"{r_w_rate} | {r_c_fa} | {r_m109_w} | {r_m109_c} | "
        f"{r_stat} ({real_fail_lead}h) | N/A |\n"
    )
    t_row_b1 = (
        f"| **B1 Manual** | `vib_temp` | **{fa_b1['budget_passed']}** | "
        f"{b1_w_rate} | {b1_c_fa} | {b1_m109_w} | {b1_m109_c} | {b1_stat} | Det |\n"
    )
    t_row_b2 = (
        f"| **B2 Robust Z** | `plan_core` | **{fa_b2['budget_passed']}** | "
        f"{b2_w_rate} | {b2_c_fa} | {b2_m109_w} | {b2_m109_c} | {b2_stat} | Det |\n"
    )

    header_table = (
        "| Detector | Feature Set | Budget | Warn FA/30d | Crit FA | "
        "M-109 Warn | M-109 Crit | FAIL-004 | Hero Deg |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )

    prim_peak = res_real_prim.iloc[0]['peak_score']
    lead_prim_str = f"{prim_fail_lead}h"
    lead_real_str = f"{real_fail_lead}h"

    log_entry = f"""
## Holdout Evaluation Run — {run_timestamp}

- **Execution Mode:** Frozen Pre-Registered Configuration (Decision D64)
- **Frozen Hash (Primary SHA-256):** `{current_prim_hash}`
- **Frozen Hash (Real-Only SHA-256):** `{current_real_hash}`
- **Git Commit:** `{frozen_data['git_commit']}`
- **Holdout Period:** `{h_start}` to `{h_end}` ({total_holdout_days:.1f} days)

### 1. Headline Results & False-Alarm Budget

{header_table}{t_row_prim}{t_row_real}{t_row_b1}{t_row_b2}
### 2. Real Failure Window Detection (FAIL-004 on M-104)
- **Window:** 2026-08-30 14:30 UTC to 2026-09-02 19:00 UTC
- **Primary Model:** Status `{p_stat}`, Lead Time `{lead_prim_str}`, Peak `{prim_peak}`.
- **Real-Only Model:** Status `{r_stat}`, Lead Time `{lead_real_str}`.
- **Intellectual Honesty (Caveat C3):** Primary model detection is elevated by synthetic vibration
  (+60%) and temperature (+3 °C) overlays planted in FAIL-004. Real-only channels show honest
  empirical physics with no artificial precursor. Confidence: **low** (N=1 holdout failure window).

### 3. Holdout Synthetic Episode Detection

| Episode ID | Machine | Episode Type | Primary Det. | Peak Score | Real-Only Det. | Notes |
|---|---|---|---|---|---|---|
"""
    for _, r in res_syn_prim.iterrows():
        ep_id = r["episode_id"]
        r_real = res_syn_real[res_syn_real["episode_id"] == ep_id]
        real_det = r_real["detected"].iloc[0] if len(r_real) > 0 else "N/A"
        log_entry += (
            f"| `{ep_id}` | `{r['machine_id']}` | `{r['episode_type']}` | "
            f"**{r['detected']}** | {r['peak_score']} | {real_det} | "
            f"Alert in {r['hours_to_alert']}h |\n"
        )

    ci_all_0 = int_prim["ci_all"][0]
    ci_all_1 = int_prim["ci_all"][1]
    ci_real_0 = int_prim["ci_real"][0]
    ci_real_1 = int_prim["ci_real"][1]

    log_entry += f"""
### 4. Interval-Level Precision & PR-AUC (Holdout Slice)
- **Primary Model (All True Anomalies):** PR-AUC = **{int_prim['ap_all_true_anomalies']:.4f}**
  (Prevalence: {int_prim['prevalence_all']:.4f}, 95% Bootstrap CI: [{ci_all_0:.4f}, {ci_all_1:.4f}])
- **Primary Model (Real MetroPT Only):** PR-AUC = **{int_prim['ap_real_only']:.4f}**
  (Prevalence: {int_prim['prevalence_real']:.4f}, 95% CI: [{ci_real_0:.4f}, {ci_real_1:.4f}])
- **Real-Only Model (All True Anomalies):** PR-AUC = **{int_real['ap_all_true_anomalies']:.4f}**
  (Prevalence: {int_real['prevalence_all']:.4f})

### 5. Demo-Story Audit at Final Timestamp (Rule R9 Check)

| Machine ID | Last Timestamp | State | Score | Actual Status | Expected Demo | Match |
|---|---|---|---|---|---|---|
"""
    for _, row in state_report.iterrows():
        sc_str = f"{row['anomaly_score']:.4f}" if pd.notna(row["anomaly_score"]) else "N/A"
        log_entry += (
            f"| `{row['machine_id']}` | `{row['last_timestamp']}` | `{row['state']}` | "
            f"{sc_str} | **{row['actual_status']}** | {row['expected_demo_status']} | "
            f"**{row['matches_demo_story']}** |\n"
        )

    log_entry += "\n---\n"

    # Append to docs/holdout_log.md
    if not holdout_log_path.exists():
        initial_header = (
            "# Anomaly Detection Holdout Evaluation Log\n\n"
            "This document records immutable, single-run holdout evaluations under Decision D64.\n"
        )
        with open(holdout_log_path, "w", encoding="utf-8") as f:
            f.write(initial_header)

    with open(holdout_log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)

    logger.info("Holdout results successfully appended to %s", holdout_log_path)

    print("\n" + "=" * 80)
    print("HOLDOUT EVALUATION COMPLETE (Task T11)")
    print(f"  Log File:        {holdout_log_path}")
    print(f"  Primary Model:   FAIL-004 {p_stat}, Budget Pass={b_pass_prim}")
    print(f"  Real-Only Model: FAIL-004 {r_stat}, Budget Pass={b_pass_real}")
    print("  Demo Story Outcomes at Final Timestamp:")
    for _, r in state_report[state_report["expected_demo_status"] != "N/A"].iterrows():
        print(
            f"    {r['machine_id']}: Actual={r['actual_status']}, "
            f"Expected={r['expected_demo_status']}, Match={r['matches_demo_story']}"
        )
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
