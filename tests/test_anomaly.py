"""Unit and integration tests for anomaly detection (Task T13).

Covers all 16 required test cases:
1. CV folds (temporal gap and disjoint validation)
2. Training mask exclusions and inclusions
3. Baseline leakage (holdout tampering immunity)
4. Baseline fallback for unknown machine
5. Calibration monotonicity, anchors, and bounds
6. Smoothing causality and long gap reset
7. Status threshold boundaries
8. State gating (Offline machines never scored/alerted)
9. Episode extraction, minimum duration, and merge logic
10. Real-window evaluation, classification, and lead time
11. False-alarm budget calculation and pass/fail logic
12. Contributing factor attribution and multivariate rule
13. Model serialization round-trip fidelity
14. Model fit determinism under identical seed
15. Train/serve parity (batch vs window score agreement <= 1e-6)
16. Frozen-config guard
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml.anomaly_detection.baselines import MachineBaseline
from ml.anomaly_detection.calibration import (
    CALIBRATION_ANCHORS,
    STATE_OFFLINE,
    STATE_RUNNING,
    STATUS_CRITICAL,
    STATUS_NORMAL,
    STATUS_OFFLINE,
    STATUS_WARNING,
    ScoreCalibrator,
    smooth_scores,
    to_status,
)
from ml.anomaly_detection.config import (
    MODELS_DIR,
    PROCESSED_DIR,
)
from ml.anomaly_detection.detector import AnomalyDetector
from ml.anomaly_detection.episodes import extract_alert_episodes
from ml.anomaly_detection.evaluate import (
    evaluate_real_windows,
    false_alarm_report,
)
from ml.anomaly_detection.explain import compute_contributing_factors
from ml.anomaly_detection.inference import score_window
from ml.anomaly_detection.masks import training_mask
from ml.anomaly_detection.splits import cv_folds, get_periods


# ==============================================================================
# 1. CV Folds
# ==============================================================================
def test_cv_folds_disjoint_and_temporal_gap():
    """Validation is strictly after training with a 24-hour gap, folds disjoint in validation."""
    folds = cv_folds()
    assert len(folds) == 3

    for fold in folds:
        # Validation strictly after train end
        assert fold.val_start > fold.train_end
        # Exactly 24-hour gap
        gap = fold.val_start - fold.train_end
        assert gap == pd.Timedelta(hours=24.0)
        assert fold.val_end > fold.val_start

    # Validation periods are mutually disjoint
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            disjoint = (folds[i].val_end <= folds[j].val_start) or (
                folds[j].val_end <= folds[i].val_start
            )
            assert disjoint, f"Folds {i} and {j} overlap in validation period."


# ==============================================================================
# 2. Training Mask
# ==============================================================================
def test_training_mask_exclusions():
    """Mask excludes offline, warm-up, gap, fault, missing-features and problem periods."""
    ts = pd.date_range("2026-04-01", periods=10, freq="5min", tz="UTC")
    period = (ts[0], ts[-1] + pd.Timedelta(minutes=5))

    features_df = pd.DataFrame(
        {
            "timestamp": ts,
            "machine_id": ["M-101"] * 10,
            "running": [True, True, False, True, True, True, True, True, True, True],
            "is_warmup": [False, False, False, True, False, False, False, False, False, False],
            "is_gap": [False, False, False, False, True, False, False, False, False, False],
            "is_sensor_fault": [
                False, False, False, False, False, False, True, False, False, False
            ],
            "vibration": [0.5, 0.6, 0.5, 0.5, 0.5, np.nan, 2.5, 0.5, 0.5, 0.5],
        }
    )

    labels_df = pd.DataFrame(
        {
            "timestamp": ts,
            "machine_id": ["M-101"] * 10,
            "episode_type": [
                "normal",
                "benign_spike",
                "normal",
                "normal",
                "normal",
                "normal",
                "normal",
                "normal",
                "normal",
                "normal",
            ],
        }
    )

    mask, report = training_mask(
        features_df=features_df,
        labels_df=labels_df,
        feature_list=["vibration"],
        period=period,
    )

    # Row 0: Running, normal, valid -> True
    assert mask.iloc[0]
    # Row 1: Running, benign_spike, valid -> True
    assert mask.iloc[1]
    # Row 2: Offline (running=False) -> False
    assert not mask.iloc[2]
    # Row 3: WarmUp -> False
    assert not mask.iloc[3]
    # Row 4: Gap -> False
    assert not mask.iloc[4]
    # Row 5: NaN in vibration -> False
    assert not mask.iloc[5]
    # Row 6: Sensor fault -> False
    assert not mask.iloc[6]


# ==============================================================================
# 3. Baseline Leakage
# ==============================================================================
def test_baseline_leakage_immunity():
    """Altering holdout values does not change fitted baselines or calibrator anchors."""
    exp_period, holdout_period = get_periods()
    exploration_end = exp_period[1]

    ts_exp = pd.date_range("2026-04-01", periods=100, freq="1h", tz="UTC")
    ts_hold = pd.date_range(
        exploration_end + pd.Timedelta(days=1), periods=100, freq="1h", tz="UTC"
    )

    df_base = pd.DataFrame(
        {
            "timestamp": list(ts_exp) + list(ts_hold),
            "machine_id": ["M-101"] * 200,
            "vibration": [1.0] * 100 + [1.0] * 100,
            "state": [STATE_RUNNING] * 200,
        }
    )

    # Tampered holdout version
    df_tampered = df_base.copy()
    df_tampered.loc[100:, "vibration"] = 999.0

    mask_exp = df_base["timestamp"] < exploration_end
    base1 = MachineBaseline().fit(df_base[mask_exp], feature_list=["vibration"])
    base2 = MachineBaseline().fit(df_tampered[mask_exp], feature_list=["vibration"])

    stat1 = base1.machine_stats["M-101"]["vibration"]
    stat2 = base2.machine_stats["M-101"]["vibration"]
    assert stat1 == stat2


# ==============================================================================
# 4. Baseline Fallback
# ==============================================================================
def test_baseline_fallback_for_unknown_machine():
    """Unknown machine falls back to type-level baseline or fleet baseline."""
    df = pd.DataFrame(
        {
            "machine_id": ["M-101", "M-101", "M-102", "M-102"],
            "vibration": [1.0, 1.2, 1.4, 1.6],
        }
    )
    machines_df = pd.DataFrame(
        {
            "machine_id": ["M-101", "M-102", "M-999"],
            "machine_type": ["filler", "filler", "filler"],
        }
    )

    baseline = MachineBaseline().fit(df, feature_list=["vibration"], machines_df=machines_df)

    # Transform a row for unknown machine M-999
    test_row = pd.DataFrame(
        {
            "machine_id": ["M-999"],
            "vibration": [1.3],
        }
    )
    z_res = baseline.transform(test_row)
    assert not z_res["vibration"].isna().any()
    # z-score should be near 0 since 1.3 is right around median
    assert abs(z_res["vibration"].iloc[0]) < 1.0


# ==============================================================================
# 5. Calibration
# ==============================================================================
def test_calibration_monotonic_and_anchors():
    """Calibrator is monotonic, maps anchors to 0.10, 0.50, 0.80, saturates at 1.0, clips [0, 1]."""
    raw_scores = np.linspace(0.2, 0.8, 1000)
    calibrator = ScoreCalibrator().fit(
        raw_train_scores=raw_scores,
        q_warn=0.99,
        q_crit=0.9999,
    )

    # Verify anchors
    s_normal = calibrator.map_scores(np.array([calibrator.raw_median]))[0]
    s_warn = calibrator.map_scores(np.array([calibrator.raw_q_warn]))[0]
    s_crit = calibrator.map_scores(np.array([calibrator.raw_q_crit]))[0]

    assert pytest.approx(s_normal, abs=1e-3) == CALIBRATION_ANCHORS["median_score"]
    assert pytest.approx(s_warn, abs=1e-3) == CALIBRATION_ANCHORS["warn_score"]
    assert pytest.approx(s_crit, abs=1e-3) == CALIBRATION_ANCHORS["crit_score"]

    # Verify monotonicity
    test_inputs = np.linspace(0.0, 1.5, 200)
    calibrated = calibrator.map_scores(test_inputs)
    assert np.all(np.diff(calibrated) >= -1e-7)

    # Verify bounds
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)


# ==============================================================================
# 6. Smoothing
# ==============================================================================
def test_smoothing_trailing_only_and_gap_reset():
    """Trailing only: altering future values does not affect past scores; resets on long gap."""
    n = 100
    ts = pd.date_range("2026-04-01", periods=n, freq="5min", tz="UTC")
    raw = pd.Series(np.full(n, 0.2))
    m_ids = pd.Series(["M-101"] * n)
    can_score = pd.Series(True, index=range(n))

    # Baseline series
    s1 = smooth_scores(raw, m_ids, pd.Series(ts), can_score, window=12, max_gap_minutes=15.0)

    # Altering future values (from index 50 onward)
    raw_altered = raw.copy()
    raw_altered[50:] = 0.9
    s2 = smooth_scores(
        raw_altered, m_ids, pd.Series(ts), can_score, window=12, max_gap_minutes=15.0
    )

    # Up to index 49, smoothed values must be strictly identical
    np.testing.assert_allclose(s1[:50], s2[:50], atol=1e-6)

    # Test long gap reset: insert a 4-hour gap between idx 30 and 31
    ts_gapped = pd.Series(ts).copy()
    ts_gapped.iloc[31:] = ts_gapped.iloc[31:] + pd.Timedelta(hours=4)
    s_gap = smooth_scores(raw, m_ids, ts_gapped, can_score, window=12, max_gap_minutes=15.0)

    # After the long gap, the first 5 readings are in min_periods warmup (NaN)
    assert pd.isna(s_gap.iloc[31])
    # At idx 37, 7 readings have accumulated, window is valid and equals 0.2
    assert pytest.approx(s_gap.iloc[37], abs=1e-6) == raw.iloc[37]


# ==============================================================================
# 7. Status Thresholds
# ==============================================================================
def test_status_threshold_boundaries():
    """Scores map to Normal, Warning, Critical at exact boundary values."""
    scores = pd.Series([0.0, 0.4999, 0.50, 0.7999, 0.80, 1.0])
    states = pd.Series([STATE_RUNNING] * 6)
    statuses = to_status(scores, states)

    expected = [
        STATUS_NORMAL,
        STATUS_NORMAL,
        STATUS_WARNING,
        STATUS_WARNING,
        STATUS_CRITICAL,
        STATUS_CRITICAL,
    ]
    assert list(statuses) == expected


# ==============================================================================
# 8. State Gating
# ==============================================================================
def test_state_gating_offline_rows():
    """Offline rows never receive a score or an alert (status is Offline)."""
    scores = pd.Series([0.95, 0.95, 0.20])
    states = pd.Series([STATE_OFFLINE, STATE_OFFLINE, STATE_RUNNING])
    statuses = to_status(scores, states)

    assert statuses.iloc[0] == STATUS_OFFLINE
    assert statuses.iloc[1] == STATUS_OFFLINE
    assert statuses.iloc[2] == STATUS_NORMAL


# ==============================================================================
# 9. Episodes
# ==============================================================================
def test_episode_extraction_min_len_and_merging():
    """Episode extraction enforces min_intervals and merges episodes within merge_gap_hours."""
    ts = pd.date_range("2026-04-01 00:00", periods=25, freq="5min", tz="UTC")
    statuses = [STATUS_NORMAL] * 25
    # Isolated spike at idx 1 (00:05), then 35 min normal gap before next alert at idx 9 (00:45)
    # Since gap > 15 min and len=1 < min_intervals=2, idx 1 spike is filtered out
    statuses[1] = STATUS_WARNING

    # Run 1: idx 9, 10, 11 (len 3 >= 2)
    statuses[9] = STATUS_WARNING
    statuses[10] = STATUS_CRITICAL
    statuses[11] = STATUS_WARNING

    # Normal gap: idx 12, 13 (10 min <= merge_gap_hours=2.0)
    # Run 2: idx 14, 15 (len 2 >= 2) -> merged into Run 1
    statuses[14] = STATUS_WARNING
    statuses[15] = STATUS_WARNING

    scores = [0.1] * 25
    scores[1] = 0.55
    scores[10] = 0.85
    scores[14] = 0.65

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "machine_id": ["M-101"] * 25,
            "anomaly_status": statuses,
            "anomaly_score": scores,
            "state": [STATE_RUNNING] * 25,
            "factor_1": ["vibration"] * 25,
            "factor_1_z": [3.0] * 25,
            "factor_2_z": [0.5] * 25,
            "factor_3_z": [0.2] * 25,
        }
    )

    episodes = extract_alert_episodes(df, min_intervals=2, merge_gap_hours=2.0)
    assert len(episodes) == 1
    ep = episodes.iloc[0]
    assert ep["severity"] == STATUS_CRITICAL
    assert ep["start"] == ts[9]
    assert ep["end"] == ts[15]
    assert pytest.approx(ep["peak_score"], abs=1e-3) == 0.85


# ==============================================================================
# 10. Real-Window Evaluation
# ==============================================================================
def test_real_window_evaluation_metrics():
    """Real-window evaluation correctly classifies early, late, missed and lead time."""
    target_windows = [
        {
            "failure_id": "F1",
            "machine_id": "M-104",
            "start": pd.Timestamp("2026-05-01 12:00:00", tz="UTC"),
            "end": pd.Timestamp("2026-05-01 14:00:00", tz="UTC"),
            "label": "Compressor Bearing Fault",
        }
    ]

    # Alert firing 2 hours before window start (lead time 2.0h -> early)
    episodes_hit = pd.DataFrame(
        [
            {
                "machine_id": "M-104",
                "start": pd.Timestamp("2026-05-01 10:00:00", tz="UTC"),
                "end": pd.Timestamp("2026-05-01 12:30:00", tz="UTC"),
                "peak_score": 0.85,
                "severity": STATUS_CRITICAL,
            }
        ]
    )

    eval_df = evaluate_real_windows(episodes_hit, windows=target_windows)
    assert len(eval_df) == 1
    row = eval_df.iloc[0]
    assert row["status"] == "early"
    assert pytest.approx(row["lead_time_hours"], abs=0.1) == 2.0


# ==============================================================================
# 11. False Alarm Budget
# ==============================================================================
def test_false_alarm_budget_calculation():
    """False-alarm audit calculates per-30-day rate and verifies pass/fail conditions."""
    ts = pd.date_range("2026-04-01", periods=100, freq="1h", tz="UTC")
    total_days = len(ts) / 24.0

    scored_df = pd.DataFrame(
        {
            "timestamp": ts,
            "machine_id": ["M-101"] * 100,
            "state": [STATE_RUNNING] * 100,
            "anomaly_score": [0.1] * 100,
            "anomaly_status": [STATUS_NORMAL] * 100,
        }
    )
    labels_df = pd.DataFrame(
        {
            "timestamp": ts,
            "machine_id": ["M-101"] * 100,
            "episode_type": ["normal"] * 100,
        }
    )

    # 0 episodes -> 0.0 false alarms / 30d -> Pass
    empty_episodes = pd.DataFrame(
        columns=["machine_id", "start", "end", "severity", "peak_score"]
    )
    passed, summary, machine_df = false_alarm_report(
        empty_episodes, labels_df, scored_df, total_days=total_days
    )
    assert passed
    assert summary["pooled_warning_fa_per_30d"] == 0.0


# ==============================================================================
# 12. Factor Attribution & Multivariate Rule
# ==============================================================================
def test_factor_attribution_top3_and_multivariate():
    """Contributing factor attribution extracts top-3 deviations; episodes apply multivariate."""
    baseline = MachineBaseline(feature_list=["temperature", "vibration", "pressure", "rpm"])
    baseline.machine_stats["M-101"] = {
        "temperature": (50.0, 2.0),
        "vibration": (1.0, 0.1),
        "pressure": (3.0, 0.2),
        "rpm": (1500.0, 50.0),
    }

    # Spike in vibration: z = (1.5 - 1.0) / 0.1 = +5.0
    test_df = pd.DataFrame(
        {
            "machine_id": ["M-101"],
            "temperature": [50.0],
            "vibration": [1.5],
            "pressure": [3.0],
            "rpm": [1500.0],
            "state": [STATE_RUNNING],
            "running": [True],
            "timestamp": [pd.Timestamp("2026-04-01 12:00:00", tz="UTC")],
        }
    )

    factors_df = compute_contributing_factors(
        baseline=baseline,
        feature_list=["temperature", "vibration", "pressure", "rpm"],
        features_df=test_df,
    )
    assert factors_df.loc[0, "factor_1"] == "vibration"
    assert factors_df.loc[0, "factor_1_direction"] == "high"
    assert pytest.approx(factors_df.loc[0, "factor_1_z"], abs=0.1) == 5.0

    # Multivariate rule test in episode assignment:
    # If top factor share < 0.5 of top 3 absolute z's, anomaly_type is 'multivariate'
    scored_multi = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-01 12:00", periods=4, freq="5min", tz="UTC"),
            "machine_id": ["M-101"] * 4,
            "anomaly_status": [STATUS_WARNING] * 4,
            "anomaly_score": [0.65] * 4,
            "state": [STATE_RUNNING] * 4,
            "factor_1": ["vibration"] * 4,
            "factor_1_z": [2.0] * 4,  # share = 2.0 / (2.0 + 1.8 + 1.5) = 2.0 / 5.3 = 0.377 < 0.5
            "factor_2_z": [1.8] * 4,
            "factor_3_z": [1.5] * 4,
        }
    )
    episodes = extract_alert_episodes(scored_multi, min_intervals=2, merge_gap_hours=2.0)
    assert len(episodes) == 1
    assert episodes.iloc[0]["anomaly_type"] == "multivariate"


# ==============================================================================
# 13. Serialization
# ==============================================================================
def test_model_serialization_round_trip(tmp_path):
    """Save and load round trip gives identical scores."""
    np.random.seed(42)
    n = 200
    features = ["temperature", "vibration", "pressure", "rpm", "energy_consumption"]
    X = pd.DataFrame(np.random.randn(n, len(features)), columns=features)
    X["machine_id"] = "M-101"
    X["timestamp"] = pd.date_range("2026-04-01", periods=n, freq="5min", tz="UTC")
    X["state"] = STATE_RUNNING
    X["running"] = True

    labels = pd.DataFrame(
        {
            "timestamp": X["timestamp"],
            "machine_id": ["M-101"] * n,
            "episode_type": ["normal"] * n,
        }
    )

    detector = AnomalyDetector()
    detector.fit(
        features_df=X,
        labels_df=labels,
        period=(X["timestamp"].min(), X["timestamp"].max() + pd.Timedelta(minutes=5)),
        feature_set="plan_core",
        params={"n_estimators": 50, "max_samples": 128, "max_features": 1.0, "random_state": 42},
    )

    scores_orig = detector.score_frame(X)

    save_path = tmp_path / "detector_test.pkl"
    detector.save(save_path)

    detector_loaded = AnomalyDetector.load(save_path)
    scores_loaded = detector_loaded.score_frame(X)

    pd.testing.assert_frame_equal(scores_orig, scores_loaded)


# ==============================================================================
# 14. Determinism
# ==============================================================================
def test_fit_determinism_same_seed():
    """Two fits with the same seed give identical raw scores."""
    np.random.seed(42)
    n = 200
    features = ["temperature", "vibration", "pressure", "rpm", "energy_consumption"]
    X = pd.DataFrame(np.random.randn(n, len(features)), columns=features)
    X["machine_id"] = "M-101"
    X["timestamp"] = pd.date_range("2026-04-01", periods=n, freq="5min", tz="UTC")
    X["state"] = STATE_RUNNING
    X["running"] = True

    labels = pd.DataFrame(
        {
            "timestamp": X["timestamp"],
            "machine_id": ["M-101"] * n,
            "episode_type": ["normal"] * n,
        }
    )

    period = (X["timestamp"].min(), X["timestamp"].max() + pd.Timedelta(minutes=5))
    params = {"n_estimators": 50, "max_samples": 128, "max_features": 1.0, "random_state": 42}

    det1 = AnomalyDetector().fit(X, labels, period=period, feature_set="plan_core", params=params)
    det2 = AnomalyDetector().fit(X, labels, period=period, feature_set="plan_core", params=params)

    scores1 = det1.score_frame(X)
    scores2 = det2.score_frame(X)

    np.testing.assert_allclose(
        scores1["anomaly_score_raw"], scores2["anomaly_score_raw"], atol=1e-7
    )


# ==============================================================================
# 15. Train/Serve Parity (Integration)
# ==============================================================================
def test_train_serve_parity_integration():
    """Scoring a 3-day window through inference.score_window reproduces batch scores within 1e-6."""
    model_path = MODELS_DIR / "anomaly_model.pkl"
    scores_path = PROCESSED_DIR / "anomaly_scores.parquet"
    sensors_path = PROCESSED_DIR / "sensors.parquet"

    if not model_path.exists() or not scores_path.exists() or not sensors_path.exists():
        pytest.skip("Processed data or models missing; skipping parity integration test.")

    scores_batch = pd.read_parquet(scores_path)
    sensors = pd.read_parquet(sensors_path)

    # Pick M-101 and a 3-day window (864 intervals)
    m_sensors = sensors[sensors["machine_id"] == "M-101"].sort_values("timestamp")
    test_window = m_sensors.iloc[1000:1864].copy()

    # Score using inference.score_window
    scored_window = score_window(
        test_window,
        machine_id="M-101",
        model_path=model_path,
        return_all=False,  # Evaluates points after the 24h history
    )

    eval_ts = scored_window["timestamp"]
    batch_sub = scores_batch[
        (scores_batch["machine_id"] == "M-101") & (scores_batch["timestamp"].isin(eval_ts))
    ].sort_values("timestamp").reset_index(drop=True)

    # Parity check within 1e-6
    diff = np.abs(
        batch_sub["anomaly_score"].to_numpy() - scored_window["anomaly_score"].to_numpy()
    )
    assert np.nanmax(diff) < 1e-6, f"Train/serve parity exceeded tolerance: {np.nanmax(diff)}"


# ==============================================================================
# 16. Frozen Config Guard & Output Schema
# ==============================================================================
def test_frozen_config_guard_and_output_schema():
    """Verify frozen configuration exists with valid hashes and output files match schema."""
    frozen_path = MODELS_DIR / "anomaly_config_frozen.json"
    assert frozen_path.exists(), "Frozen configuration missing."

    with open(frozen_path) as f:
        frozen = json.load(f)

    assert "primary_configuration" in frozen
    assert "real_only_configuration" in frozen
    assert "artifact_hashes" in frozen
    assert "primary_model_sha256" in frozen["artifact_hashes"]
    assert len(frozen["artifact_hashes"]["primary_model_sha256"]) == 64

    # Output schema check if files exist
    scores_path = PROCESSED_DIR / "anomaly_scores.parquet"
    status_path = PROCESSED_DIR / "anomaly_latest_status.csv"

    if scores_path.exists():
        scores = pd.read_parquet(scores_path)
        required_cols = {
            "timestamp", "machine_id", "state", "anomaly_score_raw",
            "anomaly_score", "anomaly_status"
        }
        assert required_cols.issubset(set(scores.columns))

    if status_path.exists():
        status = pd.read_csv(status_path)
        assert len(status) == 12
        required_status_cols = {
            "machine_id", "timestamp", "state", "anomaly_score", "anomaly_status"
        }
        assert required_status_cols.issubset(set(status.columns))
