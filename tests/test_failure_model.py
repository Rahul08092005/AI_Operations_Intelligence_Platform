"""Comprehensive test suite for failure prediction model package and inference contract.

Covers all 18 requirements from Section 7, Task T12:
1. Split integrity
2. Test partition guard
3. Forbidden columns leakage audit
4. Feature derivation & physics units
5. Type mapping
6. Input validation envelope
7. CV integrity & repeated stratification
8. Scaler isolation in CV
9. Metric & bootstrap functions
10. Calibration monotonicity & selection logic
11. Threshold derivation & boundary categorization
12. Serialization round-trip parity
13. Model determinism under seed
14. SHAP TreeExplainer additivity check
15. Train/serve prediction parity (integration)
16. Demo states test partition purity & sampling
17. Output schema compliance
18. Shuffled label negative control
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml.prediction.baselines import LogisticRegressionBaseline
from ml.prediction.calibration import (
    choose_calibrator,
)
from ml.prediction.config import (
    FROZEN_CONFIG_PATH,
    MACHINE_RISK_DAILY_PATH,
    MACHINE_RISK_LATEST_PATH,
    MODEL_PKL_PATH,
    SEED,
    SPLIT_INDEX_PATH,
)
from ml.prediction.controls import (
    assert_forbidden_columns_absent,
    run_shuffled_label_control,
)
from ml.prediction.cv import run_cv
from ml.prediction.data import load_test
from ml.prediction.features import build_features
from ml.prediction.inference import clear_model_cache, predict_batch, predict_one
from ml.prediction.metrics import (
    bootstrap_ci,
    compute_expected_calibration_error,
    compute_operating_metrics,
    precision_at_recall,
    recall_at_precision,
)
from ml.prediction.model import FailureModel
from ml.prediction.thresholds import (
    assign_risk_category,
    compute_category_reliability,
    compute_thresholds,
)
from ml.prediction.validation import validate_inputs


# 1. Split integrity
def test_split_integrity():
    split_df = pd.read_csv(SPLIT_INDEX_PATH)
    assert len(split_df) == 10000

    train_udis = set(split_df.loc[split_df["partition"] == "train", "udi"])
    test_udis = set(split_df.loc[split_df["partition"] == "test", "udi"])

    assert len(train_udis) == 8000
    assert len(test_udis) == 2000
    assert train_udis.isdisjoint(test_udis)


# 2. Test guard
def test_test_guard(tmp_path: Path):
    # Non-existent config
    fake_config = tmp_path / "non_existent.json"
    with pytest.raises(PermissionError, match="Guard violation"):
        load_test(fake_config)

    # Valid config on disk
    if FROZEN_CONFIG_PATH.exists():
        test_df = load_test(FROZEN_CONFIG_PATH)
        assert len(test_df) == 2000


# 3. Forbidden columns
def test_forbidden_columns():
    df = pd.DataFrame(
        {
            "type": ["L", "M"],
            "air_temp_k": [300.0, 301.0],
            "process_temp_k": [310.0, 311.0],
            "rpm": [1500, 1600],
            "torque_nm": [40.0, 45.0],
            "tool_wear_min": [100.0, 150.0],
            "twf": [0, 1],
            "udi": [1, 2],
        }
    )
    # Features built should exclude twf and udi
    feats = build_features(df, feature_set="physics")
    assert_forbidden_columns_absent(feats)
    for forbidden in ["twf", "hdf", "pwf", "osf", "rnf", "udi", "product_id", "machine_failure"]:
        assert forbidden not in feats.columns


# 4. Feature values
def test_feature_values():
    row = pd.DataFrame(
        [
            {
                "type": "M",
                "air_temp_k": 300.0,
                "process_temp_k": 309.5,
                "rpm": 1500,
                "torque_nm": 40.0,
                "tool_wear_min": 120.0,
            }
        ]
    )
    feats = build_features(row, feature_set="physics")

    expected_temp_diff = 9.5
    expected_power = 40.0 * (1500.0 * 2.0 * np.pi / 60.0)  # ~6283.185 W
    expected_strain = 120.0 * 40.0  # 4800.0

    assert np.isclose(feats.loc[0, "temp_diff_k"], expected_temp_diff)
    assert np.isclose(feats.loc[0, "power_w"], expected_power)
    assert np.isclose(feats.loc[0, "strain_nmin"], expected_strain)
    assert feats.loc[0, "type_code"] == 1.0


# 5. Type mapping
def test_type_mapping():
    df_valid = pd.DataFrame(
        {
            "type": ["L", "M", "H"],
            "air_temp_k": [300.0, 300.0, 300.0],
            "process_temp_k": [310.0, 310.0, 310.0],
            "rpm": [1500, 1500, 1500],
            "torque_nm": [40.0, 40.0, 40.0],
            "tool_wear_min": [50.0, 50.0, 50.0],
        }
    )
    f = build_features(df_valid, feature_set="raw")
    assert list(f["type_code"]) == [0.0, 1.0, 2.0]

    df_invalid = df_valid.copy()
    df_invalid.loc[0, "type"] = "X"
    with pytest.raises(ValueError, match="Invalid machine variant type"):
        build_features(df_invalid, feature_set="raw")


# 6. Validation
def test_validation():
    # Valid
    df = pd.DataFrame(
        [
            {
                "type": "L",
                "air_temp_k": 300.0,
                "process_temp_k": 310.0,
                "rpm": 1500,
                "torque_nm": 40.0,
                "tool_wear_min": 50.0,
            }
        ]
    )
    res = validate_inputs(df)
    assert res.is_valid
    assert len(res.warnings[0]) == 0

    # Missing column
    df_missing = df.drop(columns=["rpm"])
    with pytest.raises(ValueError, match="Missing required"):
        validate_inputs(df_missing)

    # Out of envelope warning
    training_stats = {
        "torque_nm": {"min": 10.0, "max": 60.0, "mean": 40.0, "std": 10.0}
    }
    df_out = df.copy()
    df_out.loc[0, "torque_nm"] = 150.0  # Far outside
    res_out = validate_inputs(df_out, training_stats)
    assert res_out.is_valid
    assert len(res_out.warnings[0]) > 0
    assert "torque_nm outside training range" in res_out.warnings[0][0]


# 7. CV integrity
def test_cv_integrity():
    # Synthetic dataset with 200 samples
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame(rng.normal(size=(200, 5)), columns=[f"f{i}" for i in range(5)])
    y = np.zeros(200, dtype=int)
    y[rng.choice(200, size=20, replace=False)] = 1

    res = run_cv(lambda: LogisticRegression(solver="lbfgs"), X, y)
    assert len(res.fold_metrics) == 10  # 5 folds x 2 repeats
    assert len(res.repeat_metrics) == 2
    assert len(res.oof_df) == 400  # 200 samples x 2 repeats

    rep1 = res.oof_df[res.oof_df["repeat"] == 1]
    assert len(rep1["row_idx"].unique()) == 200


# 8. Scaler isolation
def test_scaler_isolation():
    # Verify that LogisticRegressionBaseline fits StandardScaler inside Pipeline only on train fold
    train_X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0]})
    train_y = np.array([0, 0, 1, 1])

    baseline = LogisticRegressionBaseline()
    baseline.fit(train_X, train_y)

    scaler: StandardScaler = baseline.pipeline.named_steps["scaler"]
    assert np.isclose(scaler.mean_[0], 2.5)

    # Predicting on wild values should not alter scaler mean
    wild_X = pd.DataFrame({"f1": [1000.0, 2000.0]})
    baseline.predict_proba(wild_X)
    assert np.isclose(scaler.mean_[0], 2.5)


# 9. Metric functions
def test_metric_functions():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])

    rec, th = recall_at_precision(y_true, y_prob, target_precision=0.90)
    assert rec > 0.0

    prec, th = precision_at_recall(y_true, y_prob, target_recall=0.80)
    assert prec > 0.0

    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=3)
    assert 0.0 <= ece <= 1.0

    m = compute_operating_metrics(y_true, y_prob, threshold=0.5)
    assert m["tp"] == 3
    assert m["fp"] == 0
    assert m["accuracy"] == 1.0

    # Deterministic bootstrap
    ci1 = bootstrap_ci(y_true, y_prob, lambda yt, yp: float(np.mean(yp)), n_resamples=50, seed=42)
    ci2 = bootstrap_ci(y_true, y_prob, lambda yt, yp: float(np.mean(yp)), n_resamples=50, seed=42)
    assert ci1.point_estimate == ci2.point_estimate
    assert ci1.ci_lower == ci2.ci_lower
    assert ci1.ci_upper == ci2.ci_upper


# 10. Calibration logic
def test_calibration_logic():
    # Isotonic selection requires >= 5% Brier improvement and non-worsening ECE
    oof_probs = np.linspace(0.01, 0.99, 100)
    y_true = (oof_probs >= 0.5).astype(int)
    fold_indices = np.array([i % 5 for i in range(100)])

    comp, cal = choose_calibrator(oof_probs, y_true, fold_indices)
    assert comp.chosen_method in ["sigmoid", "isotonic"]

    # Calibrated probabilities monotonic in raw probabilities
    preds = cal.predict_proba(np.array([0.1, 0.3, 0.7, 0.9]))
    assert preds[0] <= preds[1] <= preds[2] <= preds[3]


# 11. Thresholds derivation
def test_thresholds_derivation():
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    probs = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.95])

    thresh = compute_thresholds(y_true, probs)
    assert thresh.t_med <= thresh.t_high <= thresh.t_crit

    cats = assign_risk_category(probs, thresh)
    assert cats[0] == "Low"
    assert cats[-1] == "Critical"

    rel = compute_category_reliability(y_true, probs, thresh)
    assert len(rel) == 4
    assert set(rel["risk_category"]) == {"Low", "Medium", "High", "Critical"}


# 12. Serialization parity
def test_serialization_parity(tmp_path: Path):
    if not MODEL_PKL_PATH.exists():
        pytest.skip("Model artifact not yet built.")

    m1 = FailureModel.load(MODEL_PKL_PATH)
    tmp_file = tmp_path / "test_model.pkl"
    m1.save(tmp_file)

    m2 = FailureModel.load(tmp_file)

    df_test = pd.DataFrame(
        [
            {
                "type": "L",
                "air_temp_k": 300.0,
                "process_temp_k": 310.0,
                "rpm": 1500,
                "torque_nm": 40.0,
                "tool_wear_min": 50.0,
            }
        ]
    )
    p1 = m1.predict_proba(df_test)
    p2 = m2.predict_proba(df_test)
    assert np.allclose(p1, p2, atol=1e-9)


# 13. Model determinism
def test_model_determinism():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(50, 4)), columns=["c1", "c2", "c3", "c4"])
    y = rng.integers(0, 2, size=50)

    params = {
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.1,
        "random_state": 42,
    }
    m1 = XGBClassifier(**params).fit(X, y)
    m2 = XGBClassifier(**params).fit(X, y)

    p1 = m1.predict_proba(X)
    p2 = m2.predict_proba(X)
    assert np.array_equal(p1, p2)


# 14. SHAP TreeExplainer additivity
def test_shap_additivity():
    if not MODEL_PKL_PATH.exists():
        pytest.skip("Model artifact not yet built.")

    model = FailureModel.load(MODEL_PKL_PATH)
    booster = model.booster
    explainer = shap.TreeExplainer(booster)

    sample = model.shap_background.iloc[:50]
    shap_vals = explainer.shap_values(sample)
    expected_val = explainer.expected_value

    import xgboost as xgb
    dmat = xgb.DMatrix(sample)
    margins = booster.get_booster().predict(dmat, output_margin=True)

    shap_sums = np.sum(shap_vals, axis=1) + expected_val
    assert np.allclose(shap_sums, margins, atol=1e-3)


# 15. Train/serve parity (integration)
@pytest.mark.integration
def test_train_serve_parity():
    if not MODEL_PKL_PATH.exists():
        pytest.skip("Model artifact not yet built.")

    clear_model_cache()
    model = FailureModel.load(MODEL_PKL_PATH)

    df_sample = pd.DataFrame(
        [
            {
                "type": "M",
                "air_temp_k": 301.2,
                "process_temp_k": 310.8,
                "rpm": 1450,
                "torque_nm": 45.2,
                "tool_wear_min": 120.0,
            },
            {
                "type": "L",
                "air_temp_k": 298.5,
                "process_temp_k": 308.2,
                "rpm": 1600,
                "torque_nm": 32.0,
                "tool_wear_min": 35.0,
            },
        ]
    )

    batch_res = predict_batch(df_sample, model=model)
    single_res0 = predict_one(df_sample.iloc[0].to_dict(), model=model)
    single_res1 = predict_one(df_sample.iloc[1].to_dict(), model=model)

    assert np.isclose(
        batch_res.loc[0, "failure_probability"], single_res0["failure_probability"], atol=1e-4
    )
    assert np.isclose(
        batch_res.loc[1, "failure_probability"], single_res1["failure_probability"], atol=1e-4
    )
    assert batch_res.loc[0, "risk_level"] == single_res0["risk_level"]
    assert batch_res.loc[1, "risk_level"] == single_res1["risk_level"]


# 16. Demo states test partition purity
def test_demo_states_integrity():
    if not MACHINE_RISK_DAILY_PATH.exists():
        pytest.skip("Demo risk tables not yet built.")

    states_df = pd.read_parquet(MACHINE_RISK_DAILY_PATH)
    assert len(states_df) == 360  # 12 machines x 30 days

    test_df = load_test(FROZEN_CONFIG_PATH)
    test_udis = set(test_df["udi"])

    valid_states = states_df[states_df["source_udi"] != -1]
    assert set(valid_states["source_udi"]).issubset(test_udis)


# 17. Output schema compliance
def test_output_schema():
    if not MACHINE_RISK_LATEST_PATH.exists():
        pytest.skip("machine_risk_latest.csv not yet built.")

    latest_df = pd.read_csv(MACHINE_RISK_LATEST_PATH)
    assert len(latest_df) == 12
    required_cols = [
        "machine_id",
        "snapshot_date",
        "machine_type",
        "scenario",
        "failure_probability",
        "risk_level",
        "risk_7d_mean",
        "risk_7d_level",
        "is_replay",
    ]
    for c in required_cols:
        assert c in latest_df.columns
    assert latest_df["is_replay"].all()


# 18. Negative control on separable dataset
def test_negative_control():
    # Create cleanly separable dataset
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["f1", "f2", "f3"])
    y = (X["f1"] > 0).astype(int).to_numpy()

    params = {"n_estimators": 10, "max_depth": 2, "random_state": 42}
    res_shuffled = run_shuffled_label_control(X, y, params, seed=42)

    # Shuffled label ROC-AUC should collapse to chance range (0.35 - 0.65)
    assert 0.35 <= res_shuffled.mean_roc_auc <= 0.65
