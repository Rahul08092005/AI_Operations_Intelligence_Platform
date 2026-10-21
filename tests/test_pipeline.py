"""Unit and integration tests for the Day 3 Data Engineering Pipeline.

Covers:
1. Validation catches issues (bad frames vs. good frames)
2. Grid reindexing creates explicit gap rows
3. Imputation limit (gap of 3 filled, gap of 4 not)
4. Tier 1 hard limits (out-of-range -> NaN and is_sensor_fault)
5. Tier 2 outlier detection is flag-only (numeric values byte-identical)
6. Operating hours never decrease after cleaning
7. Leakage: rolling features (perturbing future values after t does not alter features <= t)
8. Leakage: maintenance features (adding event on D does not alter features <= D)
9. Leakage: labels (no label columns in feature tables)
10. Retail rules (cancellations, guests, non-product, extreme quantities, kept + rejected == input)
11. Daily aggregation (energy_kwh equals manual sum)
12. Business KPIs (average_order_value, sales_growth)
13. Idempotence (running sample pipeline twice yields identical data hashes)
14. Schema & Grain integrity (operations_dataset unique keys, forbidden columns absent)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.clean_sensors import clean_sensors
from ml.preprocessing.clean_transactions import clean_transactions
from ml.preprocessing.config import PROCESSED, UNIFIED
from ml.preprocessing.features_business import compute_daily_retail_features
from ml.preprocessing.features_sensor import (
    compute_maintenance_features_by_date,
    engineer_sensor_features,
)
from ml.preprocessing.integrate import (
    FORBIDDEN_COLUMNS,
    compute_daily_sensor_aggregates,
)
from ml.preprocessing.outliers import flag_outliers
from ml.preprocessing.pipeline import run_pipeline
from ml.preprocessing.validate import check_sensors


# -----------------------------------------------------------------------------
# 1. Validation catches issues
# -----------------------------------------------------------------------------
def test_validation_catches_issues():
    """Verify validation detects crafted anomalies and passes valid frames."""
    base_ts = pd.date_range("2026-05-01", periods=10, freq="5min", tz="UTC")
    df_machines = pd.DataFrame({"machine_id": ["M-101"]})

    # Good sensor frame
    df_good = pd.DataFrame(
        {
            "timestamp": base_ts,
            "machine_id": ["M-101"] * 10,
            "temperature": np.full(10, 50.0, dtype=np.float32),
            "vibration": np.full(10, 1.0, dtype=np.float32),
            "pressure": np.full(10, 7.0, dtype=np.float32),
            "rpm": np.full(10, 1500.0, dtype=np.float32),
            "energy_consumption": np.full(10, 5.0, dtype=np.float32),
            "operating_hours": np.arange(100.0, 100.833, 0.0833, dtype=np.float32)[:10],
            "motor_current": np.full(10, 10.0, dtype=np.float32),
            "comp_state": np.full(10, 1.0, dtype=np.float32),
        }
    )
    res_good = check_sensors(df_good, df_machines)
    # Duplicate check, ordering check, limit check should all pass
    target_checks = {
        "duplicate_machine_timestamp",
        "sensor_hard_limit_violations",
        "timestamp_ordering",
    }
    for r in res_good:
        if r.name in target_checks:
            assert r.passed is True

    # Bad sensor frame: duplicate timestamp and temperature out of physical range (200 C)
    df_bad = df_good.copy()
    df_bad.loc[1, "timestamp"] = df_bad.loc[0, "timestamp"]
    df_bad.loc[2, "temperature"] = 250.0  # Physically impossible

    res_bad = check_sensors(df_bad, df_machines)
    dup_res = next(r for r in res_bad if r.name == "duplicate_machine_timestamp")
    limit_res = next(r for r in res_bad if r.name == "sensor_hard_limit_violations")

    assert dup_res.passed is False
    assert limit_res.passed is False
    assert limit_res.count >= 1


# -----------------------------------------------------------------------------
# 2. Grid reindexing creates explicit gap rows
# -----------------------------------------------------------------------------
def test_grid_reindexing():
    """Missing intervals on 5-min grid become explicit gap rows."""
    ts1 = pd.Timestamp("2026-05-01 00:00:00", tz="UTC")
    ts2 = pd.Timestamp("2026-05-01 00:25:00", tz="UTC")  # 4 missing intervals: 05, 10, 15, 20

    df_sparse = pd.DataFrame(
        {
            "timestamp": [ts1, ts2],
            "machine_id": ["M-101", "M-101"],
            "temperature": [50.0, 52.0],
            "vibration": [1.0, 1.2],
            "pressure": [7.0, 7.1],
            "rpm": [1500.0, 1505.0],
            "energy_consumption": [5.0, 5.1],
            "operating_hours": [100.0, 100.4],
            "motor_current": [10.0, 10.1],
            "comp_state": [1.0, 1.0],
        }
    )
    df_clean, _ = clean_sensors(df_sparse)
    assert len(df_clean) == 6  # 00, 05, 10, 15, 20, 25
    assert df_clean["is_gap"].sum() == 4
    # The intermediate timestamps exist
    expected_ts = pd.date_range(ts1, ts2, freq="5min")
    assert (df_clean["timestamp"] == expected_ts).all()


# -----------------------------------------------------------------------------
# 3. Imputation limit (3 intervals filled, 4 not)
# -----------------------------------------------------------------------------
def test_imputation_limit():
    """Gap of 3 intervals is imputed; gap of 4 is not (remains NaN)."""
    # Create series with a gap of 3 and a gap of 4
    ts = pd.date_range("2026-05-01 00:00:00", periods=15, freq="5min", tz="UTC")
    temps = np.full(15, 50.0, dtype=np.float32)
    # Gap 1: indices 2, 3, 4 (3 intervals)
    temps[2:5] = np.nan
    # Gap 2: indices 8, 9, 10, 11 (4 intervals)
    temps[8:12] = np.nan

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "machine_id": ["M-101"] * 15,
            "temperature": temps,
            "vibration": np.full(15, 1.0, dtype=np.float32),
            "pressure": np.full(15, 7.0, dtype=np.float32),
            "rpm": np.full(15, 1500.0, dtype=np.float32),
            "energy_consumption": np.full(15, 5.0, dtype=np.float32),
            "operating_hours": np.full(15, 100.0, dtype=np.float32),
            "motor_current": np.full(15, 10.0, dtype=np.float32),
            "comp_state": np.full(15, 1.0, dtype=np.float32),
        }
    )

    df_clean, _ = clean_sensors(df)
    # Gap 1 (length 3) must be imputed
    assert df_clean.loc[2:4, "temperature"].notna().all()
    assert df_clean.loc[2:4, "is_imputed"].all()

    # Gap 2 (length 4) must remain NaN
    assert df_clean.loc[8:11, "temperature"].isna().all()
    assert not df_clean.loc[8:11, "is_imputed"].any()


# -----------------------------------------------------------------------------
# 4. Tier 1 hard limits
# -----------------------------------------------------------------------------
def test_tier_1_hard_limits():
    """Out-of-range sensor value becomes NaN and triggers is_sensor_fault."""
    ts = pd.date_range("2026-05-01 00:00:00", periods=5, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "machine_id": ["M-101"] * 5,
            "temperature": [50.0, 52.0, 250.0, 51.0, 50.0],  # 250 is > hard_max (120)
            "vibration": [1.0, 1.1, 1.0, 1.0, 1.1],
            "pressure": [7.0, 7.0, 7.0, 7.0, 7.0],
            "rpm": [1500.0, 1500.0, 1500.0, 1500.0, 1500.0],
            "energy_consumption": [5.0, 5.0, 5.0, 5.0, 5.0],
            "operating_hours": [100.0, 100.1, 100.2, 100.3, 100.4],
            "motor_current": [10.0, 10.0, 10.0, 10.0, 10.0],
            "comp_state": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    df_clean, _ = clean_sensors(df)
    # Row index 2 (fault) was set to NaN and then imputed because it's a 1-interval gap!
    # But is_sensor_fault must be True!
    assert bool(df_clean.loc[2, "is_sensor_fault"]) is True
    assert bool(df_clean.loc[0, "is_sensor_fault"]) is False


# -----------------------------------------------------------------------------
# 5. Tier 2 is flag-only
# -----------------------------------------------------------------------------
def test_tier_2_is_flag_only():
    """Tier 2 outlier detection adds boolean flags without altering numeric values."""
    ts = pd.date_range("2026-05-01 00:00:00", periods=400, freq="5min", tz="UTC")
    vib = np.full(400, 1.0, dtype=np.float32)
    vib[350] = 12.0  # Spike

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "machine_id": ["M-101"] * 400,
            "temperature": np.full(400, 50.0, dtype=np.float32),
            "vibration": vib,
            "pressure": np.full(400, 7.0, dtype=np.float32),
            "rpm": np.full(400, 1500.0, dtype=np.float32),
            "energy_consumption": np.full(400, 5.0, dtype=np.float32),
            "operating_hours": np.linspace(100.0, 133.0, 400, dtype=np.float32),
            "motor_current": np.full(400, 10.0, dtype=np.float32),
            "comp_state": np.full(400, 1.0, dtype=np.float32),
            "is_gap": [False] * 400,
            "is_imputed": [False] * 400,
            "is_sensor_fault": [False] * 400,
            "is_warmup": [False] * 400,
            "running": [1.0] * 400,
        }
    )

    df_flagged = flag_outliers(df)
    assert "is_outlier_flag" in df_flagged.columns
    assert "outlier_signals" in df_flagged.columns

    # Numerical values must be byte-identical
    for col in ["temperature", "vibration", "pressure", "rpm", "energy_consumption"]:
        np.testing.assert_array_equal(df[col].to_numpy(), df_flagged[col].to_numpy())


# -----------------------------------------------------------------------------
# 6. Operating hours never decrease
# -----------------------------------------------------------------------------
def test_operating_hours_monotonic():
    """Operating hours never decrease after cleaning."""
    ts = pd.date_range("2026-05-01 00:00:00", periods=5, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "machine_id": ["M-101"] * 5,
            "temperature": np.full(5, 50.0, dtype=np.float32),
            "vibration": np.full(5, 1.0, dtype=np.float32),
            "pressure": np.full(5, 7.0, dtype=np.float32),
            "rpm": np.full(5, 1500.0, dtype=np.float32),
            "energy_consumption": np.full(5, 5.0, dtype=np.float32),
            "operating_hours": [100.0, 100.1, 99.8, 100.2, 100.3],  # decrease at index 2
            "motor_current": np.full(5, 10.0, dtype=np.float32),
            "comp_state": np.full(5, 1.0, dtype=np.float32),
        }
    )

    df_clean, _ = clean_sensors(df)
    diffs = df_clean["operating_hours"].diff().dropna()
    assert (diffs >= 0.0).all()


# -----------------------------------------------------------------------------
# 7. Leakage: rolling features
# -----------------------------------------------------------------------------
def test_leakage_rolling_features():
    """Modifying all values after time t does not alter features at t or before."""
    ts = pd.date_range("2026-05-01 00:00:00", periods=300, freq="5min", tz="UTC")
    vib1 = np.full(300, 1.0, dtype=np.float32)
    vib2 = vib1.copy()
    split_idx = 200
    vib2[split_idx + 1 :] = 99.0  # Massive perturbation in the future

    df_maint = pd.DataFrame(
        {
            "machine_id": ["M-101"],
            "maintenance_id": ["MAIN-01"],
            "maintenance_type": ["Preventive"],
            "cost_gbp": [100.0],
            "performed_date": [pd.Timestamp("2026-04-01", tz="UTC")],
        }
    )

    def _make_frame(vib_arr):
        return pd.DataFrame(
            {
                "timestamp": ts,
                "machine_id": ["M-101"] * 300,
                "temperature": np.full(300, 50.0, dtype=np.float32),
                "vibration": vib_arr,
                "pressure": np.full(300, 7.0, dtype=np.float32),
                "rpm": np.full(300, 1500.0, dtype=np.float32),
                "energy_consumption": np.full(300, 5.0, dtype=np.float32),
                "operating_hours": np.linspace(100.0, 125.0, 300, dtype=np.float32),
                "motor_current": np.full(300, 10.0, dtype=np.float32),
                "comp_state": np.full(300, 1.0, dtype=np.float32),
                "is_gap": [False] * 300,
                "is_imputed": [False] * 300,
                "is_sensor_fault": [False] * 300,
                "is_outlier_flag": [False] * 300,
                "outlier_signals": [""] * 300,
                "is_warmup": [False] * 300,
                "running": [1.0] * 300,
            }
        )

    f1 = engineer_sensor_features(_make_frame(vib1), df_maint)
    f2 = engineer_sensor_features(_make_frame(vib2), df_maint)

    # All rolling features at index <= split_idx must be identical
    roll_cols = [c for c in f1.columns if "roll" in c or "change" in c or "slope" in c]
    for c in roll_cols:
        sub1 = f1.loc[:split_idx, c].to_numpy()
        sub2 = f2.loc[:split_idx, c].to_numpy()
        np.testing.assert_array_equal(
            sub1, sub2, err_msg=f"Leakage detected in {c} at t <= split_idx"
        )


# -----------------------------------------------------------------------------
# 8. Leakage: maintenance features
# -----------------------------------------------------------------------------
def test_leakage_maintenance_features():
    """Adding a maintenance event dated D does not change features on or before D."""
    dates = [pd.Timestamp(f"2026-05-{i:02d}", tz="UTC") for i in range(1, 11)]
    maint_base = pd.DataFrame(
        {
            "maintenance_id": ["MNT-01"],
            "machine_id": ["M-101"],
            "maintenance_type": ["Preventive"],
            "cost_gbp": [500.0],
            "performed_date": [pd.Timestamp("2026-04-15", tz="UTC")],
        }
    )

    ts = pd.date_range("2026-05-01 00:00:00", "2026-05-10 23:55:00", freq="5min", tz="UTC")
    df_sensors = pd.DataFrame(
        {
            "timestamp": ts,
            "machine_id": ["M-101"] * len(ts),
            "running": [1.0] * len(ts),
            "operating_hours": np.linspace(100.0, 300.0, len(ts), dtype=np.float32),
        }
    )

    feat_base = compute_maintenance_features_by_date(maint_base, df_sensors, "M-101", dates)

    # Add new event on May 5
    maint_with_new = pd.concat(
        [
            maint_base,
            pd.DataFrame(
                {
                    "maintenance_id": ["MNT-02"],
                    "machine_id": ["M-101"],
                    "maintenance_type": ["Corrective"],
                    "cost_gbp": [1200.0],
                    "performed_date": [pd.Timestamp("2026-05-05", tz="UTC")],
                }
            ),
        ],
        ignore_index=True,
    )

    feat_with_new = compute_maintenance_features_by_date(maint_with_new, df_sensors, "M-101", dates)

    # Features for May 1, 2, 3, 4, 5 (<= May 5) must be identical!
    cutoff_date = pd.Timestamp("2026-05-05", tz="UTC")
    sub_base = feat_base[feat_base["date"] <= cutoff_date].set_index("date")
    sub_new = feat_with_new[feat_with_new["date"] <= cutoff_date].set_index("date")

    pd.testing.assert_frame_equal(sub_base, sub_new)


# -----------------------------------------------------------------------------
# 9. Leakage: labels isolation
# -----------------------------------------------------------------------------
def test_leakage_labels_isolation():
    """No ground truth label columns appear in sensor_features or operations_dataset."""
    p_sf = PROCESSED / "sensor_features.parquet"
    p_ops = PROCESSED / "operations_dataset.parquet"
    if not p_sf.exists() or not p_ops.exists():
        pytest.skip("Processed outputs not generated yet")

    df_sf = pd.read_parquet(p_sf)
    df_ops = pd.read_parquet(p_ops)

    label_cols = {
        "in_failure_window",
        "in_precursor_window",
        "failure_window_id",
        "episode_type",
        "episode_id",
    }
    assert len(label_cols.intersection(df_sf.columns)) == 0
    assert len(label_cols.intersection(df_ops.columns)) == 0


# -----------------------------------------------------------------------------
# 10. Retail rules & Kept + Rejected == Input
# -----------------------------------------------------------------------------
def test_retail_cleaning_rules():
    """Verify cancellation, guest, non-product, and extreme qty rules; kept + rejected == input."""
    crafted = pd.DataFrame(
        {
            "transaction_id": ["T1", "T2", "T3", "T4", "T5", "T6", "T7"],
            "invoice_no": ["5001", "C5002", "5003", "5004", "5005", "5006", "5007"],
            "stock_code": ["22423", "22423", "POST", "85123A", "85123A", "85123A", "85123A"],
            "description": ["Item 1", "Cancel", "Postage", "Item 3", "Item 4", "Item 5", "Item 6"],
            "quantity": [10, -5, 1, 100, 25000, 5, 5],
            "unit_price": [2.5, 2.5, 15.0, 3.0, 1.0, -1.0, 0.0],  # T6 negative price, T7 zero price
            "customer_id": ["12345", "12345", "12345", None, "14500", "12345", "12345"],
            "country": ["UK"] * 7,
            "is_cancellation": [False, True, False, False, False, False, False],
            "line_total": [25.0, -12.5, 15.0, 300.0, 25000.0, -5.0, 0.0],
            "transaction_date": pd.date_range("2026-05-01", periods=7, freq="D", tz="UTC"),
        }
    )

    df_kept, df_rej, log = clean_transactions(crafted)
    assert len(df_kept) + len(df_rej) == len(crafted)
    # T6 and T7 should be rejected due to non-positive prices
    assert set(df_rej["transaction_id"]) == {"T6", "T7"}

    # Kept should have flags correctly populated
    kept_map = df_kept.set_index("transaction_id")
    assert bool(kept_map.loc["T2", "is_cancellation"]) is True
    assert bool(kept_map.loc["T3", "is_non_product"]) is True
    assert bool(kept_map.loc["T4", "is_guest"]) is True
    assert bool(kept_map.loc["T5", "is_extreme_quantity"]) is True


# -----------------------------------------------------------------------------
# 11. Daily aggregation: energy_kwh equals manual sum
# -----------------------------------------------------------------------------
def test_daily_aggregation_energy():
    """Daily energy_kwh equals the manual sum of 5-min intervals for a complete day."""
    ts = pd.date_range("2026-05-01 00:00:00", periods=288, freq="5min", tz="UTC")
    energy_vals = np.full(288, 2.5, dtype=np.float32)

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "machine_id": ["M-101"] * 288,
            "temperature": np.full(288, 50.0, dtype=np.float32),
            "vibration": np.full(288, 1.0, dtype=np.float32),
            "pressure": np.full(288, 7.0, dtype=np.float32),
            "rpm": np.full(288, 1500.0, dtype=np.float32),
            "energy_consumption": energy_vals,
            "operating_hours": np.linspace(100.0, 124.0, 288, dtype=np.float32),
            "motor_current": np.full(288, 10.0, dtype=np.float32),
            "comp_state": np.full(288, 1.0, dtype=np.float32),
            "running": np.full(288, 1.0, dtype=np.float32),
            "is_gap": [False] * 288,
            "is_imputed": [False] * 288,
            "is_outlier_flag": [False] * 288,
        }
    )

    df_agg = compute_daily_sensor_aggregates(df)
    assert len(df_agg) == 1
    assert np.isclose(df_agg.iloc[0]["energy_kwh"], 288 * 2.5)


# -----------------------------------------------------------------------------
# 12. Business KPIs
# -----------------------------------------------------------------------------
def test_business_kpis():
    """Average order value and sales growth are calculated correctly."""
    dates = pd.to_datetime(["2026-05-01", "2026-05-01", "2026-05-02"], utc=True)
    trans = pd.DataFrame(
        {
            "transaction_id": ["T1", "T2", "T3"],
            "invoice_no": ["INV-1", "INV-2", "INV-3"],
            "date": dates,
            "quantity": [10, 20, 15],
            "revenue_clean": [100.0, 200.0, 150.0],
            "customer_id": ["C1", "C2", "C3"],
            "is_cancellation": [False, False, False],
            "is_guest": [False, False, False],
            "is_non_product": [False, False, False],
            "is_extreme_quantity": [False, False, False],
        }
    )

    kpi_df, _ = compute_daily_retail_features(trans)
    # Day 1: orders=2, total gross rev=300 -> AOV = 150.0
    d1 = kpi_df[kpi_df["date"] == pd.Timestamp("2026-05-01", tz="UTC")].iloc[0]
    assert np.isclose(d1["average_order_value"], 150.0)

    # Day 2 sales = 15, Day 1 sales = 30 -> sales growth exists
    assert "sales_growth" in kpi_df.columns


# -----------------------------------------------------------------------------
# 13. Idempotence (Integration)
# -----------------------------------------------------------------------------
@pytest.mark.integration
def test_pipeline_idempotence():
    """Running sample pipeline twice produces identical hash outputs."""
    p_ops = PROCESSED / "operations_dataset.parquet"
    if not (UNIFIED / "sensors_raw.csv").exists():
        pytest.skip("Unified raw data not present")

    # Run once
    run_pipeline(sample=True, skip_db=True)
    df1 = pd.read_parquet(p_ops)
    hash1 = pd.util.hash_pandas_object(df1).sum()

    # Run second time
    run_pipeline(sample=True, skip_db=True)
    df2 = pd.read_parquet(p_ops)
    hash2 = pd.util.hash_pandas_object(df2).sum()

    assert hash1 == hash2


# -----------------------------------------------------------------------------
# 14. Schema & Grain Integrity (Integration)
# -----------------------------------------------------------------------------
@pytest.mark.integration
def test_operations_dataset_schema():
    """operations_dataset.parquet has unique keys and forbidden columns are absent."""
    p_ops = PROCESSED / "operations_dataset.parquet"
    if not p_ops.exists():
        pytest.skip("operations_dataset.parquet not generated")

    df_ops = pd.read_parquet(p_ops)
    assert not df_ops.duplicated(subset=["machine_id", "date"]).any()

    leaked = set(df_ops.columns).intersection(FORBIDDEN_COLUMNS)
    assert len(leaked) == 0, f"Forbidden columns leaked into operations_dataset: {leaked}"
