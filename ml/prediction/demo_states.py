"""Demo machine operating states assembly and scenario mixture simulation.

Implements Decision D78, Caveat C6, Rule R11, and Section 6.9:
- Assembles 30 daily operating states for all 12 demo machines ending at TIMELINE_END.
- Pools (computed from TEST partition, bucket thresholds from TRAINING partition):
  * low: failure=0, torque and tool_wear <= 75th percentile of training
  * stressed: failure=0, torque and tool_wear > 75th percentile of training
  * high: failure=1
- Scenarios interpolate mixtures from day 1 to day 30:
  * hero_degrading: 94/5/1 -> 5/25/70
  * mild_drift: 94/5/1 -> 50/40/10
  * overdue_maintenance: 72/25/3 -> 72/25/3
  * in_maintenance: 94/5/1 -> 94/5/1 (final day marked Maintenance)
  * all others: 94/5/1 -> 94/5/1
- Strictly sampled with replacement from the test partition only.
- Output tables:
  * data/processed/machine_ai4i_state.parquet
  * data/processed/machine_risk_daily.parquet
  * data/processed/machine_risk_latest.csv (12 rows, is_replay=True)
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ml.prediction.config import (
    DEMO_STATES_PATH,
    FROZEN_CONFIG_PATH,
    HOLDOUT_LOG_PATH,
    MACHINE_RISK_DAILY_PATH,
    MACHINE_RISK_LATEST_PATH,
    MODEL_PKL_PATH,
    RAW_DIR,
    SEED,
)
from ml.prediction.data import load_test, load_train, verify_frozen_config
from ml.prediction.model import FailureModel
from ml.prediction.thresholds import assign_risk_category

logger = logging.getLogger(__name__)

TIMELINE_END = datetime.date(2026, 10, 19)
N_DAYS = 30


def get_scenario_mixture(scenario: str, day_idx: int, n_days: int = N_DAYS) -> np.ndarray:
    """Compute interpolated [p_low, p_stressed, p_high] for a given day (0 to n_days-1)."""
    alpha = day_idx / (n_days - 1) if n_days > 1 else 1.0

    if scenario == "hero_degrading":
        start_mix = np.array([0.94, 0.05, 0.01])
        end_mix = np.array([0.05, 0.25, 0.70])
    elif scenario == "mild_drift":
        start_mix = np.array([0.94, 0.05, 0.01])
        end_mix = np.array([0.50, 0.40, 0.10])
    elif scenario == "overdue_maintenance":
        start_mix = np.array([0.72, 0.25, 0.03])
        end_mix = np.array([0.72, 0.25, 0.03])
    else:
        # in_maintenance or healthy / other
        start_mix = np.array([0.94, 0.05, 0.01])
        end_mix = np.array([0.94, 0.05, 0.01])

    mix = (1.0 - alpha) * start_mix + alpha * end_mix
    return mix / np.sum(mix)


def build_demo_states(
    frozen_config_path: Path | str | None = None,
    machines_csv_path: Path | str | None = None,
) -> pd.DataFrame:
    """Sample 360 daily operating states for 12 machines across 30 days from test partition."""
    verify_frozen_config(frozen_config_path or FROZEN_CONFIG_PATH)

    # Verify holdout log entry exists
    if not HOLDOUT_LOG_PATH.exists():
        raise PermissionError(
            f"Holdout log not found at {HOLDOUT_LOG_PATH}. "
            "Test evaluation must run before demo states."
        )

    # 1. Compute 75th percentiles from training partition to avoid peeking
    train_df = load_train()
    t_torque_75 = float(train_df["torque_nm"].quantile(0.75))
    t_wear_75 = float(train_df["tool_wear_min"].quantile(0.75))
    logger.info(
        "Training quantiles (75th percentile): torque=%.2f, wear=%.2f",
        t_torque_75,
        t_wear_75,
    )

    # 2. Build pools from test partition
    test_df = load_test(frozen_config_path or FROZEN_CONFIG_PATH)
    logger.info("Building demo pools from %d test rows...", len(test_df))

    pool_low = test_df[
        (test_df["machine_failure"] == 0)
        & (test_df["torque_nm"] <= t_torque_75)
        & (test_df["tool_wear_min"] <= t_wear_75)
    ].copy()

    pool_stressed = test_df[
        (test_df["machine_failure"] == 0)
        & (test_df["torque_nm"] > t_torque_75)
        & (test_df["tool_wear_min"] > t_wear_75)
    ].copy()

    pool_high = test_df[test_df["machine_failure"] == 1].copy()

    logger.info(
        "Pool sizes: Low=%d, Stressed=%d, High=%d (all from test partition)",
        len(pool_low),
        len(pool_stressed),
        len(pool_high),
    )

    # 3. Load machines catalog
    m_csv = Path(machines_csv_path or (RAW_DIR / "unified" / "machines.csv"))
    machines_df = pd.read_csv(m_csv)

    date_range = [
        TIMELINE_END - datetime.timedelta(days=(N_DAYS - 1 - i)) for i in range(N_DAYS)
    ]

    records: list[dict] = []

    for _m_idx, m_row in machines_df.iterrows():
        m_id = str(m_row["machine_id"])
        scenario = str(m_row["scenario"])
        target_type = str(m_row["ai4i_type"])

        # Seed per machine for reproducible sampling
        m_seed = SEED + int(m_id.split("-")[-1])
        rng = np.random.default_rng(m_seed)

        for day_idx, snap_date in enumerate(date_range):
            # Check maintenance final day special case
            is_final_day = day_idx == (N_DAYS - 1)
            if scenario == "in_maintenance" and is_final_day:
                # Mark as Maintenance
                records.append(
                    {
                        "machine_id": m_id,
                        "snapshot_date": snap_date,
                        "pool": "maintenance",
                        "source_udi": -1,
                        "type": target_type,
                        "air_temp_k": np.nan,
                        "process_temp_k": np.nan,
                        "rpm": np.nan,
                        "torque_nm": np.nan,
                        "tool_wear_min": np.nan,
                        "is_maintenance_day": True,
                    }
                )
                continue

            mix = get_scenario_mixture(scenario, day_idx, N_DAYS)
            chosen_pool_name = rng.choice(["low", "stressed", "high"], p=mix)

            if chosen_pool_name == "low":
                pool_df = pool_low
            elif chosen_pool_name == "stressed":
                pool_df = pool_stressed
            else:
                pool_df = pool_high

            # Match type if pool has >= 20 rows of that type
            type_subset = pool_df[pool_df["type"] == target_type]
            if len(type_subset) >= 20:
                selected_row = type_subset.iloc[rng.integers(0, len(type_subset))]
            else:
                selected_row = pool_df.iloc[rng.integers(0, len(pool_df))]

            records.append(
                {
                    "machine_id": m_id,
                    "snapshot_date": snap_date,
                    "pool": chosen_pool_name,
                    "source_udi": int(selected_row["udi"]),
                    "type": str(selected_row["type"]),
                    "air_temp_k": float(selected_row["air_temp_k"]),
                    "process_temp_k": float(selected_row["process_temp_k"]),
                    "rpm": float(selected_row["rpm"]),
                    "torque_nm": float(selected_row["torque_nm"]),
                    "tool_wear_min": float(selected_row["tool_wear_min"]),
                    "is_maintenance_day": False,
                }
            )

    states_df = pd.DataFrame(records)
    logger.info("Assembled %d machine operating state rows across 12 machines.", len(states_df))
    return states_df


def build_and_save_machine_risk() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score demo states with frozen FailureModel and output daily and latest risk tables."""
    states_df = build_demo_states()

    # Save state table
    DEMO_STATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    states_df.to_parquet(DEMO_STATES_PATH, index=False)
    logger.info("Saved machine_ai4i_state.parquet to %s", DEMO_STATES_PATH)

    # Load frozen FailureModel
    model = FailureModel.load(MODEL_PKL_PATH)

    # Score valid rows
    valid_mask = ~states_df["is_maintenance_day"]
    valid_states = states_df[valid_mask].copy()

    probs = model.predict_proba(valid_states)
    cats = assign_risk_category(probs, model.thresholds)

    states_df["failure_probability"] = np.nan
    states_df["risk_level"] = "Maintenance"
    states_df["model_version"] = model.model_version

    states_df.loc[valid_mask, "failure_probability"] = np.round(probs, 4)
    states_df.loc[valid_mask, "risk_level"] = cats

    # Compute trailing 7-day mean and max per machine
    states_df["risk_7d_mean"] = (
        states_df.groupby("machine_id")["failure_probability"]
        .rolling(7, min_periods=1)
        .mean()
        .reset_index(drop=True)
    )
    states_df["risk_7d_max"] = (
        states_df.groupby("machine_id")["failure_probability"]
        .rolling(7, min_periods=1)
        .max()
        .reset_index(drop=True)
    )

    states_df["risk_7d_mean"] = np.round(states_df["risk_7d_mean"], 4)
    states_df["risk_7d_max"] = np.round(states_df["risk_7d_max"], 4)

    # Save daily risk table
    MACHINE_RISK_DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    states_df.to_parquet(MACHINE_RISK_DAILY_PATH, index=False)
    logger.info("Saved machine_risk_daily.parquet to %s", MACHINE_RISK_DAILY_PATH)

    # Build latest 12-row table (Day 30 snapshot)
    m_csv = RAW_DIR / "unified" / "machines.csv"
    machines_df = pd.read_csv(m_csv)

    latest_rows: list[dict] = []
    for _, m_row in machines_df.iterrows():
        m_id = str(m_row["machine_id"])
        scenario = str(m_row["scenario"])
        m_type = str(m_row["machine_type"])

        m_daily = states_df[states_df["machine_id"] == m_id].sort_values("snapshot_date")
        last_row = m_daily.iloc[-1]

        # Calculate implied level for 7d mean
        r_7d = last_row["risk_7d_mean"]
        if pd.notna(r_7d):
            level_7d = assign_risk_category(np.array([r_7d]), model.thresholds)[0]
        else:
            level_7d = "Maintenance"

        latest_rows.append(
            {
                "machine_id": m_id,
                "snapshot_date": str(last_row["snapshot_date"]),
                "machine_type": m_type,
                "scenario": scenario,
                "failure_probability": last_row["failure_probability"],
                "risk_level": last_row["risk_level"],
                "risk_7d_mean": r_7d,
                "risk_7d_level": level_7d,
                "is_replay": True,
            }
        )

    latest_df = pd.DataFrame(latest_rows)
    MACHINE_RISK_LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    latest_df.to_csv(MACHINE_RISK_LATEST_PATH, index=False)
    logger.info("Saved machine_risk_latest.csv to %s", MACHINE_RISK_LATEST_PATH)

    return states_df, states_df, latest_df
