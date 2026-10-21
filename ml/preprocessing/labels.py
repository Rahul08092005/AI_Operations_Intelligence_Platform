"""Ground-truth anomaly evaluation labels generator.

Produces `data/processed/anomaly_eval_labels.parquet` without contaminating
feature tables. Provides clean targets for Day 5 model evaluation:
- Real MetroPT air leak failure windows and 72h precursor windows on M-104.
- Synthetic ground-truth episodes across the fleet traced strictly to Day 2 metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ml.preprocessing.config import TIMELINE_END

logger = logging.getLogger(__name__)


def generate_anomaly_eval_labels(
    df_clean_sensors: pd.DataFrame,
    df_failure_windows: pd.DataFrame,
    metadata: dict[str, Any],
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate ground truth anomaly evaluation labels aligned with clean sensors."""
    logger.info("Generating anomaly evaluation ground-truth labels...")
    shift_days = metadata.get("shift_days", 2240)
    shift_delta = pd.Timedelta(days=shift_days)

    labels_df = pd.DataFrame(
        {
            "machine_id": df_clean_sensors["machine_id"].astype("string"),
            "timestamp": df_clean_sensors["timestamp"],
            "in_failure_window": False,
            "in_precursor_window": False,
            "failure_window_id": pd.Series([None] * len(df_clean_sensors), dtype="string"),
            "episode_type": "normal",
            "episode_id": pd.Series([None] * len(df_clean_sensors), dtype="string"),
            "label_source": "synthetic",
        }
    )

    timeline_end_ts = (
        pd.Timestamp(TIMELINE_END, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
    )

    # 1. Real MetroPT failure windows on M-104
    m104_mask = labels_df["machine_id"] == "M-104"
    for _, fw in df_failure_windows.iterrows():
        fw_id = str(fw["failure_id"])
        ws = pd.to_datetime(fw["start_time"], utc=True) + shift_delta
        we = pd.to_datetime(fw["end_time"], utc=True) + shift_delta
        pre_start = ws - pd.Timedelta(hours=72)

        # Real failure window
        fw_match = m104_mask & (labels_df["timestamp"] >= ws) & (labels_df["timestamp"] <= we)
        labels_df.loc[fw_match, "in_failure_window"] = True
        labels_df.loc[fw_match, "failure_window_id"] = fw_id
        labels_df.loc[fw_match, "episode_type"] = "real_failure"
        labels_df.loc[fw_match, "episode_id"] = fw_id
        labels_df.loc[fw_match, "label_source"] = "real"

        # Real precursor window (72h prior)
        pre_match = (
            m104_mask & (labels_df["timestamp"] >= pre_start) & (labels_df["timestamp"] < ws)
        )
        # Don't overwrite an actual failure if precursor overlaps
        pre_eligible = pre_match & (~labels_df["in_failure_window"])
        labels_df.loc[pre_eligible, "in_precursor_window"] = True
        labels_df.loc[pre_eligible, "failure_window_id"] = fw_id
        labels_df.loc[pre_eligible, "episode_type"] = "real_precursor"
        labels_df.loc[pre_eligible, "episode_id"] = f"{fw_id}-PRE"
        labels_df.loc[pre_eligible, "label_source"] = "real"

    # 2. Hero degradation on M-104 (final 35 days)
    hero_start = pd.Timestamp(TIMELINE_END, tz="UTC") - pd.Timedelta(days=35)
    hero_mask = (
        m104_mask
        & (labels_df["timestamp"] >= hero_start)
        & (~labels_df["in_failure_window"])
        & (~labels_df["in_precursor_window"])
    )
    labels_df.loc[hero_mask, "episode_type"] = "hero_degradation"
    labels_df.loc[hero_mask, "episode_id"] = "HERO-DEG-M104"
    labels_df.loc[hero_mask, "label_source"] = "synthetic"

    # 3. Mild drift on M-103 (final 21 days)
    m103_mask = labels_df["machine_id"] == "M-103"
    drift_start = pd.Timestamp(TIMELINE_END, tz="UTC") - pd.Timedelta(days=21)
    drift_mask = m103_mask & (labels_df["timestamp"] >= drift_start)
    labels_df.loc[drift_mask, "episode_type"] = "mild_drift"
    labels_df.loc[drift_mask, "episode_id"] = "DRIFT-M103"
    labels_df.loc[drift_mask, "label_source"] = "synthetic"

    # 4. Synthetic failures with 3d precursor & 24h outage: M-105, M-110, M-111
    failures_spec = {
        "M-105": {"days_offset": 120, "ep_id": "SYN-FAIL-M105"},
        "M-110": {"days_offset": 75, "ep_id": "SYN-FAIL-M110"},
        "M-111": {"days_offset": 58, "ep_id": "SYN-FAIL-M111"},
    }

    for m_id, spec in failures_spec.items():
        m_mask = labels_df["machine_id"] == m_id
        if not m_mask.any():
            continue
        center = timeline_end_ts - pd.Timedelta(days=spec["days_offset"])
        pre_start = center - pd.Timedelta(days=3)
        out_end = center + pd.Timedelta(hours=24)

        # Precursor
        pre_m = m_mask & (labels_df["timestamp"] >= pre_start) & (labels_df["timestamp"] < center)
        labels_df.loc[pre_m, "in_precursor_window"] = True
        labels_df.loc[pre_m, "episode_type"] = "synthetic_failure_precursor"
        labels_df.loc[pre_m, "episode_id"] = f"{spec['ep_id']}-PRE"
        labels_df.loc[pre_m, "label_source"] = "synthetic"

        # Outage
        out_m = m_mask & (labels_df["timestamp"] >= center) & (labels_df["timestamp"] < out_end)
        labels_df.loc[out_m, "in_failure_window"] = True
        labels_df.loc[out_m, "episode_type"] = "synthetic_failure_outage"
        labels_df.loc[out_m, "episode_id"] = spec["ep_id"]
        labels_df.loc[out_m, "label_source"] = "synthetic"

    # 5. Pre-maintenance drift on M-106 (30d up to 5d before end)
    m106_mask = labels_df["machine_id"] == "M-106"
    if m106_mask.any():
        maint_time = timeline_end_ts - pd.Timedelta(days=5)
        drift_m106_start = maint_time - pd.Timedelta(days=30)
        m106_drift_mask = (
            m106_mask
            & (labels_df["timestamp"] >= drift_m106_start)
            & (labels_df["timestamp"] < maint_time)
        )
        labels_df.loc[m106_drift_mask, "episode_type"] = "pre_maintenance_drift"
        labels_df.loc[m106_drift_mask, "episode_id"] = "PRE-MAINT-M106"
        labels_df.loc[m106_drift_mask, "label_source"] = "synthetic"

    # 6. Benign spikes on M-109
    m109_mask = labels_df["machine_id"] == "M-109"
    if m109_mask.any() and "vibration" in df_clean_sensors.columns:
        m109_sensor = df_clean_sensors[df_clean_sensors["machine_id"] == "M-109"]
        med = m109_sensor["vibration"].rolling(288, min_periods=72).median()
        spike_indices = m109_sensor.index[m109_sensor["vibration"] > 1.5 * med]
        labels_df.loc[spike_indices, "episode_type"] = "benign_spike"
        labels_df.loc[spike_indices, "episode_id"] = "SPIKE-M109"
        labels_df.loc[spike_indices, "label_source"] = "synthetic"

    # 7. Planned maintenance outage on M-112 (last 36h)
    m112_mask = labels_df["machine_id"] == "M-112"
    if m112_mask.any():
        maint_m112_start = timeline_end_ts - pd.Timedelta(hours=36)
        m112_outage_mask = m112_mask & (labels_df["timestamp"] >= maint_m112_start)
        labels_df.loc[m112_outage_mask, "episode_type"] = "planned_maintenance_outage"
        labels_df.loc[m112_outage_mask, "episode_id"] = "MAINT-OUTAGE-M112"
        labels_df.loc[m112_outage_mask, "label_source"] = "synthetic"

    labels_df["machine_id"] = labels_df["machine_id"].astype("category")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labels_df.to_parquet(output_path, index=False)
        logger.info("Saved anomaly evaluation labels to %s (%d rows)", output_path, len(labels_df))

    return labels_df
