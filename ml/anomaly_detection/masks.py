"""Training and scoring masks with state gating and label isolation.

Implements Decision D54 (Operating state gating) and Decision D56 (Training exclusions).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ml.anomaly_detection.config import (
    BUFFER_HOURS,
    EXCLUDED_EPISODE_TYPES,
    STATE_NODATA,
    STATE_OFFLINE,
    STATE_RUNNING,
)

logger = logging.getLogger(__name__)


def scoring_mask(
    features_df: pd.DataFrame, feature_list: list[str]
) -> tuple[pd.Series, pd.Series]:
    """Compute the scoring mask and operating state for each row.

    Rules (R5, D54):
    - Rows with running = False receive state 'Offline'
    - Rows with warm-up, gaps, sensor faults, or missing features receive state 'NoData'
    - Only running, complete rows receive state 'Running' and can_score = True.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing telemetry, quality flags, and machine_id.
    feature_list : list[str]
        List of feature column names required by the model.

    Returns
    -------
    can_score : pd.Series[bool]
        Boolean mask of rows eligible for scoring.
    state : pd.Series[str]
        Categorical/string operating state: 'Running', 'Offline', or 'NoData'.
    """
    state = pd.Series(STATE_NODATA, index=features_df.index, dtype="string")

    # Flag checks (safely handling boolean or numeric flags)
    is_running = features_df["running"].fillna(False).astype(bool)
    is_warmup = (
        features_df["is_warmup"].fillna(False).astype(bool)
        if "is_warmup" in features_df.columns
        else pd.Series(False, index=features_df.index)
    )
    is_gap = (
        features_df["is_gap"].fillna(False).astype(bool)
        if "is_gap" in features_df.columns
        else pd.Series(False, index=features_df.index)
    )
    is_fault = (
        features_df["is_sensor_fault"].fillna(False).astype(bool)
        if "is_sensor_fault" in features_df.columns
        else pd.Series(False, index=features_df.index)
    )

    # Feature completeness
    missing_features = features_df[feature_list].isna().any(axis=1)

    # Classification
    offline_mask = ~is_running
    nodata_mask = is_running & (is_warmup | is_gap | is_fault | missing_features)
    running_mask = is_running & (~is_warmup) & (~is_gap) & (~is_fault) & (~missing_features)

    state[offline_mask] = STATE_OFFLINE
    state[nodata_mask] = STATE_NODATA
    state[running_mask] = STATE_RUNNING

    can_score = running_mask
    return can_score, state


def compute_buffer_mask(
    labels_df: pd.DataFrame, buffer_hours: int = BUFFER_HOURS
) -> pd.Series:
    """Identify rows that fall within +/- buffer_hours of any excluded problem episode.

    Vectorized per machine using a rolling centered window.
    24h buffer on 5-min intervals = 288 intervals each side -> window = 577.
    """
    buffer_intervals = int(round(buffer_hours * 60 / 5))
    window_size = 2 * buffer_intervals + 1

    in_buffer = pd.Series(False, index=labels_df.index)
    is_problem = labels_df["episode_type"].isin(EXCLUDED_EPISODE_TYPES)

    # Group by machine to ensure buffers do not leak across machines
    for _, group in labels_df.groupby("machine_id", observed=True):
        m_idx = group.index
        # Rolling max centered
        m_problem = is_problem.loc[m_idx]
        m_buffered = (
            m_problem.rolling(window=window_size, center=True, min_periods=1)
            .max()
            .astype(bool)
        )
        in_buffer.loc[m_idx] = m_buffered

    return in_buffer


def training_mask(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_list: list[str],
    period: tuple[pd.Timestamp, pd.Timestamp],
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute training mask and exclusion breakdown report (Decision D54, D56).

    Parameters
    ----------
    features_df : pd.DataFrame
        Sensor features DataFrame.
    labels_df : pd.DataFrame
        Ground truth anomaly labels DataFrame aligned with features_df.
    feature_list : list[str]
        List of required model features.
    period : tuple[pd.Timestamp, pd.Timestamp]
        (start_time, end_time) of the training period (end_time exclusive).

    Returns
    -------
    train_mask : pd.Series[bool]
        Boolean series of rows to include in training.
    exclusion_report : pd.DataFrame
        Summary table detailing kept and excluded rows per machine and reason.
    """
    start_ts, end_ts = period
    in_period = (features_df["timestamp"] >= start_ts) & (features_df["timestamp"] < end_ts)

    # 1. Scoring mask (operating state)
    can_score, state = scoring_mask(features_df, feature_list)

    # 2. Episode exclusions & 24h buffer
    is_excluded_ep = labels_df["episode_type"].isin(EXCLUDED_EPISODE_TYPES)
    in_buffer = compute_buffer_mask(labels_df, buffer_hours=BUFFER_HOURS)

    # Overall training eligibility
    is_train = in_period & can_score & (~is_excluded_ep) & (~in_buffer)

    # Build detailed exclusion report per machine
    machines = features_df["machine_id"].unique()
    report_rows: list[dict[str, Any]] = []

    for m in machines:
        m_mask = features_df["machine_id"] == m
        m_in_period = m_mask & in_period

        total_period = int(m_in_period.sum())
        if total_period == 0:
            continue

        offline_count = int((m_in_period & (state == STATE_OFFLINE)).sum())
        warmup_count = int(
            (m_in_period & features_df["is_warmup"].fillna(False).astype(bool)).sum()
            if "is_warmup" in features_df.columns
            else 0
        )
        gap_count = int(
            (m_in_period & features_df["is_gap"].fillna(False).astype(bool)).sum()
            if "is_gap" in features_df.columns
            else 0
        )
        fault_count = int(
            (m_in_period & features_df["is_sensor_fault"].fillna(False).astype(bool)).sum()
            if "is_sensor_fault" in features_df.columns
            else 0
        )
        missing_count = int(
            (
                m_in_period
                & features_df["running"].fillna(False).astype(bool)
                & features_df[feature_list].isna().any(axis=1)
            ).sum()
        )
        episode_count = int((m_in_period & is_excluded_ep).sum())
        buffer_only_count = int((m_in_period & (~is_excluded_ep) & in_buffer).sum())
        kept_count = int((m_in_period & is_train).sum())

        report_rows.append(
            {
                "machine_id": m,
                "total_in_period": total_period,
                "offline": offline_count,
                "warmup": warmup_count,
                "gap": gap_count,
                "fault": fault_count,
                "missing_features": missing_count,
                "excluded_episode": episode_count,
                "buffer_only": buffer_only_count,
                "kept_training_rows": kept_count,
                "kept_pct": round(kept_count / total_period * 100, 2) if total_period > 0 else 0.0,
            }
        )

    exclusion_report = pd.DataFrame(report_rows)
    return is_train, exclusion_report
