"""Alert episode extraction and anomaly type labeling.

Implements Decision D58 (Alert episodes: contiguous Warning+ runs >= 3 intervals,
merging runs separated by < 2h) and Decision D61 / D66 (anomaly_type and peak attributes).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ml.anomaly_detection.config import (
    ALERT_MERGE_GAP_HOURS,
    ALERT_MIN_INTERVALS,
    STATUS_CRITICAL,
    STATUS_WARNING,
)

logger = logging.getLogger(__name__)


def extract_alert_episodes(
    scored_df: pd.DataFrame,
    min_intervals: int = ALERT_MIN_INTERVALS,
    merge_gap_hours: float = ALERT_MERGE_GAP_HOURS,
) -> pd.DataFrame:
    """Extract and merge alert episodes from scored sensor intervals.

    Parameters
    ----------
    scored_df : pd.DataFrame
        DataFrame containing machine_id, timestamp, anomaly_score, anomaly_status,
        and factor columns (factor_1, factor_1_z, factor_2_z, factor_3_z).
    min_intervals : int
        Minimum contiguous intervals in Warning or Critical to qualify as an episode.
    merge_gap_hours : float
        Maximum gap in hours between consecutive alert episodes to merge them.

    Returns
    -------
    episodes_df : pd.DataFrame
        Table of alert episodes with start, end, peak_score, peak_time, severity,
        dominant_factors, and anomaly_type.
    """
    required_cols = {"machine_id", "timestamp", "anomaly_score", "anomaly_status"}
    missing = required_cols - set(scored_df.columns)
    if missing:
        raise ValueError(f"scored_df missing required columns: {sorted(missing)}")

    df = scored_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["machine_id", "timestamp"])

    is_alert = df["anomaly_status"].isin([STATUS_WARNING, STATUS_CRITICAL])

    episodes: list[dict[str, Any]] = []

    for m_id, grp in df.groupby("machine_id", observed=True):
        m_str = str(m_id)
        grp_alert = grp[is_alert.loc[grp.index]].copy()
        if len(grp_alert) == 0:
            continue

        # Identify contiguous runs
        # An alert step is contiguous if delta <= 15 minutes (allowing for single missed 5m grid)
        ts_alert = grp_alert["timestamp"]
        deltas = ts_alert.diff().dt.total_seconds().fillna(0.0) / 60.0
        # If delta > 15 min, it's a new run
        run_ids = (deltas > 15.0).cumsum()

        # Extract initial raw candidate runs
        raw_runs: list[dict[str, Any]] = []
        for _, run_grp in grp_alert.groupby(run_ids):
            if len(run_grp) < min_intervals:
                continue

            r_start = run_grp["timestamp"].iloc[0]
            r_end = run_grp["timestamp"].iloc[-1]
            raw_runs.append(
                {
                    "machine_id": m_str,
                    "start": r_start,
                    "end": r_end,
                    "intervals": len(run_grp),
                    "indices": run_grp.index.tolist(),
                }
            )

        if not raw_runs:
            continue

        # Merge runs on the same machine separated by less than merge_gap_hours
        merged_runs: list[dict[str, Any]] = [raw_runs[0]]
        for cur in raw_runs[1:]:
            prev = merged_runs[-1]
            gap_h = (cur["start"] - prev["end"]).total_seconds() / 3600.0

            if gap_h < merge_gap_hours:
                # Merge into prev
                prev["end"] = cur["end"]
                prev["intervals"] += cur["intervals"]
                prev["indices"].extend(cur["indices"])
            else:
                merged_runs.append(cur)

        # Compute peak metrics and dominant factor attribution for each merged episode
        for ep in merged_runs:
            ep_indices = ep["indices"]
            ep_slice = grp.loc[ep_indices]

            # Peak score and peak time
            peak_score = float(ep_slice["anomaly_score"].max())
            peak_rows = ep_slice[ep_slice["anomaly_score"] == peak_score]
            peak_time = peak_rows["timestamp"].iloc[0]

            # Severity: Critical if any interval was Critical, else Warning
            severity = (
                STATUS_CRITICAL
                if (ep_slice["anomaly_status"] == STATUS_CRITICAL).any()
                else STATUS_WARNING
            )

            duration_h = float((ep["end"] - ep["start"]).total_seconds() / 3600.0)

            # Anomaly type and dominant factor attribution (Decision D61 / D66)
            anomaly_type, dominant_factors = _assign_anomaly_type(ep_slice)

            episodes.append(
                {
                    "machine_id": m_str,
                    "start": ep["start"],
                    "end": ep["end"],
                    "duration_hours": round(duration_h, 3),
                    "peak_score": round(peak_score, 4),
                    "peak_time": peak_time,
                    "severity": severity,
                    "anomaly_type": anomaly_type,
                    "dominant_factors": dominant_factors,
                }
            )

    if not episodes:
        return pd.DataFrame(
            columns=[
                "machine_id",
                "start",
                "end",
                "duration_hours",
                "peak_score",
                "peak_time",
                "severity",
                "anomaly_type",
                "dominant_factors",
            ]
        )

    return pd.DataFrame(episodes).sort_values(["start", "machine_id"]).reset_index(drop=True)


def _assign_anomaly_type(ep_slice: pd.DataFrame) -> tuple[str, str]:
    """Determine dominant factors and assigned anomaly_type label for an episode.

    Rule (D61):
    If top factor's share of summed absolute z of top 3 is >= 0.50:
      label "{signal}_{direction}" (e.g. "vibration_high")
    Else:
      label "multivariate".
    """
    if "factor_1" not in ep_slice.columns or "factor_1_z" not in ep_slice.columns:
        return "multivariate", "unknown"

    # Average absolute z of top 3 factors across the alert episode
    z1_mean = float(ep_slice["factor_1_z"].abs().mean())
    z2_mean = (
        float(ep_slice["factor_2_z"].abs().mean())
        if "factor_2_z" in ep_slice.columns
        else 0.0
    )
    z3_mean = (
        float(ep_slice["factor_3_z"].abs().mean())
        if "factor_3_z" in ep_slice.columns
        else 0.0
    )

    total_z = z1_mean + z2_mean + z3_mean
    dominant_feat = str(ep_slice["factor_1"].mode().iloc[0]) if len(ep_slice) > 0 else "unknown"
    dominant_dir = (
        str(ep_slice["factor_1_direction"].mode().iloc[0])
        if "factor_1_direction" in ep_slice.columns
        else "high"
    )

    base_signal = dominant_feat.split("_")[0]  # e.g. vibration from vibration_roll_mean_1h

    if total_z > 0 and (z1_mean / total_z) >= 0.50:
        anomaly_type = f"{base_signal}_{dominant_dir}"
    else:
        anomaly_type = "multivariate"

    dominant_factors_summary = f"{dominant_feat}:{dominant_dir} (Z={z1_mean:.1f})"
    return anomaly_type, dominant_factors_summary
