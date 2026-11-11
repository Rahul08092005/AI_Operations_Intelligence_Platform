"""Evaluation metrics: event-level detection, lead time, false-alarm budget, and interval AP.

Implements Decision D58, D60, D62:
- Primary: Event-level real failure window detection (early, late, missed, lead time)
- Synthetic episode detection (time to first Warning/Critical, peak score)
- False-alarm budget audit (scaled to 30 days, M-109 spike test, Critical count)
- Secondary: Interval-level average precision with block-bootstrap 95% CI
- Demo state report (R9 check).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml.anomaly_detection.config import (
    BUFFER_HOURS,
    MAX_CRITICAL_PER_30D,
    MAX_WARNING_PER_30D,
    NEGATIVE_EPISODE_TYPES,
    SEED,
    STATE_RUNNING,
    STATUS_CRITICAL,
    STATUS_NORMAL,
    STATUS_WARNING,
    TRUE_ANOMALY_EPISODE_TYPES,
)
from ml.anomaly_detection.masks import compute_buffer_mask

logger = logging.getLogger(__name__)

# Predefined ground-truth real failure windows on M-104
METROPT_FAILURE_WINDOWS: list[dict[str, Any]] = [
    {
        "failure_id": "FAIL-001",
        "machine_id": "M-104",
        "start": pd.Timestamp("2026-06-03 00:00:00", tz="UTC"),
        "end": pd.Timestamp("2026-06-06 23:55:00", tz="UTC"),
        "period": "exploration",
    },
    {
        "failure_id": "FAIL-002",
        "machine_id": "M-104",
        "start": pd.Timestamp("2026-07-14 23:30:00", tz="UTC"),
        "end": pd.Timestamp("2026-07-18 06:00:00", tz="UTC"),
        "period": "exploration",
    },
    {
        "failure_id": "FAIL-003",
        "machine_id": "M-104",
        "start": pd.Timestamp("2026-07-21 10:00:00", tz="UTC"),
        "end": pd.Timestamp("2026-07-26 14:30:00", tz="UTC"),
        "period": "exploration",
    },
    {
        "failure_id": "FAIL-004",
        "machine_id": "M-104",
        "start": pd.Timestamp("2026-08-30 14:30:00", tz="UTC"),
        "end": pd.Timestamp("2026-09-02 19:00:00", tz="UTC"),
        "period": "holdout",
    },
]


def evaluate_real_windows(
    episodes_df: pd.DataFrame,
    windows: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Evaluate event-level detection on real failure windows (Decision D62 / Section 6.7).

    Rules:
    - Early: alert episode starts in [start - 72h, start).
             lead_time_hours = (start - alert_start)
    - Late:  alert episode starts in [start, end].
             lead_time_hours = 0.0
    - Missed: no alert episode overlaps [start - 72h, end].
    """
    target_windows = list(windows or METROPT_FAILURE_WINDOWS)
    results: list[dict[str, Any]] = []

    for w in target_windows:
        w_id = str(w["failure_id"])
        m_id = str(w["machine_id"])
        w_start = pd.to_datetime(w["start"], utc=True)
        w_end = pd.to_datetime(w["end"], utc=True)
        w_period = str(w.get("period", "unknown"))

        pre_start = w_start - pd.Timedelta(hours=72)

        # Alerts on this machine
        m_alerts = episodes_df[episodes_df["machine_id"] == m_id]

        # Check overlapping alerts in [pre_start, w_end]
        overlapping = m_alerts[
            (m_alerts["end"] >= pre_start) & (m_alerts["start"] <= w_end)
        ].sort_values("start")

        if len(overlapping) == 0:
            status = "missed"
            lead_time = np.nan
            peak_score = np.nan
            severity = "None"
        else:
            first_alert = overlapping.iloc[0]
            alert_start = pd.to_datetime(first_alert["start"], utc=True)
            peak_score = float(overlapping["peak_score"].max())
            severity = (
                STATUS_CRITICAL
                if (overlapping["severity"] == STATUS_CRITICAL).any()
                else STATUS_WARNING
            )

            if alert_start < w_start:
                status = "early"
                lead_time = round((w_start - alert_start).total_seconds() / 3600.0, 2)
            else:
                status = "late"
                lead_time = 0.0

        results.append(
            {
                "failure_id": w_id,
                "machine_id": m_id,
                "window_start": w_start,
                "window_end": w_end,
                "period": w_period,
                "status": status,
                "lead_time_hours": lead_time,
                "peak_score": peak_score,
                "severity": severity,
            }
        )

    return pd.DataFrame(results)


def evaluate_synthetic_episodes(
    episodes_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    scored_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate detection metrics for labeled synthetic problem episodes."""
    # Find all unique non-null synthetic episode_ids
    syn_labels = labels_df[
        labels_df["episode_type"].isin(TRUE_ANOMALY_EPISODE_TYPES)
        & labels_df["episode_id"].notna()
        & (labels_df["label_source"] == "synthetic")
    ]

    syn_results: list[dict[str, Any]] = []

    for ep_id, grp in syn_labels.groupby("episode_id", observed=True):
        m_id = str(grp["machine_id"].iloc[0])
        ep_type = str(grp["episode_type"].iloc[0])
        ep_start = grp["timestamp"].min()
        ep_end = grp["timestamp"].max()

        # Scored rows in episode
        scored_m = scored_df[
            (scored_df["machine_id"] == m_id)
            & (scored_df["timestamp"] >= ep_start)
            & (scored_df["timestamp"] <= ep_end)
        ]
        peak_score = (
            float(scored_m["anomaly_score"].max()) if len(scored_m) > 0 else np.nan
        )

        # Alert episodes overlapping [ep_start, ep_end]
        m_alerts = episodes_df[
            (episodes_df["machine_id"] == m_id)
            & (episodes_df["end"] >= ep_start)
            & (episodes_df["start"] <= ep_end)
        ].sort_values("start")

        if len(m_alerts) > 0:
            first_alert = m_alerts.iloc[0]
            alert_start = pd.to_datetime(first_alert["start"], utc=True)
            hours_to_alert = max(0.0, (alert_start - ep_start).total_seconds() / 3600.0)
            detected = True
            crit_alerts = m_alerts[m_alerts["severity"] == STATUS_CRITICAL]
            if len(crit_alerts) > 0:
                crit_ts = pd.to_datetime(crit_alerts["start"].iloc[0], utc=True)
                hours_to_crit = max(0.0, (crit_ts - ep_start).total_seconds() / 3600.0)
            else:
                hours_to_crit = np.nan
        else:
            detected = False
            hours_to_alert = np.nan
            hours_to_crit = np.nan

        syn_results.append(
            {
                "episode_id": str(ep_id),
                "machine_id": m_id,
                "episode_type": ep_type,
                "start": ep_start,
                "end": ep_end,
                "detected": detected,
                "hours_to_alert": (
                    round(hours_to_alert, 2) if not np.isnan(hours_to_alert) else np.nan
                ),
                "hours_to_critical": (
                    round(hours_to_crit, 2) if not np.isnan(hours_to_crit) else np.nan
                ),
                "peak_score": round(peak_score, 4) if not np.isnan(peak_score) else np.nan,
            }
        )

    return pd.DataFrame(syn_results)


def false_alarm_report(
    episodes_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    total_days: float,
) -> tuple[bool, dict[str, Any], pd.DataFrame]:
    """Audit false-alarm budget across healthy and benign-spike machines (Section 6.6).

    Rules:
    - Normal/benign periods: outside 24h problem buffer
    - Budget:
      1. <= 1.0 Warning-or-higher episode per machine per 30 days
      2. 0 Critical episodes outside problem episodes
      3. M-109 (benign spikes): 0 Critical and Warning rate within budget.
    """
    in_buffer = compute_buffer_mask(labels_df, buffer_hours=BUFFER_HOURS)
    is_negative = labels_df["episode_type"].isin(NEGATIVE_EPISODE_TYPES) & (~in_buffer)

    # Filter episodes starting in negative intervals
    fa_episodes: list[dict[str, Any]] = []
    for _, ep in episodes_df.iterrows():
        ep_start = ep["start"]
        m_id = ep["machine_id"]
        # Check if episode start falls in negative period for that machine
        m_neg = labels_df[
            (labels_df["machine_id"] == m_id) & (labels_df["timestamp"] == ep_start)
        ]
        if len(m_neg) > 0 and is_negative.loc[m_neg.index[0]]:
            fa_episodes.append(ep.to_dict())

    df_fa = pd.DataFrame(fa_episodes) if fa_episodes else pd.DataFrame(columns=episodes_df.columns)

    # Breakdown per machine
    machines = sorted(scored_df["machine_id"].unique())
    machine_rows: list[dict[str, Any]] = []
    total_warn_fa = 0
    total_crit_fa = 0

    scale_30d = 30.0 / max(total_days, 1.0)

    m109_crit_fa = 0
    m109_warn_rate_30d = 0.0

    for m in machines:
        m_fa = df_fa[df_fa["machine_id"] == m] if len(df_fa) > 0 else pd.DataFrame()
        n_warn = len(m_fa)
        n_crit = len(m_fa[m_fa["severity"] == STATUS_CRITICAL]) if len(m_fa) > 0 else 0

        rate_30d = round(n_warn * scale_30d, 3)
        total_warn_fa += n_warn
        total_crit_fa += n_crit

        if m == "M-109":
            m109_crit_fa = n_crit
            m109_warn_rate_30d = rate_30d

        machine_rows.append(
            {
                "machine_id": m,
                "warning_fa_count": n_warn,
                "critical_fa_count": n_crit,
                "warning_fa_per_30d": rate_30d,
            }
        )

    pooled_rate_30d = round((total_warn_fa / len(machines)) * scale_30d, 3)

    # Check budget criteria
    pass_pooled_warn = pooled_rate_30d <= MAX_WARNING_PER_30D
    pass_total_crit = total_crit_fa <= MAX_CRITICAL_PER_30D
    pass_m109 = (m109_crit_fa == 0) and (m109_warn_rate_30d <= MAX_WARNING_PER_30D)

    budget_passed = bool(pass_pooled_warn and pass_total_crit and pass_m109)

    summary = {
        "budget_passed": budget_passed,
        "pooled_warning_fa_per_30d": pooled_rate_30d,
        "total_critical_fa": total_crit_fa,
        "m109_warning_fa_per_30d": m109_warn_rate_30d,
        "m109_critical_fa": m109_crit_fa,
        "pass_pooled_warn": pass_pooled_warn,
        "pass_total_crit": pass_total_crit,
        "pass_m109": pass_m109,
    }

    return budget_passed, summary, pd.DataFrame(machine_rows)


def interval_metrics(
    scored_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    block_hours: int = 24,
    n_boot: int = 500,
) -> dict[str, Any]:
    """Compute interval-level average precision (PR-AUC) with block bootstrap 95% CI."""
    can_score = scored_df["state"] == STATE_RUNNING
    scores = scored_df.loc[can_score, "anomaly_score"]
    valid_mask = can_score & scores.notna()

    df_eval = scored_df.loc[valid_mask, ["machine_id", "timestamp", "anomaly_score"]].copy()
    lbl_eval = labels_df.loc[valid_mask, ["episode_type", "label_source"]].copy()

    in_buffer = compute_buffer_mask(labels_df, buffer_hours=BUFFER_HOURS).loc[valid_mask]

    # Target 1: All True Anomalies (positives vs negative periods)
    is_pos_all = lbl_eval["episode_type"].isin(TRUE_ANOMALY_EPISODE_TYPES)
    is_neg = lbl_eval["episode_type"].isin(NEGATIVE_EPISODE_TYPES) & (~in_buffer)

    mask_all = is_pos_all | is_neg
    y_all = is_pos_all.loc[mask_all].astype(int).to_numpy()
    s_all = df_eval.loc[mask_all, "anomaly_score"].to_numpy(dtype=np.float64)

    # Target 2: Real MetroPT failure + precursor only
    is_pos_real = lbl_eval["episode_type"].isin(["real_failure", "real_precursor"])
    mask_real = is_pos_real | is_neg
    y_real = is_pos_real.loc[mask_real].astype(int).to_numpy()
    s_real = df_eval.loc[mask_real, "anomaly_score"].to_numpy(dtype=np.float64)

    # Point estimates
    ap_all = float(average_precision_score(y_all, s_all)) if len(np.unique(y_all)) > 1 else 0.0
    ap_real = float(average_precision_score(y_real, s_real)) if len(np.unique(y_real)) > 1 else 0.0

    prev_all = float(np.mean(y_all)) if len(y_all) > 0 else 0.0
    prev_real = float(np.mean(y_real)) if len(y_real) > 0 else 0.0

    # Deterministic block bootstrap
    rng = np.random.default_rng(SEED)
    time_blocks = df_eval["timestamp"].dt.floor(f"{block_hours}h").unique()
    n_blocks = len(time_blocks)

    boot_ap_all: list[float] = []
    boot_ap_real: list[float] = []

    if n_blocks > 5:
        # Precompute block indices
        block_series = df_eval["timestamp"].dt.floor(f"{block_hours}h")
        block_dict: dict[Any, np.ndarray] = {
            b: np.where(block_series == b)[0] for b in time_blocks
        }

        for _ in range(n_boot):
            sampled_blocks = rng.choice(time_blocks, size=n_blocks, replace=True)
            sampled_indices = np.concatenate([block_dict[b] for b in sampled_blocks])

            # Sampled Target 1
            s_mask_all = mask_all.to_numpy()[sampled_indices]
            if np.any(s_mask_all):
                y_b = is_pos_all.to_numpy()[sampled_indices][s_mask_all].astype(int)
                if len(np.unique(y_b)) > 1:
                    s_b = df_eval["anomaly_score"].to_numpy()[sampled_indices][s_mask_all]
                    boot_ap_all.append(float(average_precision_score(y_b, s_b)))

            # Sampled Target 2
            s_mask_real = mask_real.to_numpy()[sampled_indices]
            if np.any(s_mask_real):
                y_b_r = is_pos_real.to_numpy()[sampled_indices][s_mask_real].astype(int)
                if len(np.unique(y_b_r)) > 1:
                    s_b_r = df_eval["anomaly_score"].to_numpy()[sampled_indices][s_mask_real]
                    boot_ap_real.append(float(average_precision_score(y_b_r, s_b_r)))

    ci_all = (
        (float(np.quantile(boot_ap_all, 0.025)), float(np.quantile(boot_ap_all, 0.975)))
        if len(boot_ap_all) >= 20
        else (ap_all, ap_all)
    )
    ci_real = (
        (float(np.quantile(boot_ap_real, 0.025)), float(np.quantile(boot_ap_real, 0.975)))
        if len(boot_ap_real) >= 20
        else (ap_real, ap_real)
    )

    return {
        "ap_all_true_anomalies": round(ap_all, 4),
        "prevalence_all": round(prev_all, 4),
        "ci_all": (round(ci_all[0], 4), round(ci_all[1], 4)),
        "ap_real_only": round(ap_real, 4),
        "prevalence_real": round(prev_real, 4),
        "ci_real": (round(ci_real[0], 4), round(ci_real[1], 4)),
    }


def final_state_report(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Audit the demo storyline states at the final timestamp (Rule R9)."""
    expected_story = {
        "M-101": STATUS_NORMAL,
        "M-102": STATUS_NORMAL,
        "M-103": STATUS_WARNING,
        "M-104": STATUS_CRITICAL,
    }

    report_rows: list[dict[str, Any]] = []
    for m_id, grp in scored_df.groupby("machine_id", observed=True):
        m_str = str(m_id)
        last_row = grp.sort_values("timestamp").iloc[-1]

        actual_status = str(last_row["anomaly_status"])
        actual_score = (
            float(last_row["anomaly_score"])
            if pd.notna(last_row["anomaly_score"])
            else np.nan
        )
        expected = expected_story.get(m_str, "N/A")
        matches = actual_status == expected if expected != "N/A" else True

        report_rows.append(
            {
                "machine_id": m_str,
                "last_timestamp": last_row["timestamp"],
                "state": last_row["state"],
                "anomaly_score": round(actual_score, 4) if not np.isnan(actual_score) else np.nan,
                "actual_status": actual_status,
                "expected_demo_status": expected,
                "matches_demo_story": matches,
            }
        )

    return pd.DataFrame(report_rows)
