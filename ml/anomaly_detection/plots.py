"""Figure generation module for Day 5 Anomaly Detection (Figures F11 to F14).

Implements portfolio-quality plotting with:
- Caveats C1 (Synthetic Dynamics) and C3 (M-104 Overlays) stated in subtitles.
- Colorblind-safe palettes and consistent formatting via ml.eda.plotting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml.anomaly_detection.config import (
    FIGURES_DIR,
    STATUS_CRITICAL_THRESHOLD,
    STATUS_WARNING_THRESHOLD,
)
from ml.eda.plotting import apply_style, mark_holdout, save_fig, shade_episodes

logger = logging.getLogger(__name__)


def generate_f11_timelines(
    scored_primary_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """F11: Full timeline anomaly scores for M-101, M-103, M-104, M-109."""
    apply_style()
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    target_machines = ["M-101", "M-103", "M-104", "M-109"]
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, sharey=True)

    fig.suptitle(
        "F11: Trailing Smoothed Anomaly Scores Across Key Fleet Machines\n"
        "[Caveats C1 & C3: M-104 vibration/temperature overlays are synthetic; "
        "M-109 contains benign vibration stress spikes]",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )

    for i, m_id in enumerate(target_machines):
        ax = axes[i]
        m_scored = scored_primary_df[
            scored_primary_df["machine_id"] == m_id
        ].sort_values("timestamp")

        # Plot smoothed score
        ax.plot(
            m_scored["timestamp"],
            m_scored["anomaly_score"],
            color="#0072B2",
            linewidth=1.2,
            label="Smoothed Anomaly Score (1h)" if i == 0 else "",
        )

        # Thresholds
        ax.axhline(
            STATUS_WARNING_THRESHOLD,
            color="#E69F00",
            linestyle=":",
            linewidth=1.2,
            label="Warning (0.50)" if i == 0 else "",
        )
        ax.axhline(
            STATUS_CRITICAL_THRESHOLD,
            color="#D55E00",
            linestyle="--",
            linewidth=1.2,
            label="Critical (0.80)" if i == 0 else "",
        )

        # Ground truth episode shading
        shade_episodes(ax, labels_df, machine_id=m_id, alpha=0.20)

        # Mark holdout
        mark_holdout(ax)

        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"{m_id}\nScore")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    axes[-1].set_xlabel("Timeline (UTC)")

    plt.tight_layout()
    fig_path = save_fig(fig, "fig11_anomaly_score_timelines.png")
    plt.close(fig)
    logger.info("Saved F11 figure to %s", fig_path)
    return fig_path


def generate_f12_lead_times(
    primary_real_res: pd.DataFrame,
    real_only_real_res: pd.DataFrame,
    primary_syn_res: pd.DataFrame,
    real_only_syn_res: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """F12: Lead times per real failure window and time-to-first-alert per synthetic episode."""
    apply_style()
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    fig.suptitle(
        "F12: Event Detection Comparison — Primary (Plan Core) vs Real-Only Comparator (Set D)\n"
        "[Caveat C3: Detection on Primary benefits from synthetic ramps; "
        "Real-Only reflects honest physical telemetry]",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )

    # Panel 1: Real Failure Windows Lead Time
    df_real = pd.DataFrame(
        {
            "Window": primary_real_res["failure_id"],
            "Primary Model": primary_real_res["lead_time_hours"].fillna(0.0),
            "Real-Only Model": real_only_real_res["lead_time_hours"].fillna(0.0),
        }
    )
    df_real_melt = df_real.melt(id_vars="Window", var_name="Model", value_name="Lead Time (Hours)")

    sns.barplot(
        data=df_real_melt,
        x="Window",
        y="Lead Time (Hours)",
        hue="Model",
        palette=["#0072B2", "#E69F00"],
        ax=axes[0],
    )
    axes[0].set_title("Real Failure Windows: Lead Time (Hours)")
    axes[0].set_ylabel("Early Lead Time (Hours)")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Synthetic Episodes Hours to Alert
    common_syn = sorted(
        set(primary_syn_res["episode_id"]).intersection(set(real_only_syn_res["episode_id"]))
    )
    if common_syn:
        prim_sub = primary_syn_res.set_index("episode_id").loc[common_syn].reset_index()
        real_sub = real_only_syn_res.set_index("episode_id").loc[common_syn].reset_index()

        df_syn = pd.DataFrame(
            {
                "Episode": prim_sub["episode_id"],
                "Primary Model": prim_sub["hours_to_alert"].fillna(999.0),
                "Real-Only Model": real_sub["hours_to_alert"].fillna(999.0),
            }
        )
        df_syn_melt = df_syn.melt(id_vars="Episode", var_name="Model", value_name="Hours to Alert")
        df_syn_melt["Hours to Alert"] = df_syn_melt["Hours to Alert"].replace(999.0, np.nan)

        sns.barplot(
            data=df_syn_melt,
            x="Episode",
            y="Hours to Alert",
            hue="Model",
            palette=["#0072B2", "#E69F00"],
            ax=axes[1],
        )
        axes[1].set_title("Synthetic Episodes: Hours from Start to Alert")
        axes[1].set_ylabel("Hours to First Alert")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_path = save_fig(fig, "fig12_event_detection_lead_times.png")
    plt.close(fig)
    logger.info("Saved F12 figure to %s", fig_path)
    return fig_path


def generate_f13_model_vs_baselines(
    experiments_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """F13: Episode recall versus false alarms per 30 days for A, B, C, D, B1, B2."""
    apply_style()
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    palette = {
        "plan_core": "#0072B2",
        "core_context": "#56B4E9",
        "core_temporal": "#009E73",
        "real_only": "#E69F00",
        "vibration_temperature": "#D55E00",
    }

    sns.scatterplot(
        data=experiments_df,
        x="pooled_warning_fa_per_30d",
        y="episode_recall",
        hue="feature_set",
        style="detector",
        palette=palette,
        s=80,
        alpha=0.75,
        ax=ax,
    )

    ax.axvline(
        1.0,
        color="#D55E00",
        linestyle="--",
        linewidth=1.8,
        label="FA Budget Limit (1.0 / 30d)",
    )

    ax.set_title(
        "F13: Episode Recall vs False Alarm Rate across All CV Experiments (Pooled)\n"
        "[Budget Constraint: <= 1.0 Warning FA / 30d, 0 Critical FAs]",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Warning False Alarms per Machine per 30 Days")
    ax.set_ylabel("Validation Episode Recall")
    ax.set_xlim(-0.1, max(3.5, float(experiments_df["pooled_warning_fa_per_30d"].max()) + 0.5))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_path = save_fig(fig, "fig13_model_vs_baselines.png")
    plt.close(fig)
    logger.info("Saved F13 figure to %s", fig_path)
    return fig_path


def generate_f14_m104_contributing_factors(
    scored_primary_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """F14: M-104 final 14 days smoothed score above, contributing factor deviations below."""
    apply_style()
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    m104_df = scored_primary_df[
        scored_primary_df["machine_id"] == "M-104"
    ].sort_values("timestamp")
    max_ts = m104_df["timestamp"].max()
    start_ts = max_ts - pd.Timedelta(days=14)

    m104_14d = m104_df[m104_df["timestamp"] >= start_ts].copy()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]}
    )

    fig.suptitle(
        "F14: M-104 Final 14 Days Hero Degradation & Root-Cause Attribution\n"
        "[Top: Calibrated Anomaly Score | Bottom: Absolute Robust Z-Scores of Telemetry Drivers]",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Score & Thresholds
    ax1.plot(
        m104_14d["timestamp"],
        m104_14d["anomaly_score"],
        color="#D55E00",
        linewidth=1.8,
        label="Anomaly Score",
    )
    ax1.axhline(
        STATUS_WARNING_THRESHOLD,
        color="#E69F00",
        linestyle=":",
        linewidth=1.2,
        label="Warning (0.50)",
    )
    ax1.axhline(
        STATUS_CRITICAL_THRESHOLD,
        color="#D55E00",
        linestyle="--",
        linewidth=1.2,
        label="Critical (0.80)",
    )
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Score")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Factor deviations
    ax2.plot(
        m104_14d["timestamp"],
        m104_14d["factor_1_z"].abs(),
        label="Rank 1 Factor |Z|",
        color="#0072B2",
        alpha=0.85,
    )
    ax2.plot(
        m104_14d["timestamp"],
        m104_14d["factor_2_z"].abs(),
        label="Rank 2 Factor |Z|",
        color="#009E73",
        alpha=0.75,
    )
    ax2.plot(
        m104_14d["timestamp"],
        m104_14d["factor_3_z"].abs(),
        label="Rank 3 Factor |Z|",
        color="#CC79A7",
        alpha=0.65,
    )

    ax2.set_ylabel("Factor Deviation (|Z|)")
    ax2.set_xlabel("Timestamp (UTC)")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_path = save_fig(fig, "fig14_m104_contributing_factors.png")
    plt.close(fig)
    logger.info("Saved F14 figure to %s", fig_path)
    return fig_path
