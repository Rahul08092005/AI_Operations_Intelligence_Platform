"""Shared plotting styles, figures export, and visual helpers for EDA."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml.eda.config import FIGURES, PALETTE, TEMP_MAX_SPEC, VIB_CRITICAL, VIB_WARNING

if TYPE_CHECKING:
    import matplotlib.axes


def apply_style() -> None:
    """Apply consistent portfolio-quality style with colorblind-safe palette (D51)."""
    sns.set_theme(style="whitegrid")
    sns.set_palette(PALETTE)
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5.5),
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.4,
            "grid.linestyle": "--",
            "figure.autolayout": False,
        }
    )


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save a Matplotlib figure to docs/figures/{name}.png."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    filename = name if name.endswith(".png") else f"{name}.png"
    target_path = FIGURES / filename
    fig.savefig(target_path, dpi=150, bbox_inches="tight")
    return target_path


def mark_holdout(ax: matplotlib.axes.Axes, cutoff: Any = None) -> None:
    """Draw a vertical marker line indicating the start of the holdout period."""
    if cutoff is None:
        from ml.eda.splits import exploration_cutoff

        cutoff = exploration_cutoff()

    ax.axvline(
        cutoff,
        color="#D55E00",
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
        label="Holdout Cutoff",
        zorder=5,
    )
    ylim = ax.get_ylim()
    y_pos = ylim[1] - 0.05 * (ylim[1] - ylim[0])
    ax.text(
        cutoff,
        y_pos,
        " Holdout",
        color="#D55E00",
        fontweight="bold",
        fontsize=9,
        verticalalignment="top",
        zorder=6,
    )


def add_thresholds(ax: matplotlib.axes.Axes, kind: str = "vibration") -> None:
    """Add standard operational reference threshold lines to a plot."""
    if kind == "vibration":
        ax.axhline(
            VIB_WARNING,
            color="#E69F00",
            linestyle=":",
            linewidth=1.6,
            label=f"Warning ({VIB_WARNING} mm/s)",
            zorder=4,
        )
        ax.axhline(
            VIB_CRITICAL,
            color="#D55E00",
            linestyle="--",
            linewidth=1.6,
            label=f"Critical ({VIB_CRITICAL} mm/s)",
            zorder=4,
        )
    elif kind == "temperature":
        ax.axhline(
            TEMP_MAX_SPEC,
            color="#D55E00",
            linestyle="--",
            linewidth=1.6,
            label=f"Max Spec ({TEMP_MAX_SPEC} °C)",
            zorder=4,
        )


def shade_episodes(
    ax: matplotlib.axes.Axes,
    labels_df: pd.DataFrame,
    machine_id: str,
    real_color: str = "#D55E00",
    synth_color: str = "#56B4E9",
    alpha: float = 0.22,
) -> None:
    """Shade ground-truth episode intervals with distinct colors for real vs synthetic."""
    m_labels = labels_df[labels_df["machine_id"] == machine_id].copy()
    if m_labels.empty:
        return

    m_labels = m_labels.sort_values("timestamp")
    # Identify non-normal episodes
    m_labels["is_active"] = (
        m_labels["episode_type"].fillna("normal").ne("normal")
        | m_labels["in_failure_window"].fillna(False)
        | m_labels["in_precursor_window"].fillna(False)
    )

    if not m_labels["is_active"].any():
        return

    # Contiguous interval detection
    change = (
        (m_labels["is_active"] != m_labels["is_active"].shift(1))
        | (m_labels["episode_type"] != m_labels["episode_type"].shift(1))
    ).cumsum()

    active_groups = m_labels[m_labels["is_active"]].groupby(change)
    real_labeled = False
    synth_labeled = False

    for _, grp in active_groups:
        start_t = grp["timestamp"].min()
        end_t = grp["timestamp"].max()
        ep_type = grp["episode_type"].iloc[0]
        source = grp["label_source"].iloc[0] if "label_source" in grp.columns else ""

        is_real = (source == "real") or ("real" in str(ep_type))
        color = real_color if is_real else synth_color

        lbl = None
        if is_real and not real_labeled:
            lbl = "Real Failure Window"
            real_labeled = True
        elif not is_real and not synth_labeled:
            lbl = "Synthetic Episode"
            synth_labeled = True

        ax.axvspan(start_t, end_t, color=color, alpha=alpha, label=lbl, zorder=2)


__all__ = [
    "add_thresholds",
    "apply_style",
    "mark_holdout",
    "save_fig",
    "shade_episodes",
]
