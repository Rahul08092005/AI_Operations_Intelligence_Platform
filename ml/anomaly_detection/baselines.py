"""Per-machine robust baseline normalization.

Implements Decision D53 (Per-machine robust baseline normalization: median and 1.4826 * MAD).
Ensures zero leakage across machines and time splits (Rule R3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCALE_EPSILON = 1e-6
MIN_TRAINING_ROWS = 5000

# Mapping of known machines to equipment types if metadata is absent
DEFAULT_MACHINE_TYPES: dict[str, str] = {
    "M-101": "Air Compressor",
    "M-102": "Air Compressor",
    "M-103": "Air Compressor",
    "M-104": "Air Compressor",
    "M-105": "Hydraulic Press",
    "M-106": "Hydraulic Press",
    "M-107": "Hydraulic Press",
    "M-108": "Hydraulic Press",
    "M-109": "Conveyor Drive",
    "M-110": "Conveyor Drive",
    "M-111": "Conveyor Drive",
    "M-112": "Conveyor Drive",
}


def _compute_median_mad(arr: np.ndarray) -> tuple[float, float]:
    """Compute median and 1.4826 * MAD of a 1D numeric array (ignoring NaNs)."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return 0.0, 1.0
    med = float(np.median(valid))
    mad = float(np.median(np.abs(valid - med)))
    scale = float(1.4826 * mad)
    if scale < SCALE_EPSILON:
        scale = SCALE_EPSILON
    return med, scale


@dataclass
class MachineBaseline:
    """Stores per-machine robust baseline statistics (median and 1.4826 * MAD)."""

    feature_list: list[str] = field(default_factory=list)
    machine_stats: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
    type_stats: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
    fleet_stats: dict[str, tuple[float, float]] = field(default_factory=dict)
    machine_types: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MACHINE_TYPES))
    row_counts: dict[str, int] = field(default_factory=dict)

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_list: list[str],
        machines_df: pd.DataFrame | None = None,
    ) -> MachineBaseline:
        """Compute robust medians and scales strictly from training rows (Rule R3).

        Parameters
        ----------
        train_df : pd.DataFrame
            Training partition rows with complete features and machine_id.
        feature_list : list[str]
            Feature columns to normalize.
        machines_df : pd.DataFrame | None
            Optional machine metadata containing machine_id and machine_type.
        """
        self.feature_list = list(feature_list)
        self.machine_stats = {}
        self.type_stats = {}
        self.fleet_stats = {}
        self.row_counts = {}

        # 1. Update machine types if provided
        if (
            machines_df is not None
            and "machine_id" in machines_df.columns
            and "machine_type" in machines_df.columns
        ):
            for _, r in machines_df.iterrows():
                self.machine_types[str(r["machine_id"])] = str(r["machine_type"])

        # 2. Compute Fleet-level fallback statistics
        for col in self.feature_list:
            vals = train_df[col].to_numpy(dtype=np.float64, na_value=np.nan)
            self.fleet_stats[col] = _compute_median_mad(vals)

        # 3. Compute Type-level fallback statistics
        train_df_copy = train_df.copy()
        train_df_copy["_m_type"] = (
            train_df_copy["machine_id"]
            .astype(str)
            .map(lambda m: self.machine_types.get(m, "Unknown"))
        )

        for m_type, grp in train_df_copy.groupby("_m_type", observed=True):
            self.type_stats[str(m_type)] = {}
            for col in self.feature_list:
                vals = grp[col].to_numpy(dtype=np.float64, na_value=np.nan)
                self.type_stats[str(m_type)][col] = _compute_median_mad(vals)

        # 4. Compute Per-machine statistics with fallback if rows < 5,000
        for m_id, grp in train_df.groupby("machine_id", observed=True):
            m_str = str(m_id)
            n_m = len(grp)
            self.row_counts[m_str] = n_m
            m_type = self.machine_types.get(m_str, "Unknown")

            self.machine_stats[m_str] = {}
            if n_m < MIN_TRAINING_ROWS:
                logger.warning(
                    "Machine %s has %d training rows (< %d). Falling back to type/fleet scale.",
                    m_str,
                    n_m,
                    MIN_TRAINING_ROWS,
                )
                # Use type or fleet statistics
                fallback = self.type_stats.get(m_type, self.fleet_stats)
                for col in self.feature_list:
                    self.machine_stats[m_str][col] = fallback[col]
            else:
                for col in self.feature_list:
                    vals = grp[col].to_numpy(dtype=np.float64, na_value=np.nan)
                    self.machine_stats[m_str][col] = _compute_median_mad(vals)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features into per-machine robust z-scores (float32).

        Z = (X - median) / scale
        """
        out = pd.DataFrame(index=df.index)
        m_ids = df["machine_id"].astype(str)

        # Pre-allocate arrays for fast vector assignment
        for col in self.feature_list:
            col_vals = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
            z_vals = np.full(len(df), np.nan, dtype=np.float32)

            for m_str in m_ids.unique():
                m_mask = (m_ids == m_str).to_numpy()
                m_type = self.machine_types.get(m_str, "Unknown")

                # Retrieve baseline parameters: machine -> type -> fleet
                if m_str in self.machine_stats:
                    med, scale = self.machine_stats[m_str][col]
                elif m_type in self.type_stats:
                    med, scale = self.type_stats[m_type][col]
                else:
                    med, scale = self.fleet_stats[col]

                # Compute robust z
                z_vals[m_mask] = ((col_vals[m_mask] - med) / scale).astype(np.float32)

            out[col] = z_vals

        return out

    def to_dict(self) -> dict[str, Any]:
        """Serialize baseline parameters to dictionary."""
        return {
            "feature_list": self.feature_list,
            "machine_stats": self.machine_stats,
            "type_stats": self.type_stats,
            "fleet_stats": self.fleet_stats,
            "machine_types": self.machine_types,
            "row_counts": self.row_counts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MachineBaseline:
        """Reconstruct MachineBaseline from serialized dictionary."""
        mb = cls()
        mb.feature_list = list(data["feature_list"])
        mb.machine_stats = data["machine_stats"]
        mb.type_stats = data["type_stats"]
        mb.fleet_stats = data["fleet_stats"]
        mb.machine_types = data.get("machine_types", dict(DEFAULT_MACHINE_TYPES))
        mb.row_counts = data.get("row_counts", {})
        return mb
