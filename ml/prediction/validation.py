"""Input validation and telemetry range checking for failure prediction.

Enforces Decision D80:
- Validates presence and types of the 6 required operating state features.
- Converts API aliases to canonical feature names.
- Flags operating conditions falling outside the training envelope with range warnings.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ml.prediction.config import RAW_FEATURE_COLS, TYPE_MAP

logger = logging.getLogger(__name__)

# Column name aliases accepted from API payloads
COLUMN_ALIASES: dict[str, str] = {
    "Type": "type",
    "air_temperature_k": "air_temp_k",
    "Air temperature [K]": "air_temp_k",
    "Air temperature [k]": "air_temp_k",
    "process_temperature_k": "process_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Process temperature [k]": "process_temp_k",
    "rotational_speed_rpm": "rpm",
    "Rotational speed [rpm]": "rpm",
    "torque_nm": "torque_nm",
    "Torque [Nm]": "torque_nm",
    "tool_wear_min": "tool_wear_min",
    "Tool wear [min]": "tool_wear_min",
}

# Default training range bounds (calculated from AI4I training partition + 10% margin)
# min_allowed = min - 0.10 * span, max_allowed = max + 0.10 * span
DEFAULT_FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "air_temp_k": (293.30, 306.40),       # Train [295.3, 304.4], span=9.1
    "process_temp_k": (303.70, 315.80),   # Train [305.7, 313.8], span=8.1
    "rpm": (1032.0, 3012.0),              # Train [1168, 2876], span=1708
    "torque_nm": (-4.50, 83.90),          # Train [3.8, 75.6], span=71.8
    "tool_wear_min": (-25.30, 278.30),    # Train [0.0, 253.0], span=253.0
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names using alias mapping."""
    rename_dict = {col: COLUMN_ALIASES[col] for col in df.columns if col in COLUMN_ALIASES}
    return df.rename(columns=rename_dict)


def validate_inputs(
    df: pd.DataFrame,
    feature_ranges: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, list[list[str]]]:
    """Validate operating state input DataFrame and check for range warnings.

    Parameters
    ----------
    df : pd.DataFrame
        Input table containing the 6 required operating state signals.
    feature_ranges : dict[str, tuple[float, float]] | None
        Allowed (min, max) range bounds per continuous feature. Defaults to DEFAULT_FEATURE_RANGES.

    Returns
    -------
    validated_df : pd.DataFrame
        Clean DataFrame guaranteed to contain canonical RAW_FEATURE_COLS with validated types.
    warnings_per_row : list[list[str]]
        List of warning strings for each row whose values fall outside the training envelope.

    Raises
    ------
    ValueError or TypeError
        If required columns are missing, types are non-numeric, or 'type' is invalid.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    clean_df = normalize_column_names(df.copy())
    ranges = feature_ranges or DEFAULT_FEATURE_RANGES

    # 1. Check required columns exist
    missing_cols = [col for col in RAW_FEATURE_COLS if col not in clean_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required operating state features: {missing_cols}. "
            f"Expected: {RAW_FEATURE_COLS}"
        )

    # 2. Validate 'type' column
    clean_df["type"] = clean_df["type"].astype(str).str.strip().str.upper()
    invalid_types = clean_df[~clean_df["type"].isin(TYPE_MAP.keys())]["type"].unique()
    if len(invalid_types) > 0:
        raise ValueError(
            f"Invalid machine variant type(s): {invalid_types.tolist()}. "
            f"Allowed types: {list(TYPE_MAP.keys())}"
        )

    # 3. Validate numeric columns
    numeric_cols = [c for c in RAW_FEATURE_COLS if c != "type"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(clean_df[col]):
            # Attempt coercion
            coerced = pd.to_numeric(clean_df[col], errors="coerce")
            if coerced.isna().any():
                raise TypeError(
                    f"Column '{col}' must be numeric; found invalid non-numeric values."
                )
            clean_df[col] = coerced

        if clean_df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NaN or null values.")

    # 4. Check for range warnings (min and max with 10% margin)
    warnings_per_row: list[list[str]] = []
    for _, row in clean_df.iterrows():
        row_warnings: list[str] = []
        for col in numeric_cols:
            val = float(row[col])
            min_bound, max_bound = ranges.get(col, (-np.inf, np.inf))
            if val < min_bound or val > max_bound:
                msg = (
                    f"{col} outside training range "
                    f"({val:.1f} not in [{min_bound:.1f}, {max_bound:.1f}])"
                )
                row_warnings.append(msg)
        warnings_per_row.append(row_warnings)

    return clean_df[RAW_FEATURE_COLS].copy(), warnings_per_row
