"""Feature engineering and encoding for failure risk prediction.

Implements Decision D69 and Rule R4:
- F0 (raw): type_code, air_temp_k, process_temp_k, rpm, torque_nm, tool_wear_min
- F1 (physics): F0 plus continuous physics quantities:
    - temp_diff_k = process_temp_k - air_temp_k
    - power_w = torque_nm * rpm * 2 * pi / 60
    - strain_nmin = tool_wear_min * torque_nm
- Fixed ordinal encoding for type (L=0, M=1, H=2).
- Strict exclusion of leakage columns (R3).
- No threshold copying or discretized failure indicators (R4).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ml.prediction.config import (
    FEATURE_SETS,
    FORBIDDEN_COLS,
    TYPE_MAP,
)
from ml.prediction.validation import validate_inputs

logger = logging.getLogger(__name__)


def get_feature_names(feature_set: str = "physics") -> list[str]:
    """Return the ordered feature column names for the requested feature set."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set '{feature_set}'. Allowed: {list(FEATURE_SETS.keys())}"
        )
    return list(FEATURE_SETS[feature_set])


def build_features(
    df: pd.DataFrame,
    feature_set: str = "physics",
    validate: bool = True,
) -> pd.DataFrame:
    """Build engineered feature matrix for model training and serving.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing the required raw operating state columns.
    feature_set : str
        Feature set specification ('raw' or 'physics'). Defaults to 'physics'.
    validate : bool
        Whether to run input validation and range checking. Defaults to True.

    Returns
    -------
    pd.DataFrame
        Clean feature matrix with exact expected column ordering, typed as float64.

    Raises
    ------
    ValueError
        If unknown feature set or validation fails.
    AssertionError
        If any forbidden column enters the feature output.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set '{feature_set}'. Expected one of {list(FEATURE_SETS.keys())}"
        )

    expected_cols = get_feature_names(feature_set)

    # 1. Validate inputs if requested
    if validate:
        clean_df, _ = validate_inputs(df)
    else:
        clean_df = df.copy()

    # 2. Encode type: L=0, M=1, H=2
    if "type_code" not in clean_df.columns:
        if "type" in clean_df.columns:
            types_upper = clean_df["type"].astype(str).str.strip().str.upper()
            clean_df["type_code"] = types_upper.map(TYPE_MAP)
            if clean_df["type_code"].isna().any():
                invalid_types = clean_df.loc[clean_df["type_code"].isna(), "type"].unique()
                raise ValueError(f"Unknown machine type(s): {invalid_types}")
        else:
            raise ValueError("Input data must contain 'type' column for type_code encoding.")

    # 3. Compute continuous physics quantities if requested
    if feature_set == "physics":
        # Temperature difference (K)
        clean_df["temp_diff_k"] = clean_df["process_temp_k"] - clean_df["air_temp_k"]

        # Mechanical power (Watts): P = tau * omega = torque_nm * (rpm * 2 * pi / 60)
        clean_df["power_w"] = (
            clean_df["torque_nm"] * clean_df["rpm"] * (2.0 * np.pi / 60.0)
        )

        # Strain proxy (Nm * min): strain = tool_wear_min * torque_nm
        clean_df["strain_nmin"] = clean_df["tool_wear_min"] * clean_df["torque_nm"]

    # 4. Strict leakage guard assertion (Rule R3, D70)
    leakage_present = set(clean_df.columns).intersection(FORBIDDEN_COLS)
    assert not leakage_present, (
        f"CRITICAL LEAKAGE VIOLATION: Forbidden columns in feature matrix: {leakage_present}"
    )

    # 5. Extract features in deterministic column order and cast to float64
    out_df = clean_df[expected_cols].astype(np.float64).copy()

    # Verify no NaN values
    if out_df.isna().any().any():
        nan_cols = out_df.columns[out_df.isna().any()].tolist()
        raise ValueError(f"Feature matrix contains NaN values in columns: {nan_cols}")

    return out_df
