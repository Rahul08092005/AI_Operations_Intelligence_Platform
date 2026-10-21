"""Operations dataset integration and daily aggregation module.

Integrates machine-level telemetry aggregates, fleet production, business KPIs,
and retail demand signals into `operations_dataset.parquet` and `.csv`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.preprocessing.config import PROCESSED

logger = logging.getLogger(__name__)

FORBIDDEN_COLUMNS = {
    "status",
    "scenario",
    "data_origin",
    "ai4i_type",
    "in_failure_window",
    "episode_type",
    "episode_id",
    "failure_window_id",
}


def compute_daily_sensor_aggregates(df_sensor_features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5-minute sensor features to daily machine-level resolution."""
    logger.info("Computing daily sensor aggregates from sensor features...")
    df = df_sensor_features.copy()
    df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("D")

    # Only include complete days (288 intervals) per machine (D40)
    day_counts = df.groupby(["machine_id", "date"], observed=True).size()
    complete_days = day_counts[day_counts == 288].index
    df_complete = df.set_index(["machine_id", "date"]).loc[complete_days].reset_index()

    agg_rows: list[dict[str, Any]] = []

    for (m_id, d), grp in df_complete.groupby(["machine_id", "date"], observed=True):
        last_interval = grp.iloc[-1]

        # Valid signal coverage (intervals where at least 1 primary signal is non-null)
        valid_signals = grp[["temperature", "vibration", "pressure", "rpm"]].notna().any(axis=1)
        coverage = float(valid_signals.mean())

        imputed_share = float(grp["is_imputed"].mean()) if "is_imputed" in grp else 0.0
        outlier_share = float(grp["is_outlier_flag"].mean()) if "is_outlier_flag" in grp else 0.0

        # Minutes above limits (count of intervals * 5 min)
        vib_warn_min = float(grp["vibration_above_warning"].fillna(0).sum() * 5.0)
        vib_crit_min = float(grp["vibration_above_critical"].fillna(0).sum() * 5.0)
        temp_spec_min = float(grp["temperature_above_spec"].fillna(0).sum() * 5.0)

        # End of day operating hours
        op_h_valid = grp["operating_hours"].dropna()
        op_h_end = float(op_h_valid.iloc[-1]) if len(op_h_valid) > 0 else np.nan

        # Running hours
        running_h = float(grp["running"].sum() * (5.0 / 60.0))

        row: dict[str, Any] = {
            "machine_id": str(m_id),
            "date": d,
            "temperature_mean": float(grp["temperature"].mean()),
            "temperature_max": float(grp["temperature"].max()),
            "temperature_std": float(grp["temperature"].std()),
            "vibration_mean": float(grp["vibration"].mean()),
            "vibration_max": float(grp["vibration"].max()),
            "vibration_std": float(grp["vibration"].std()),
            "pressure_mean": float(grp["pressure"].mean()),
            "pressure_max": float(grp["pressure"].max()),
            "pressure_std": float(grp["pressure"].std()),
            "rpm_mean": float(grp["rpm"].mean()),
            "rpm_max": float(grp["rpm"].max()),
            "rpm_std": float(grp["rpm"].std()),
            "energy_kwh": float(grp["energy_consumption"].sum()),
            "motor_current_mean": float(grp["motor_current"].mean()),
            "operating_hours_end": op_h_end,
            "running_hours": running_h,
            "vibration_above_warning_minutes": vib_warn_min,
            "vibration_above_critical_minutes": vib_crit_min,
            "temperature_above_spec_minutes": temp_spec_min,
            "sensor_coverage": coverage,
            "imputed_share": imputed_share,
            "outlier_flag_share": outlier_share,
            "vibration_roll_mean_24h_end": float(
                last_interval.get("vibration_roll_mean_24h", np.nan)
            ),
            "temperature_roll_mean_24h_end": float(
                last_interval.get("temperature_roll_mean_24h", np.nan)
            ),
            "vibration_slope_24h_end": float(last_interval.get("vibration_slope_24h", np.nan)),
        }
        agg_rows.append(row)

    df_agg = pd.DataFrame(agg_rows)
    df_agg["machine_id"] = df_agg["machine_id"].astype("category")
    logger.info("Computed daily sensor aggregates for %d machine-day rows", len(df_agg))
    return df_agg


def build_operations_dataset(
    df_machines: pd.DataFrame,
    df_production: pd.DataFrame,
    df_business_daily: pd.DataFrame,
    df_sensor_aggregates: pd.DataFrame,
    df_sensor_features: pd.DataFrame,
    df_maintenance: pd.DataFrame,
    df_retail_daily: pd.DataFrame,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble the unified daily operations dataset."""
    logger.info("Integrating daily operations dataset...")
    out_dir = output_dir or PROCESSED

    # 1. Spine: complete machine-days present in daily sensor aggregates (D40)
    spine = df_sensor_aggregates[["machine_id", "date"]].copy()
    spine["date"] = pd.to_datetime(spine["date"], utc=True).dt.floor("D")

    # 2. Machines static metadata (strictly excluding forbidden columns)
    safe_m_cols = ["machine_id", "machine_type", "location", "installation_date"]
    m_clean = df_machines[[c for c in safe_m_cols if c in df_machines.columns]].copy()
    m_clean["installation_date"] = pd.to_datetime(m_clean["installation_date"], utc=True).dt.floor(
        "D"
    )

    ops = spine.merge(m_clean, on="machine_id", how="left")
    ops["machine_age_days"] = (ops["date"] - ops["installation_date"]).dt.days.astype(np.int32)

    # 3. Production
    prod_clean = df_production.copy()
    prod_clean["date"] = pd.to_datetime(prod_clean["date"], utc=True).dt.floor("D")
    if "units_produced" in prod_clean.columns and "production_quantity" not in prod_clean.columns:
        prod_clean["production_quantity"] = prod_clean["units_produced"]
    if "units_target" in prod_clean.columns and "target_quantity" not in prod_clean.columns:
        prod_clean["target_quantity"] = prod_clean["units_target"]
    if "downtime_hours" in prod_clean.columns and "downtime_minutes" not in prod_clean.columns:
        prod_clean["downtime_minutes"] = prod_clean["downtime_hours"] * 60.0

    prod_cols = ["machine_id", "date", "production_quantity", "target_quantity", "downtime_minutes"]
    ops = ops.merge(
        prod_clean[[c for c in prod_cols if c in prod_clean.columns]],
        on=["machine_id", "date"],
        how="left",
    )

    # 4. Business daily
    b_clean = df_business_daily.copy()
    b_clean["date"] = pd.to_datetime(b_clean["date"], utc=True).dt.floor("D")
    if "inventory_level" in b_clean.columns and "inventory" not in b_clean.columns:
        b_clean["inventory"] = b_clean["inventory_level"]
    if "maintenance_cost_gbp" in b_clean.columns and "maintenance_cost_day" not in b_clean.columns:
        b_clean["maintenance_cost_day"] = b_clean["maintenance_cost_gbp"]

    # Align orders from retail if not present in business_daily
    if "orders" not in b_clean.columns and "daily_orders" in df_retail_daily.columns:
        b_clean = b_clean.merge(df_retail_daily[["date", "daily_orders"]], on="date", how="left")
        b_clean["orders"] = b_clean["daily_orders"].fillna(0).astype(np.int32)

    b_cols = ["date", "orders", "inventory", "maintenance_cost_day"]
    ops = ops.merge(b_clean[[c for c in b_cols if c in b_clean.columns]], on="date", how="left")

    # 5. Daily sensor aggregates
    ops = ops.merge(df_sensor_aggregates, on=["machine_id", "date"], how="left")

    # 6. Maintenance context features as of start of day (extracted from sensor_features)
    maint_feature_cols = [
        "days_since_maintenance",
        "days_since_inspection",
        "days_since_preventive",
        "operating_hours_since_inspection",
        "operating_hours_since_preventive",
        "hours_since_estimated",
        "inspection_overdue",
        "preventive_overdue",
        "maintenance_count",
        "maintenance_count_90d",
        "maintenance_cost_avg",
        "maintenance_cost_90d",
        "failure_count",
        "days_since_failure",
        "previous_failure",
    ]

    sf_day = df_sensor_features.copy()
    sf_day["date"] = pd.to_datetime(sf_day["timestamp"], utc=True).dt.floor("D")
    # Take first row of the day as start-of-day features
    cols_to_take = ["machine_id", "date"] + [c for c in maint_feature_cols if c in sf_day.columns]
    maint_context = sf_day[cols_to_take].drop_duplicates(
        subset=["machine_id", "date"], keep="first"
    )
    ops = ops.merge(maint_context, on=["machine_id", "date"], how="left")

    # 7. Same-day maintenance event context
    maint_ev = df_maintenance.copy()
    maint_ev["date"] = pd.to_datetime(maint_ev["performed_date"], utc=True).dt.floor("D")
    maint_today = (
        maint_ev.groupby(["machine_id", "date"], observed=True)
        .agg(
            maintenance_event_today=("maintenance_id", lambda x: 1),
            maintenance_type_today=("maintenance_type", lambda x: str(x.iloc[0]).capitalize()),
        )
        .reset_index()
    )
    ops = ops.merge(maint_today, on=["machine_id", "date"], how="left")
    ops["maintenance_event_today"] = ops["maintenance_event_today"].fillna(0).astype(np.int8)
    ops["maintenance_type_today"] = ops["maintenance_type_today"].fillna("None").astype("string")

    # 8. Retail context prefixed with retail_
    ret_df = df_retail_daily.copy()
    ret_df["date"] = pd.to_datetime(ret_df["date"], utc=True).dt.floor("D")
    ret_rename = {
        "daily_sales": "retail_daily_sales",
        "revenue": "retail_revenue",
        "daily_orders": "retail_daily_orders",
        "average_order_value": "retail_average_order_value",
        "weekly_sales": "retail_weekly_sales",
        "monthly_sales": "retail_monthly_sales",
        "sales_growth": "retail_sales_growth",
    }
    ret_df = ret_df.rename(columns=ret_rename)
    ret_cols = ["date"] + list(ret_rename.values())
    ops = ops.merge(ret_df[[c for c in ret_cols if c in ret_df.columns]], on="date", how="left")

    # 9d. Derived daily features
    ops["efficiency"] = (ops["production_quantity"] / ops["target_quantity"]).astype(np.float32)
    ops["downtime_hours"] = (ops["downtime_minutes"] / 60.0).astype(np.float32)

    zero_prod = ops["production_quantity"] <= 0
    ops["energy_per_unit"] = np.where(
        zero_prod, np.nan, ops["energy_kwh"] / ops["production_quantity"]
    ).astype(np.float32)

    # Machine-level lags and rolling averages (strictly past)
    ops = ops.sort_values(by=["machine_id", "date"]).reset_index(drop=True)

    ops["production_lag_1"] = (
        ops.groupby("machine_id", observed=True)["production_quantity"].shift(1).astype(np.float32)
    )
    ops["production_lag_7"] = (
        ops.groupby("machine_id", observed=True)["production_quantity"].shift(7).astype(np.float32)
    )

    ops["production_roll_mean_7d"] = (
        ops.groupby("machine_id", observed=True)["production_quantity"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        .astype(np.float32)
    )

    ops["downtime_roll_mean_7d"] = (
        ops.groupby("machine_id", observed=True)["downtime_minutes"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        .astype(np.float32)
    )

    if "retail_daily_orders" in ops.columns:
        ops["orders_lag_1"] = (
            ops.groupby("machine_id", observed=True)["retail_daily_orders"]
            .shift(1)
            .astype(np.float32)
        )
    else:
        ops["orders_lag_1"] = np.nan

    ops["day_of_week"] = ops["date"].dt.dayofweek.astype(np.int8)
    ops["is_weekend"] = ops["day_of_week"].isin([5, 6]).astype(np.int8)
    ops["month"] = ops["date"].dt.month.astype(np.int8)

    # Ensure no forbidden columns leaked
    leaked = set(ops.columns).intersection(FORBIDDEN_COLUMNS)
    if leaked:
        raise ValueError(f"CRITICAL: Forbidden columns leaked into operations_dataset: {leaked}")

    # Write files
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "operations_dataset.parquet"
    csv_path = out_dir / "operations_dataset.csv"

    ops.to_parquet(parquet_path, index=False)
    ops.to_csv(csv_path, index=False, date_format="%Y-%m-%d")

    logger.info(
        "Saved operations dataset to %s and %s (%d rows, %d cols)",
        parquet_path.name,
        csv_path.name,
        len(ops),
        len(ops.columns),
    )

    meta = {
        "rows": len(ops),
        "columns": len(ops.columns),
        "machines": int(ops["machine_id"].nunique()),
        "dates_count": int(ops["date"].nunique()),
    }
    return ops, meta
