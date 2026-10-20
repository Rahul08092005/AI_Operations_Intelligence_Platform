"""Daily Business Dataset Generator for AI Operations Intelligence Platform.

Generates data/raw/unified/business_daily_raw.csv (213 days)
integrating production, sensor energy, maintenance costs, and retail demand.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ml.preprocessing.config import (
    ENERGY_UNIT_RATE_GBP,
    SEED,
    TIMELINE_END,
    TIMELINE_START,
    UNIFIED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INITIAL_INVENTORY = 25000
AVERAGE_UNIT_PRICE_GBP = 14.50


def generate_business_daily() -> pd.DataFrame:
    """Generate daily business records aggregating production, sensors, and demand."""
    rng = np.random.Generator(np.random.PCG64(SEED))

    prod_path = UNIFIED / "production_raw.csv"
    sensors_path = UNIFIED / "sensors_raw.csv"
    maint_path = UNIFIED / "maintenance_raw.csv"
    trans_path = UNIFIED / "business_transactions_raw.csv"

    if not prod_path.exists():
        raise FileNotFoundError(f"Missing {prod_path}. Run generate_production.py first.")
    if not sensors_path.exists():
        raise FileNotFoundError(f"Missing {sensors_path}. Run generate_sensors.py first.")
    if not maint_path.exists():
        raise FileNotFoundError(f"Missing {maint_path}. Run generate_maintenance.py first.")
    if not trans_path.exists():
        raise FileNotFoundError(f"Missing {trans_path}. Run standardize_retail.py first.")

    logger.info("Reading production_raw.csv...")
    prod_df = pd.read_csv(prod_path)
    prod_df["date"] = pd.to_datetime(prod_df["date"]).dt.date

    # Aggregate production by date
    daily_prod = (
        prod_df.groupby("date")
        .agg(
            total_production_units=("units_produced", "sum"),
            total_scrap_units=("scrap_units", "sum"),
            total_operating_hours=("operating_hours", "sum"),
            total_downtime_hours=("downtime_hours", "sum"),
        )
        .reset_index()
    )

    logger.info("Aggregating energy consumption from sensors_raw.csv...")
    sensor_df = pd.read_csv(
        sensors_path,
        usecols=["timestamp", "energy_consumption"],
    )
    sensor_df["date"] = pd.to_datetime(sensor_df["timestamp"], utc=True).dt.date
    daily_energy = (
        sensor_df.groupby("date")["energy_consumption"].sum().round(2).to_dict()
    )

    logger.info("Aggregating maintenance costs from maintenance_raw.csv...")
    maint_df = pd.read_csv(maint_path, usecols=["performed_date", "cost_gbp"])
    maint_df["date"] = pd.to_datetime(maint_df["performed_date"], utc=True).dt.date
    daily_maint = maint_df.groupby("date")["cost_gbp"].sum().round(2).to_dict()

    logger.info("Extracting retail demand signal from business_transactions_raw.csv...")
    trans_df = pd.read_csv(
        trans_path,
        usecols=["transaction_date", "quantity"],
    )
    trans_df["date"] = pd.to_datetime(trans_df["transaction_date"], utc=True).dt.date
    valid_trans = trans_df[trans_df["quantity"] > 0]
    daily_trans_qty = valid_trans.groupby("date")["quantity"].sum().to_dict()

    # Calculate mean retail demand for scaling index
    all_dates = [d.date() for d in pd.date_range(TIMELINE_START, TIMELINE_END, freq="D")]
    timeline_retail_qtys = [
        daily_trans_qty.get(d, 0) for d in all_dates if daily_trans_qty.get(d, 0) > 0
    ]
    mean_retail_qty = float(np.mean(timeline_retail_qtys)) if timeline_retail_qtys else 15000.0

    # Fleet nominal good production capacity
    mean_fleet_output = float(
        (daily_prod["total_production_units"] - daily_prod["total_scrap_units"]).mean()
    )

    records: list[dict[str, Any]] = []
    current_inventory = INITIAL_INVENTORY

    for _, row in daily_prod.iterrows():
        cur_date = row["date"]
        prod_units = int(row["total_production_units"])
        scrap_units = int(row["total_scrap_units"])
        good_units = prod_units - scrap_units
        op_h = round(float(row["total_operating_hours"]), 2)
        down_h = round(float(row["total_downtime_hours"]), 2)

        # Energy & Maintenance
        kwh = float(daily_energy.get(cur_date, 0.0))
        energy_cost = round(kwh * ENERGY_UNIT_RATE_GBP, 2)
        maint_cost = float(daily_maint.get(cur_date, 0.0))

        # Demand modeling grounded in retail transactions
        raw_qty = daily_trans_qty.get(cur_date, 0)
        if raw_qty > 0:
            demand_index = raw_qty / mean_retail_qty
        else:
            # Weekend / store closure day: steady lower-bound distribution
            demand_index = float(rng.uniform(0.35, 0.55))

        daily_demand = int(round(mean_fleet_output * demand_index))
        # Ensure positive demand
        daily_demand = max(1000, daily_demand)

        # Inventory Stock-Flow Balance
        available = current_inventory + good_units
        if available >= daily_demand:
            shipped_units = daily_demand
            backorder = 0
            current_inventory = available - shipped_units
        else:
            shipped_units = available
            backorder = daily_demand - available
            current_inventory = 0

        revenue = round(shipped_units * AVERAGE_UNIT_PRICE_GBP, 2)

        created_at = datetime(
            cur_date.year, cur_date.month, cur_date.day, 23, 59, tzinfo=UTC
        ).isoformat()

        records.append(
            {
                "date": cur_date.isoformat(),
                "plant_id": "PLANT-GLOBAL",
                "total_production_units": prod_units,
                "total_scrap_units": scrap_units,
                "total_operating_hours": op_h,
                "total_downtime_hours": down_h,
                "total_energy_kwh": kwh,
                "energy_cost_gbp": energy_cost,
                "maintenance_cost_gbp": maint_cost,
                "revenue_gbp": revenue,
                "inventory_level": current_inventory,
                "backorder_units": backorder,
                "created_at": created_at,
            }
        )

    df_out = pd.DataFrame(records)
    df_out = df_out.sort_values("date").reset_index(drop=True)
    return df_out


def run() -> None:
    """Generate and validate business_daily_raw.csv."""
    UNIFIED.mkdir(parents=True, exist_ok=True)
    out_file = UNIFIED / "business_daily_raw.csv"

    df = generate_business_daily()
    logger.info("Generated %d daily business records.", len(df))

    # Acceptance Checks
    # 1. Exactly 213 rows
    assert len(df) == 213, f"Expected 213 rows, got {len(df)}"

    # 2. Date range matches timeline
    dates = pd.to_datetime(df["date"]).dt.date
    assert dates.min() == TIMELINE_START, f"Min date mismatch: {dates.min()}"
    assert dates.max() == TIMELINE_END, f"Max date mismatch: {dates.max()}"

    # 3. Inventory and backorders are strictly non-negative
    assert (df["inventory_level"] >= 0).all(), "Negative inventory found"
    assert (df["backorder_units"] >= 0).all(), "Negative backorders found"

    # 4. Production, energy, and maintenance costs non-negative
    assert (df["total_production_units"] >= 0).all(), "Negative production"
    assert (df["total_energy_kwh"] > 0).all(), "Energy should be positive"
    assert (df["energy_cost_gbp"] > 0).all(), "Energy cost should be positive"
    assert (df["maintenance_cost_gbp"] >= 0).all(), "Negative maintenance cost"
    assert (df["revenue_gbp"] > 0).all(), "Revenue should be positive"

    # 5. Plant ID
    assert (df["plant_id"] == "PLANT-GLOBAL").all(), "Invalid plant_id"

    df.to_csv(out_file, index=False)
    logger.info("Successfully wrote %d daily business records to %s", len(df), out_file)


if __name__ == "__main__":
    run()
