"""Business and retail feature engineering module.

Computes company-level daily retail features, rolling demand indicators,
and product demand across the top 50 SKUs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_daily_retail_features(df_trans: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute daily company-level retail features and rolling velocity metrics."""
    logger.info("Computing daily company-level retail demand metrics...")
    df = df_trans.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.floor("D")

    min_date = df["date"].min()
    max_date = df["date"].max()
    all_dates = pd.date_range(min_date, max_date, freq="D", tz="UTC")

    # Exclude non-product and extreme quantity outliers from KPI calculations
    valid_mask = (~df["is_non_product"]) & (~df["is_extreme_quantity"])
    df_kpi = df[valid_mask].copy()

    # Sales lines: positive quantity and not cancellation
    sales_mask = (df_kpi["quantity"] > 0) & (~df_kpi["is_cancellation"])
    sales_df = df_kpi[sales_mask]

    # Returns lines: negative quantity
    returns_mask = df_kpi["quantity"] < 0
    returns_df = df_kpi[returns_mask]

    # 1. Daily sales units
    daily_sales = sales_df.groupby("date")["quantity"].sum().reindex(all_dates, fill_value=0)

    # 2. Daily returns units (absolute)
    daily_returns = (
        returns_df.groupby("date")["quantity"].sum().abs().reindex(all_dates, fill_value=0)
    )

    # 3. Revenue: sum of revenue_clean (net)
    daily_rev = df_kpi.groupby("date")["revenue_clean"].sum().reindex(all_dates, fill_value=0.0)

    # 4. Daily orders (distinct non-cancellation invoices)
    daily_orders = sales_df.groupby("date")["invoice_no"].nunique().reindex(all_dates, fill_value=0)

    # 5. Daily customers (distinct non-guest customers)
    cust_mask = sales_mask & (~df_kpi["is_guest"])
    daily_customers = (
        df_kpi[cust_mask].groupby("date")["customer_id"].nunique().reindex(all_dates, fill_value=0)
    )

    # 6. Average order value: gross revenue of sales lines / daily_orders
    gross_sales_rev = (
        sales_df.groupby("date")["revenue_clean"].sum().reindex(all_dates, fill_value=0.0)
    )
    aov = np.where(daily_orders > 0, gross_sales_rev / daily_orders, 0.0)

    res_df = pd.DataFrame(
        {
            "date": all_dates,
            "daily_sales": daily_sales.astype(np.int32).values,
            "daily_returns_units": daily_returns.astype(np.int32).values,
            "revenue": daily_rev.astype(np.float32).values,
            "daily_orders": daily_orders.astype(np.int32).values,
            "daily_customers": daily_customers.astype(np.int32).values,
            "average_order_value": aov.astype(np.float32),
        }
    )

    # 7. Rolling windows
    res_df["weekly_sales"] = (
        res_df["daily_sales"].rolling(7, min_periods=1).sum().astype(np.float32)
    )
    res_df["monthly_sales"] = (
        res_df["daily_sales"].rolling(28, min_periods=1).sum().astype(np.float32)
    )

    # 8. Sales growth: weekly_sales / weekly_sales(t-7) - 1.0
    prev_weekly = res_df["weekly_sales"].shift(7)
    res_df["sales_growth"] = np.where(
        (prev_weekly > 0) & prev_weekly.notna(),
        (res_df["weekly_sales"] / prev_weekly) - 1.0,
        0.0,
    ).astype(np.float32)

    days_no_sales = int((res_df["daily_sales"] == 0).sum())
    meta = {
        "date_range": (str(min_date.date()), str(max_date.date())),
        "total_days": len(res_df),
        "days_no_sales": days_no_sales,
    }
    logger.info(
        "Retail business features computed for %d days (%d zero-sales days)",
        len(res_df),
        days_no_sales,
    )
    return res_df, meta


def compute_product_demand_daily(df_trans: pd.DataFrame) -> pd.DataFrame:
    """Compute daily unit and revenue demand for the top 50 products."""
    logger.info("Computing daily demand for top 50 products...")
    df = df_trans.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.floor("D")

    # Filter out non-product lines and non-positive units
    valid_df = df[(~df["is_non_product"]) & (~df["is_extreme_quantity"]) & (df["quantity"] > 0)]

    # Determine top 50 products by total units
    top_50 = (
        valid_df.groupby("product_id")["quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(50)
        .index.tolist()
    )

    # Build description mapping (most frequent description per product)
    desc_map = (
        valid_df[valid_df["product_id"].isin(top_50)]
        .groupby("product_id")["description"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "")
        .to_dict()
    )

    min_date = df["date"].min()
    max_date = df["date"].max()
    all_dates = pd.date_range(min_date, max_date, freq="D", tz="UTC")

    # Cartesian product of all dates and top 50 products
    grid = (
        pd.MultiIndex.from_product([all_dates, top_50], names=["date", "product_id"])
        .to_frame()
        .reset_index(drop=True)
    )

    # Aggregate daily demand
    top_df = valid_df[valid_df["product_id"].isin(top_50)]
    agg = (
        top_df.groupby(["date", "product_id"])
        .agg(
            units=("quantity", "sum"),
            revenue=("revenue_clean", "sum"),
        )
        .reset_index()
    )

    demand_df = grid.merge(agg, on=["date", "product_id"], how="left")
    demand_df["units"] = demand_df["units"].fillna(0).astype(np.int32)
    demand_df["revenue"] = demand_df["revenue"].fillna(0.0).astype(np.float32)
    demand_df["description"] = demand_df["product_id"].map(desc_map).fillna("").astype("string")

    demand_df = demand_df[["date", "product_id", "description", "units", "revenue"]]
    logger.info("Product demand computed: %d product-day rows", len(demand_df))
    return demand_df
