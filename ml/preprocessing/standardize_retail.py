"""Online Retail standardization script for AI Operations Intelligence Platform.

Processes Online Retail data, applies whole-week time shift to align with
the operations timeline (2026-03-21 to 2026-10-19), cleans records, and outputs
data/raw/unified/business_transactions_raw.csv.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, date, timedelta
from typing import Any

import pandas as pd

from ml.preprocessing.config import RAW, TIMELINE_END, UNIFIED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ADMIN_STOCK_CODES = {
    "POST",
    "D",
    "M",
    "CRUK",
    "BANK CHARGES",
    "DOT",
    "AMAZONFEE",
    "S",
    "B",
    "PADS",
    "m",
}


def compute_retail_shift(max_invoice_date: date, target_end: date = TIMELINE_END) -> int:
    """Compute number of days to shift by whole weeks so retail covers target_end."""
    diff_days = (target_end - max_invoice_date).days
    shift_days = 7 * (diff_days // 7)
    if (max_invoice_date + timedelta(days=shift_days)) < target_end:
        shift_days += 7
    return shift_days


def load_and_standardize_retail() -> pd.DataFrame:
    """Load, clean, and time-shift Online Retail dataset."""
    parquet_path = RAW / "retail" / "online_retail.parquet"
    excel_path = RAW / "retail" / "Online Retail.xlsx"

    if parquet_path.exists():
        logger.info("Loading cached retail parquet from %s...", parquet_path)
        df = pd.read_parquet(parquet_path)
    elif excel_path.exists():
        logger.info("Reading retail excel from %s...", excel_path)
        df = pd.read_excel(excel_path)
    else:
        raise FileNotFoundError("Online Retail dataset not found in data/raw/retail/")

    logger.info("Raw retail rows: %d", len(df))

    # Clean string columns
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()
    df["Country"] = df["Country"].fillna("Unknown").astype(str).str.strip()

    # Drop non-positive UnitPrice (bad debt, adjustments, free gifts)
    df = df[df["UnitPrice"] > 0].copy()
    logger.info("Rows after UnitPrice > 0 filter: %d", len(df))

    # Filter out administrative stock codes
    df = df[~df["StockCode"].isin(ADMIN_STOCK_CODES)].copy()
    logger.info("Rows after admin code filter: %d", len(df))

    # Whole-week time shift
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    max_d = df["InvoiceDate"].dt.date.max()
    shift_days = compute_retail_shift(max_d, TIMELINE_END)
    logger.info("Shifting retail dates by %d days (%d whole weeks)...", shift_days, shift_days // 7)

    df["transaction_date"] = (df["InvoiceDate"] + pd.Timedelta(days=shift_days)).dt.tz_localize(
        UTC
    )

    # Identifiers and cancellations
    df["is_cancellation"] = df["InvoiceNo"].str.startswith("C")
    df["quantity"] = df["Quantity"].astype(int)
    df["unit_price"] = df["UnitPrice"].round(2)
    df["line_total"] = (df["quantity"] * df["unit_price"]).round(2)

    # Clean CustomerID (preserve nulls, format integers)
    def _clean_customer_id(val: Any) -> str:
        if pd.isna(val) or val == "" or str(val).lower() == "nan":
            return ""
        try:
            return str(int(float(val)))
        except (ValueError, TypeError):
            return str(val)

    df["customer_id"] = df["CustomerID"].apply(_clean_customer_id)

    # Renaming and formatting
    df["invoice_no"] = df["InvoiceNo"]
    df["stock_code"] = df["StockCode"]
    df["description"] = df["Description"]
    df["country"] = df["Country"]

    # Deterministic UUIDv5 transaction ID: uuid5 based on invoice, stock, and index
    logger.info("Generating deterministic transaction UUIDs...")
    df = df.reset_index(drop=True)
    # Fast vectorized-ish string join for UUID generation
    base_strings = (
        df["invoice_no"]
        + "-"
        + df["stock_code"]
        + "-"
        + df.index.astype(str)
    )
    dns_namespace = uuid.NAMESPACE_DNS
    df["transaction_id"] = [str(uuid.uuid5(dns_namespace, s)) for s in base_strings]

    out_cols = [
        "transaction_id",
        "invoice_no",
        "stock_code",
        "description",
        "quantity",
        "unit_price",
        "customer_id",
        "country",
        "transaction_date",
        "is_cancellation",
        "line_total",
    ]
    df_out = df[out_cols].sort_values("transaction_date").reset_index(drop=True)
    return df_out


def run() -> None:
    """Run standardization and write business_transactions_raw.csv."""
    UNIFIED.mkdir(parents=True, exist_ok=True)
    out_file = UNIFIED / "business_transactions_raw.csv"

    df = load_and_standardize_retail()
    logger.info("Writing %d standardized transactions to %s...", len(df), out_file)

    # Acceptance Checks
    # 1. Scale between 400,000 and 540,000 rows
    assert 400000 <= len(df) <= 540000, f"Unexpected row count: {len(df)}"

    # 2. Cancellations preserved
    cancels = df[df["is_cancellation"]]
    assert len(cancels) > 5000, f"Expected > 5000 cancellations, got {len(cancels)}"
    assert (cancels["quantity"] < 0).all(), "Cancellations should have negative quantity"

    # 3. All unit prices strictly positive
    assert (df["unit_price"] > 0).all(), "Found non-positive unit price"

    # 4. Covers the operations timeline
    t_min = pd.Timestamp("2026-03-21", tz="UTC")
    t_max = pd.Timestamp("2026-10-19", tz="UTC")
    assert df["transaction_date"].min() <= t_min, "Shifted min date does not precede timeline start"
    assert df["transaction_date"].max() >= t_max, "Shifted max date does not reach timeline end"

    # 5. Null customers preserved as empty strings
    empty_cust = (df["customer_id"] == "").sum()
    assert empty_cust > 100000, f"Expected > 100k unassigned customers, got {empty_cust}"

    df.to_csv(out_file, index=False)
    logger.info("Successfully created %s (%d rows)", out_file, len(df))


if __name__ == "__main__":
    run()
