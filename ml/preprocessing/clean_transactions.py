"""Retail transaction cleaning and audit flagging module.

Implements policy D35:
- Rejects non-positive prices, duplicates, and missing required keys into a separate table.
- Flags returns, cancellations, guest checkouts, non-product codes, and extreme quantities.
- Computes `revenue_clean` for safe KPI aggregation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ml.preprocessing.config import RETAIL_EXTREME_QTY
from ml.preprocessing.validation_rules import NON_PRODUCT_CODE_REGEX

logger = logging.getLogger(__name__)


@dataclass
class TransactionCleaningLog:
    """Summary metrics of transaction cleaning and flagging."""

    raw_rows: int = 0
    kept_rows: int = 0
    rejected_rows: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    cancellations: int = 0
    returns: int = 0
    guests: int = 0
    non_product_lines: int = 0
    non_product_codes: list[str] = field(default_factory=list)
    extreme_quantities: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_transactions(
    df_trans: pd.DataFrame, extreme_qty_threshold: int = RETAIL_EXTREME_QTY
) -> tuple[pd.DataFrame, pd.DataFrame, TransactionCleaningLog]:
    """Clean and flag retail transactions into kept and rejected tables."""
    raw_count = len(df_trans)
    logger.info("Cleaning %d retail transactions...", raw_count)
    log = TransactionCleaningLog(raw_rows=raw_count)

    df = df_trans.copy()

    # Step 1: Identify rejected rows
    ts_col = "transaction_date" if "transaction_date" in df.columns else "timestamp"
    missing_required = df[ts_col].isna() | df["quantity"].isna()
    non_pos_price = df["unit_price"] <= 0
    dup_id = df.duplicated(subset=["transaction_id"], keep="first")

    reject_mask = missing_required | non_pos_price | dup_id
    rejected_df = df[reject_mask].copy()

    # Assign reject_reason in priority order
    if len(rejected_df) > 0:
        reasons = pd.Series(index=rejected_df.index, dtype="string")
        reasons[missing_required.loc[rejected_df.index]] = "missing_required"
        reasons[non_pos_price.loc[rejected_df.index]] = "non_positive_price"
        reasons[dup_id.loc[rejected_df.index]] = "duplicate_id"
        rejected_df["reject_reason"] = reasons
        log.reject_reasons = rejected_df["reject_reason"].value_counts().to_dict()
    else:
        rejected_df["reject_reason"] = pd.Series(dtype="string")
        log.reject_reasons = {}

    log.rejected_rows = len(rejected_df)

    # Step 2: Keep remaining rows
    kept_df = df[~reject_mask].copy()
    log.kept_rows = len(kept_df)

    # Step 3: Audit flags
    kept_df["is_cancellation"] = kept_df["invoice_no"].astype(str).str.startswith("C")
    kept_df["is_return"] = kept_df["quantity"] < 0
    kept_df["is_guest"] = (
        kept_df["customer_id"].isna()
        | (kept_df["customer_id"].astype(str).str.strip() == "")
        | (kept_df["customer_id"].astype(str).str.strip().str.lower() == "nan")
    )

    stock_col = "stock_code" if "stock_code" in kept_df.columns else "product_id"
    code_pattern = re.compile(NON_PRODUCT_CODE_REGEX)
    codes_str = kept_df[stock_col].astype(str)
    kept_df["is_non_product"] = ~codes_str.str.match(code_pattern)

    kept_df["is_extreme_quantity"] = kept_df["quantity"].abs() >= extreme_qty_threshold

    # Populate log metrics
    log.cancellations = int(kept_df["is_cancellation"].sum())
    log.returns = int(kept_df["is_return"].sum())
    log.guests = int(kept_df["is_guest"].sum())
    log.non_product_lines = int(kept_df["is_non_product"].sum())
    log.non_product_codes = sorted(
        kept_df.loc[kept_df["is_non_product"], stock_col].unique().tolist()
    )
    log.extreme_quantities = int(kept_df["is_extreme_quantity"].sum())

    # Step 4: Derived columns
    kept_df["date"] = pd.to_datetime(kept_df[ts_col], utc=True).dt.floor("D")

    # Ensure product_id exists
    if "product_id" not in kept_df.columns:
        kept_df["product_id"] = kept_df[stock_col]

    rev_source = "line_total" if "line_total" in kept_df.columns else "revenue"
    if "revenue" not in kept_df.columns:
        kept_df["revenue"] = kept_df[rev_source]

    # revenue_clean: zeroed out for non-product and extreme quantity outliers
    safe_kpi_mask = (~kept_df["is_non_product"]) & (~kept_df["is_extreme_quantity"])
    kept_df["revenue_clean"] = np.where(safe_kpi_mask, kept_df["revenue"], 0.0).astype("float32")

    logger.info(
        "Transactions cleaned: %d kept, %d rejected (non_product=%d, extreme=%d, cancellations=%d)",
        log.kept_rows,
        log.rejected_rows,
        log.non_product_lines,
        log.extreme_quantities,
        log.cancellations,
    )

    return kept_df, rejected_df, log
