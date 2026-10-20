"""Profile raw data sources and generate the data dictionary for Day 2.

Datasets profiled:
1. MetroPT-3 Air Compressor dataset
2. AI4I 2020 Predictive Maintenance dataset
3. UCI Online Retail dataset
4. M5 Forecasting dataset (if available)

Outputs docs/data_dictionary.md with all computed statistics.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DOCS = ROOT / "docs"


def profile_metropt(file_path: Path) -> dict:
    """Profile MetroPT-3 using chunked reading to limit memory consumption."""
    logger.info("Profiling MetroPT-3 from %s...", file_path)

    float_cols = [
        "TP2",
        "TP3",
        "H1",
        "DV_pressure",
        "Reservoirs",
        "Oil_temperature",
        "Motor_current",
        "COMP",
        "DV_eletric",
        "Towers",
        "MPG",
        "LPS",
        "Pressure_switch",
        "Oil_level",
        "Caudal_impulses",
    ]
    dtypes = {c: "float32" for c in float_cols}
    if "Unnamed: 0" in dtypes:
        dtypes["Unnamed: 0"] = "int32"

    total_rows = 0
    missing_counts = {c: 0 for c in ["timestamp"] + float_cols}
    min_vals = {c: float("inf") for c in float_cols}
    max_vals = {c: float("-inf") for c in float_cols}
    sum_vals = {c: 0.0 for c in float_cols}

    first_timestamp = None
    last_timestamp = None
    prev_ts = None
    delta_counts: dict[int, int] = {}
    gaps: list[dict] = []
    motor_current_samples: list[float] = []

    chunksize = 150_000
    for chunk_idx, chunk in enumerate(
        pd.read_csv(file_path, chunksize=chunksize, dtype=dtypes, low_memory=False)
    ):
        total_rows += len(chunk)
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])

        if first_timestamp is None:
            first_timestamp = chunk["timestamp"].iloc[0]

        # Sample motor_current for histogram & thresholding (1 in 10 rows)
        mc_sample = chunk["Motor_current"].iloc[::10].dropna().tolist()
        motor_current_samples.extend(mc_sample)

        # Track deltas and gaps
        ts_series = chunk["timestamp"]
        if prev_ts is not None:
            first_delta = (ts_series.iloc[0] - prev_ts).total_seconds()
            if first_delta > 300:  # Gap > 5 minutes
                gaps.append(
                    {
                        "start": str(prev_ts),
                        "end": str(ts_series.iloc[0]),
                        "duration_sec": first_delta,
                    }
                )

        diffs = ts_series.diff().dt.total_seconds().dropna()
        for d in diffs[diffs > 300]:
            gaps.append({"duration_sec": d})

        # Sample rounded deltas for median sampling interval calculation
        for d in diffs.iloc[:1000].round().astype(int):
            delta_counts[d] = delta_counts.get(d, 0) + 1

        prev_ts = ts_series.iloc[-1]
        last_timestamp = ts_series.iloc[-1]

        # Accumulate column stats
        for c in float_cols:
            if c in chunk.columns:
                series = chunk[c].dropna()
                missing_counts[c] += int(chunk[c].isna().sum())
                if len(series) > 0:
                    min_vals[c] = min(min_vals[c], float(series.min()))
                    max_vals[c] = max(max_vals[c], float(series.max()))
                    sum_vals[c] += float(series.sum())

        logger.info("Processed chunk %d (%d rows accumulated)", chunk_idx + 1, total_rows)

    # Calculate overall means
    means = {c: sum_vals[c] / (total_rows - missing_counts[c]) for c in float_cols}

    # Determine motor current threshold (I_OFF)
    mc_arr = np.array(motor_current_samples, dtype=np.float32)
    counts, bin_edges = np.histogram(mc_arr[mc_arr < 4.0], bins=40, range=(0.0, 4.0))
    valid_bins = [
        (i, (bin_edges[i] + bin_edges[i + 1]) / 2, counts[i])
        for i in range(len(counts))
        if 0.5 <= (bin_edges[i] + bin_edges[i + 1]) / 2 <= 3.0
    ]
    if valid_bins:
        min_bin = min(valid_bins, key=lambda x: x[2])
        i_off = round(float(min_bin[1]), 2)
    else:
        i_off = 1.5

    # Fraction running
    fraction_running = float((mc_arr > i_off).mean())

    # Modal sampling interval
    sorted_deltas = sorted(delta_counts.items(), key=lambda x: x[1], reverse=True)
    modal_interval = sorted_deltas[0][0] if sorted_deltas else 10

    return {
        "file": file_path.name,
        "row_count": total_rows,
        "start_time": str(first_timestamp),
        "end_time": str(last_timestamp),
        "modal_sampling_interval_sec": modal_interval,
        "num_gaps_gt_5min": len(gaps),
        "max_gap_seconds": max([g["duration_sec"] for g in gaps]) if gaps else 0,
        "I_OFF": i_off,
        "fraction_running": fraction_running,
        "columns": {
            c: {
                "dtype": "float32",
                "min": round(min_vals[c], 4),
                "max": round(max_vals[c], 4),
                "mean": round(means[c], 4),
                "missing": missing_counts[c],
            }
            for c in float_cols
        },
    }


def profile_ai4i(file_path: Path) -> dict:
    """Profile AI4I 2020 Predictive Maintenance dataset."""
    logger.info("Profiling AI4I 2020 from %s...", file_path)
    df = pd.read_csv(file_path)

    total_failures = int(df["Machine failure"].sum())
    failure_pct = float(total_failures / len(df) * 100)

    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    mode_counts = {m: int(df[m].sum()) for m in failure_modes}

    # Rows with multiple failure modes
    df["failure_mode_sum"] = df[failure_modes].sum(axis=1)
    multi_failure_rows = int((df["failure_mode_sum"] > 1).sum())

    type_counts = df["Type"].value_counts().to_dict()

    col_stats = {}
    for c in df.columns:
        stat = {
            "dtype": str(df[c].dtype),
            "missing": int(df[c].isna().sum()),
        }
        if pd.api.types.is_numeric_dtype(df[c]):
            stat["min"] = round(float(df[c].min()), 2)
            stat["max"] = round(float(df[c].max()), 2)
            stat["mean"] = round(float(df[c].mean()), 2)
        col_stats[c] = stat

    return {
        "file": file_path.name,
        "row_count": len(df),
        "total_failures": total_failures,
        "failure_pct": round(failure_pct, 2),
        "failure_mode_counts": mode_counts,
        "multi_failure_rows": multi_failure_rows,
        "type_counts": type_counts,
        "columns": col_stats,
    }


def profile_online_retail(file_path: Path) -> dict:
    """Profile UCI Online Retail dataset, caching Excel to Parquet for speed."""
    parquet_path = file_path.parent / "online_retail.parquet"
    if parquet_path.exists():
        logger.info("Reading cached Online Retail Parquet from %s...", parquet_path)
        df = pd.read_parquet(parquet_path)
    else:
        logger.info("Reading Online Retail Excel from %s (this may take ~20s)...", file_path)
        df = pd.read_excel(file_path)
        missing_customer = int(df["CustomerID"].isna().sum())
        missing_description = int(df["Description"].isna().sum())
        # Ensure string columns are uniform strings for pyarrow conversion
        df["InvoiceNo"] = df["InvoiceNo"].astype(str)
        df["StockCode"] = df["StockCode"].astype(str)
        df["Description"] = df["Description"].fillna("").astype(str)
        df["CustomerID"] = df["CustomerID"].astype(str).replace("nan", "")
        logger.info("Caching Online Retail to %s...", parquet_path)
        df.to_parquet(parquet_path, index=False)

    df["revenue"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceNo_str"] = df["InvoiceNo"].astype(str)

    cancellations = int(df["InvoiceNo_str"].str.startswith("C").sum())
    neg_quantity = int((df["Quantity"] < 0).sum())
    missing_customer = int((df["CustomerID"] == "").sum() + df["CustomerID"].isna().sum())
    missing_description = int((df["Description"] == "").sum() + df["Description"].isna().sum())
    zero_or_neg_price = int((df["UnitPrice"] <= 0).sum())
    num_invoices = int(df["InvoiceNo_str"].nunique())

    # Top 5 countries by revenue
    country_revenue = (
        df[df["revenue"] > 0]
        .groupby("Country")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    )

    col_stats = {}
    for c in [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    ]:
        missing_cnt = (
            missing_customer
            if c == "CustomerID"
            else (missing_description if c == "Description" else int(df[c].isna().sum()))
        )
        stat = {
            "dtype": str(df[c].dtype),
            "missing": missing_cnt,
        }
        if pd.api.types.is_numeric_dtype(df[c]):
            stat["min"] = round(float(df[c].min()), 2)
            stat["max"] = round(float(df[c].max()), 2)
            stat["mean"] = round(float(df[c].mean()), 2)
        col_stats[c] = stat

    return {
        "file": file_path.name,
        "row_count": len(df),
        "date_min": str(df["InvoiceDate"].min()),
        "date_max": str(df["InvoiceDate"].max()),
        "unique_invoices": num_invoices,
        "cancellations": cancellations,
        "negative_quantity_rows": neg_quantity,
        "missing_customer_id_rows": missing_customer,
        "zero_or_negative_price_rows": zero_or_neg_price,
        "top_5_countries_by_revenue": {k: round(v, 2) for k, v in country_revenue.items()},
        "columns": col_stats,
    }


def generate_data_dictionary_md(metro: dict, ai4i: dict, retail: dict, output_path: Path) -> None:
    """Generate docs/data_dictionary.md with all computed profile values."""
    logger.info("Writing data dictionary to %s...", output_path)

    lines: list[str] = [
        "# Data Dictionary: Source Datasets and Unified Internal Schema",
        "",
        "**Project:** AI Operations Intelligence Platform  ",
        f"**Generated Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
        "**Author:** AI Operations Intelligence Platform (Automated Profiler)",
        "",
        "---",
        "",
        "## 1. MetroPT-3 (Air Compressor Dataset)",
        "",
        f"- **Source File:** `data/raw/metropt/{metro['file']}`",
        "- **Documentation:** `data/raw/metropt/Data Description_Metro.pdf`",
        f"- **Total Rows:** {metro['row_count']:,}",
        f"- **Time Span:** {metro['start_time']} to {metro['end_time']}",
        f"- **Modal Sampling Interval:** ~{metro['modal_sampling_interval_sec']} seconds (0.1 Hz)",
        f"- **Data Gaps (> 5 min):** {metro['num_gaps_gt_5min']} gaps "
        f"(Maximum gap: {metro['max_gap_seconds']:,} seconds)",
        f"- **Motor Current Threshold (`I_OFF`):** **{metro['I_OFF']} A** "
        "(separating inactive/off states from operational/running states)",
        f"- **Operating Fraction:** {metro['fraction_running'] * 100:.1f}% running time",
        "- **`COMP` Signal Definition (from documentation):** Active (1.0) when air intake "
        "valve is closed (compressor off or offloaded/unloaded). Inactive (0.0) when "
        "compressor is loading and compressing air into reservoirs.",
        "",
        "### MetroPT Columns & Statistics",
        "",
        "| Column | Type | Unit | Min | Max | Mean | Missing | Description / Notes |",
        "|---|---|---|---|---|---|---|---|",
        f"| `timestamp` | datetime | ISO 8601 | {metro['start_time']} | {metro['end_time']} "
        "| - | 0 | Recorded sensor timestamp (10s intervals) |",
    ]

    desc_map = {
        "TP2": "Pressure on compressor (bar)",
        "TP3": "Pressure generated at pneumatic panel (bar)",
        "H1": "Cyclonic separator pressure drop (bar)",
        "DV_pressure": "Air dryers discharge pressure (bar, 0 = under load)",
        "Reservoirs": "Reservoir downstream pressure (bar)",
        "Oil_temperature": "Compressor oil temperature (°C)",
        "Motor_current": "Phase current (A; ~0A off, ~4A offload, ~7A load)",
        "COMP": "Air intake valve (1 = off/offload, 0 = load)",
        "DV_eletric": "Electrical outlet valve signal (0/1)",
        "Towers": "Dryer tower in operation (0 = T1, 1 = T2)",
        "MPG": "Intake valve start signal (< 8.2 bar)",
        "LPS": "Low pressure signal (< 7.0 bar)",
        "Pressure_switch": "Air drying tower discharge detection (0/1)",
        "Oil_level": "Low oil level indicator (0/1)",
        "Caudal_impulses": "Air flow impulse counter (count)",
    }

    unit_map = {
        "TP2": "bar",
        "TP3": "bar",
        "H1": "bar",
        "DV_pressure": "bar",
        "Reservoirs": "bar",
        "Oil_temperature": "°C",
        "Motor_current": "A",
        "COMP": "0/1",
        "DV_eletric": "0/1",
        "Towers": "0/1",
        "MPG": "0/1",
        "LPS": "0/1",
        "Pressure_switch": "0/1",
        "Oil_level": "0/1",
        "Caudal_impulses": "count",
    }

    for col, st in metro["columns"].items():
        desc = desc_map.get(col, "Sensor measurement")
        unit = unit_map.get(col, "")
        lines.append(
            f"| `{col}` | {st['dtype']} | {unit} | {st['min']} | {st['max']} | "
            f"{st['mean']} | {st['missing']} | {desc} |"
        )

    lines.extend(
        [
            "",
            "### Known Quirks",
            "- The raw CSV starts with an unnamed index column (`Unnamed: 0`).",
            "- Several sampling intervals deviate from 10s (ranging from 1s to ~20s, "
            "plus a few offline maintenance gaps).",
            "- `COMP` is inverse logic to intuition: 1 denotes no air intake (offloaded/idle), "
            "0 denotes active air intake (under load).",
            "",
            "---",
            "",
            "## 2. AI4I 2020 Predictive Maintenance Dataset",
            "",
            f"- **Source File:** `data/raw/ai4i/{ai4i['file']}`",
            f"- **Total Rows:** {ai4i['row_count']:,}",
            f"- **Total Machine Failures:** {ai4i['total_failures']} ({ai4i['failure_pct']}%)",
            f"- **Multi-failure Rows:** {ai4i['multi_failure_rows']} rows with >1 failure flag",
            f"- **Variant Distribution:** L: {ai4i['type_counts'].get('L', 0):,}, "
            f"M: {ai4i['type_counts'].get('M', 0):,}, H: {ai4i['type_counts'].get('H', 0):,}",
            "",
            "### Failure Mode Breakdown",
            f"- **Tool Wear Failure (`TWF`):** {ai4i['failure_mode_counts']['TWF']}",
            f"- **Heat Dissipation Failure (`HDF`):** {ai4i['failure_mode_counts']['HDF']}",
            f"- **Power Failure (`PWF`):** {ai4i['failure_mode_counts']['PWF']}",
            f"- **Overstrain Failure (`OSF`):** {ai4i['failure_mode_counts']['OSF']}",
            f"- **Random Failure (`RNF`):** {ai4i['failure_mode_counts']['RNF']}",
            "",
            "### Column Summary",
            "| Column | Type | Min | Max | Mean | Missing | Notes |",
            "|---|---|---|---|---|---|---|",
        ]
    )

    for col, st in ai4i["columns"].items():
        min_v = st.get("min", "-")
        max_v = st.get("max", "-")
        mean_v = st.get("mean", "-")
        lines.append(
            f"| `{col}` | {st['dtype']} | {min_v} | {max_v} | {mean_v} | {st['missing']} | |"
        )

    lines.extend(
        [
            "",
            "### Known Quirks & Leakage Traps",
            "- `Machine failure` is a deterministic combination of failure modes plus random "
            "failures. Modes `TWF`, `HDF`, `PWF`, `OSF`, `RNF` **must not** be used as "
            "predictive features.",
            "- AI4I has no temporal dimension or machine persistence; each row is an "
            "independent synthetic machine state.",
            "",
            "---",
            "",
            "## 3. UCI Online Retail Dataset",
            "",
            f"- **Source File:** `data/raw/retail/{retail['file']}` "
            "(cached as `online_retail.parquet`)",
            f"- **Total Rows:** {retail['row_count']:,}",
            f"- **Date Range:** {retail['date_min']} to {retail['date_max']}",
            f"- **Distinct Invoices:** {retail['unique_invoices']:,}",
            f"- **Cancellations:** {retail['cancellations']:,} (InvoiceNo starting with 'C')",
            f"- **Negative Quantity Records:** {retail['negative_quantity_rows']:,}",
            f"- **Missing Customer IDs:** {retail['missing_customer_id_rows']:,}",
            f"- **Zero or Negative Unit Price:** {retail['zero_or_negative_price_rows']:,}",
            "- **Top 5 Countries by Revenue (GBP):**",
        ]
    )

    for c, rev in retail["top_5_countries_by_revenue"].items():
        lines.append(f"  - **{c}**: £{rev:,.2f}")

    lines.extend(
        [
            "",
            "### Column Summary",
            "| Column | Type | Min | Max | Mean | Missing | Notes |",
            "|---|---|---|---|---|---|---|",
        ]
    )

    for col, st in retail["columns"].items():
        min_v = st.get("min", "-")
        max_v = st.get("max", "-")
        mean_v = st.get("mean", "-")
        lines.append(
            f"| `{col}` | {st['dtype']} | {min_v} | {max_v} | {mean_v} | {st['missing']} | |"
        )

    lines.extend(
        [
            "",
            "### Known Quirks",
            "- Cancellations carry negative quantities and often mirror earlier valid invoices.",
            "- Several stock codes are non-product administrative records (e.g. `POST`, `D`).",
            "- Day 2 preserves all cancellations, negative quantities, and missing customer IDs "
            "as raw data; cleaning is deferred to Day 3.",
            "",
            "---",
            "",
            "## 4. M5 Forecasting Dataset (Optional)",
            "",
            "- **Source Status:** Optional. If present in `data/raw/m5/`, consists of "
            "`calendar.csv`, `sales_train_validation.csv`, and `sell_prices.csv`.",
            "- **Planned Subset:** Top 30 items in store `CA_1` across 1,913 days (57,390 rows).",
            "- **Notes:** Unshifted demand benchmark for retail demand modeling.",
            "",
            "---",
            "",
            "## 5. Unified Internal Schema (Contract)",
            "",
            "The target schema generated on Day 2 and stored under `data/raw/unified/`:",
            "",
            "### 5.1 `machines.csv`",
            "- `machine_id` (str, PK): Unique machine code (`M-101` to `M-112`)",
            "- `machine_type` (str): `Air Compressor`, `Hydraulic Press`, `Conveyor Drive`",
            "- `location` (str): Plant and production line",
            "- `installation_date` (date): Date of installation",
            "- `status` (str): Operational state snapshot at timeline end "
            "(`Running`, `Warning`, `Critical`, `Maintenance`, `Idle`)",
            "- `data_origin` (str, extension): `metropt_real_signals` or `metropt_derived`",
            "- `scenario` (str, extension): Simulation scenario (e.g. `hero_degrading`, `healthy`)",
            "- `ai4i_type` (str, extension): Quality tier variant (`L`, `M`, `H`)",
            "",
            "### 5.2 `sensors_raw.csv`",
            "- `timestamp` (datetime ISO 8601 UTC): 5-minute sampling interval",
            "- `machine_id` (str, FK): Identifier of the machine",
            "- `temperature` (float, °C): Operating oil / body temperature",
            "- `vibration` (float, mm/s RMS): Vibration level (synthetic formula with AR(1))",
            "- `pressure` (float, bar): System pneumatic / hydraulic pressure",
            "- `rpm` (float, rev/min): Rotational operating speed",
            "- `energy_consumption` (float, kWh): 5-minute electrical energy consumption",
            "- `operating_hours` (float, hours): Cumulative non-decreasing operating hours",
            "- `motor_current` (float, A, extension): Motor phase current",
            "- `comp_state` (int 0/1, extension): Compressor load/unload state",
            "",
            "### 5.3 `maintenance_raw.csv`",
            "- `maintenance_id` (str, PK): Sequential ID (`MNT-000001`...)",
            "- `machine_id` (str, FK): Machine serviced",
            "- `maintenance_date` (date): Date of intervention",
            "- `maintenance_type` (str): `Inspection`, `Preventive`, `Corrective`, `Predictive`",
            "- `failure_type` (str): Cause of corrective maintenance (`Air Leak` or `None`)",
            "- `maintenance_cost` (float, GBP): Associated intervention cost",
            "",
            "### 5.4 `production_raw.csv`",
            "- `timestamp` (date 00:00 UTC): Production date",
            "- `machine_id` (str, FK): Machine identifier",
            "- `production_quantity` (int): Units produced",
            "- `target_quantity` (int): Target planned units",
            "- `downtime_minutes` (int): Total downtime (0 to 1,440 min)",
            "",
            "### 5.5 `business_daily_raw.csv`",
            "- `date` (date): Date",
            "- `machine_id` (str, FK): Machine identifier",
            "- `production` (int): Actual daily units produced",
            "- `orders` (int): Demand orders allocated",
            "- `inventory` (int): Non-negative finished goods inventory",
            "- `downtime` (int): Downtime in minutes",
            "- `maintenance_cost` (float): Direct maintenance expenditure in GBP",
            "",
            "### 5.6 `business_transactions_raw.csv`",
            "- `transaction_id` (str, PK): Unique transaction reference (`{InvoiceNo}-{LineNo}`)",
            "- `timestamp` (datetime ISO 8601 UTC): Time-shifted transaction timestamp",
            "- `product_id` (str): Product identifier (`StockCode`)",
            "- `quantity` (int): Transacted quantity",
            "- `unit_price` (float): Unit price in GBP",
            "- `revenue` (float): Total transaction value (`Quantity * UnitPrice`)",
            "- `customer_id` (str, nullable): Customer account ID",
            "- `invoice_no` (str, extension): Original invoice number",
            "- `description` (str, extension): Product description",
            "- `country` (str, extension): Destination country",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Data dictionary written successfully.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile raw datasets and produce dictionary.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DOCS / "data_dictionary.md",
        help="Path to output markdown data dictionary",
    )
    args = parser.parse_args()

    metro_csv = list((RAW / "metropt").glob("*.csv"))[0]
    ai4i_csv = list((RAW / "ai4i").glob("*.csv"))[0]
    retail_xlsx = list((RAW / "retail").glob("*.xlsx"))[0]

    metro_stats = profile_metropt(metro_csv)
    ai4i_stats = profile_ai4i(ai4i_csv)
    retail_stats = profile_online_retail(retail_xlsx)

    generate_data_dictionary_md(metro_stats, ai4i_stats, retail_stats, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
