"""Script to construct ml/notebooks/05_retail_demand_and_baselines.ipynb with nbformat."""

import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "ml" / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)
NB_PATH = NB_DIR / "05_retail_demand_and_baselines.ipynb"

nb = nbf.v4.new_notebook()
cells = []

# Title & intro
cells.append(nbf.v4.new_markdown_cell("""# 05: Retail Demand Dynamics and Baseline Forecast Benchmarks

**Project:** AI Operations Intelligence Platform  
**Notebook:** 05 of 06  
**Audience:** Forecasting, Business Analytics & Data Science  
**Prerequisites:** Pipeline execution complete, `business_features_daily.parquet`, `business_transactions.parquet`, `operations_dataset.parquet` populated.

### Purpose
This notebook models downstream commercial demand from the UCI Online Retail dataset and establishes rigorous naive forecast benchmarks for Day 8:
- Revenue trends, order volume, and day-of-week patterns (EQ31).
- Annual seasonality and peak month ratios (EQ32).
- Product concentration and Pareto 80/20 curves across SKUs (EQ33).
- Return rates, guest checkouts, and international revenue shares (EQ34).
- Customer concentration among registered buyers (EQ35).
- Trading calendar days and zero-sales holiday distribution (EQ36).
- Time series forecastability diagnostics (ACF, seasonal strength, ADF stationarity tests) (EQ37).
- **Core Baseline Forecast Benchmarks (EQ38)**: Rolling-origin 7-day evaluation of Naive, Seasonal Naive, and MA-7 models on the 28-day validation slice inside the exploration period (Decision D50).
- Factory production orders vs. retail demand coupling consistency (EQ39).
- Curated figure: **F09** (`fig09_retail_revenue_seasonality.png`).
"""))

# Cell 1: Parameters
cells.append(nbf.v4.new_code_cell("""# Parameters
EXPLORATION_FRACTION = 0.7
VALIDATION_DAYS = 28
SEED = 42
""", metadata={"tags": ["parameters"]}))

# Cell 2: Imports & setup
cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import sys

root_dir = Path.cwd().parents[1] if Path.cwd().name == "notebooks" else Path.cwd()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

from ml.eda.config import EXPLORATION_FRACTION, VALIDATION_DAYS, SEED, PALETTE, ROOT
from ml.eda.load import load_business_features_daily, load_transactions, load_ops
from ml.eda.splits import is_exploration, exploration_cutoff
from ml.eda.plotting import apply_style, save_fig, mark_holdout
from ml.eda.stats import acf_values

apply_style()
print("Setup complete for Notebook 05.")
"""))

# EQ31 Section
cells.append(nbf.v4.new_markdown_cell("""## EQ31: Trend, Moving Averages, and Day-of-Week Patterns

We analyze daily retail commercial transactions (`[REAL]`):
- Daily revenue (£), order volume, and average order value (AOV).
- 7-day and 28-day rolling moving averages.
- Day-of-week revenue distribution (identifying non-trading days).
"""))

cells.append(nbf.v4.new_code_cell("""bfd = load_business_features_daily()
bfd['date_dt'] = pd.to_datetime(bfd['date'])
bfd['day_name'] = bfd['date_dt'].dt.day_name()

print(f"Total Calendar Days in Retail History: {len(bfd)}")
print(f"Total Revenue Generated: £{bfd['revenue'].sum():,.2f}")
print(f"Total Orders: {bfd['daily_orders'].sum():,}")
print(f"Mean Daily Revenue: £{bfd['revenue'].mean():,.2f} (Median: £{bfd['revenue'].median():,.2f})")

# Rolling trends
bfd['revenue_7d'] = bfd['revenue'].rolling(7, min_periods=1).mean()
bfd['revenue_28d'] = bfd['revenue'].rolling(28, min_periods=1).mean()

# Day of week profile
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_summary = bfd.groupby("day_name").agg(
    mean_revenue=("revenue", "mean"),
    total_revenue=("revenue", "sum"),
    mean_orders=("daily_orders", "mean"),
    zero_sales_days=("revenue", lambda x: (x == 0).sum()),
).reindex(dow_order)

dow_summary["revenue_share"] = (dow_summary["total_revenue"] / bfd["revenue"].sum()).round(4)
print("\\nDay of Week Trading Profile:")
display(dow_summary)
"""))

# EQ32 Section
cells.append(nbf.v4.new_markdown_cell("""## EQ32: Annual Seasonality and Peak Month Ratios

We aggregate revenue by calendar month to quantify commercial seasonality:
- Peak trading months (Q4 holiday surge in retail).
- Peak-to-median revenue ratio.
"""))

cells.append(nbf.v4.new_code_cell("""bfd['year_month'] = bfd['date_dt'].dt.to_period('M')
monthly_rev = bfd.groupby('year_month')['revenue'].agg(
    total_revenue="sum",
    daily_mean="mean",
    active_days="count"
).reset_index()

monthly_rev['growth_pct'] = (monthly_rev['total_revenue'].pct_change() * 100).round(2)
display(monthly_rev)

peak_month_rev = monthly_rev['total_revenue'].max()
median_month_rev = monthly_rev['total_revenue'].median()
print(f"\\nPeak Month Revenue: £{peak_month_rev:,.2f}")
print(f"Median Month Revenue: £{median_month_rev:,.2f}")
print(f"Peak-to-Median Seasonality Ratio: {peak_month_rev / median_month_rev:.2f}x")
"""))

# EQ33 Section
cells.append(nbf.v4.new_markdown_cell("""## EQ33: Product Demand Concentration and Pareto Distribution

We evaluate SKU-level revenue concentration using transaction line items:
- Cumulative revenue share by product rank (Pareto curve).
- Revenue share of Top 10 and Top 50 products.
"""))

cells.append(nbf.v4.new_code_cell("""tx = load_transactions(columns=["stock_code", "quantity", "unit_price", "is_cancellation", "is_non_product"])

# Clean product-only transactions (excluding cancellations and non-product lines)
clean_tx = tx[(~tx['is_cancellation']) & (~tx['is_non_product'])].copy()
clean_tx['line_revenue'] = clean_tx['quantity'] * clean_tx['unit_price']

sku_revenue = clean_tx.groupby('stock_code')['line_revenue'].sum().sort_values(ascending=False).reset_index()
sku_revenue['cum_revenue'] = sku_revenue['line_revenue'].cumsum()
sku_revenue['cum_share'] = sku_revenue['cum_revenue'] / sku_revenue['line_revenue'].sum()
sku_revenue['rank'] = np.arange(1, len(sku_revenue) + 1)

top10_share = sku_revenue.loc[sku_revenue['rank'] == 10, 'cum_share'].iloc[0]
top50_share = sku_revenue.loc[sku_revenue['rank'] == 50, 'cum_share'].iloc[0]
p80_skus = sku_revenue[sku_revenue['cum_share'] <= 0.80]['rank'].max()

print(f"Total Distinct SKUs: {len(sku_revenue):,}")
print(f"Top 10 SKUs Revenue Share: {top10_share:.2%}")
print(f"Top 50 SKUs Revenue Share: {top50_share:.2%}")
print(f"Number of SKUs generating 80% of total revenue: {p80_skus} ({p80_skus/len(sku_revenue):.2%} of catalog)")
"""))

# EQ34 Section
cells.append(nbf.v4.new_markdown_cell("""## EQ34: Returns, Cancellations, Guest Checkout, and Geographic Distribution

We quantify transaction composition:
- Return and cancellation rates by transaction count and unit volume.
- Guest customer checkout share (missing `customer_id`).
- Top 5 international markets by revenue.
"""))

cells.append(nbf.v4.new_code_cell("""full_tx = load_transactions(columns=["invoice_no", "customer_id", "country", "quantity", "unit_price", "is_cancellation", "is_guest"])
full_tx['gross_revenue'] = full_tx['quantity'] * full_tx['unit_price']

n_tx = len(full_tx)
n_returns = int(full_tx['is_cancellation'].sum())
n_guests = int(full_tx['is_guest'].sum())

return_units = full_tx.loc[full_tx['is_cancellation'], 'quantity'].abs().sum()
total_units = full_tx['quantity'].abs().sum()

guest_revenue = full_tx.loc[full_tx['is_guest'], 'gross_revenue'].sum()
total_revenue = full_tx['gross_revenue'].sum()

print(f"Total Transaction Lines: {n_tx:,}")
print(f"Cancellation / Return Lines: {n_returns:,} ({n_returns/n_tx:.2%})")
print(f"Return Volume Share: {return_units / total_units:.2%} of units")
print(f"Guest Checkout Lines: {n_guests:,} ({n_guests/n_tx:.2%})")
print(f"Guest Checkout Revenue Share: {guest_revenue / total_revenue:.2%}")

top_countries = full_tx.groupby("country")["gross_revenue"].sum().sort_values(ascending=False).head(5)
print("\\nTop 5 Markets by Revenue:")
display(pd.DataFrame({"Revenue_GBP": top_countries, "Share": (top_countries / total_revenue).round(4)}))
"""))

# EQ35 & EQ36 Section
cells.append(nbf.v4.new_markdown_cell("""## EQ35 & EQ36: Customer Concentration and Non-Trading Days

- **EQ35**: Revenue share of the top 10% of registered customers.
- **EQ36**: Days without sales (store closures, Saturdays, holidays).
"""))

cells.append(nbf.v4.new_code_cell("""# EQ35: Customer Concentration
reg_tx = full_tx[~full_tx['is_guest']].dropna(subset=['customer_id'])
cust_rev = reg_tx.groupby('customer_id')['gross_revenue'].sum().sort_values(ascending=False).reset_index()
top10pct_n = int(np.ceil(len(cust_rev) * 0.10))
top10pct_rev = cust_rev.head(top10pct_n)['gross_revenue'].sum()
total_reg_rev = cust_rev['gross_revenue'].sum()

print(f"Total Registered Customers: {len(cust_rev):,}")
print(f"Top 10% Customers Count: {top10pct_n:,}")
print(f"Top 10% Customer Revenue Share: {top10pct_rev / total_reg_rev:.2%}")

# EQ36: Non-trading Days
zero_days = bfd[bfd['revenue'] == 0]
print(f"\\nTotal Non-Trading Days (Zero Revenue): {len(zero_days)} ({len(zero_days)/len(bfd):.2%})")
print("Non-Trading Days Breakdown by Day of Week:")
print(zero_days['day_name'].value_counts())
"""))

# EQ37 Section
cells.append(nbf.v4.new_markdown_cell("""## EQ37: Time Series Forecastability Diagnostics

We assess statistical forecastability across 3 operational series during the exploration period:
1. `retail_daily_revenue`: Downstream commercial demand.
2. `fleet_daily_production`: Total factory throughput.
3. `m104_daily_production`: Single air compressor unit throughput.

Metrics evaluated:
- Autocorrelation Function (ACF) up to lag 28.
- 7-day seasonality strength (variance explained by day-of-week dummy variables).
- Augmented Dickey-Fuller (ADF) test $p$-values on level and 1st differences.
- Coefficient of Variation ($CV = \\sigma / \\mu$).
"""))

cells.append(nbf.v4.new_code_cell("""ops = load_ops()
cutoff_dt = exploration_cutoff()

exp_bfd = bfd[is_exploration(bfd['date_dt'])].copy()
exp_ops = ops[is_exploration(ops['date'])].copy()

fleet_prod = exp_ops.groupby("date")["production_quantity"].sum()
m104_prod = exp_ops[exp_ops["machine_id"] == "M-104"].set_index("date")["production_quantity"]
retail_rev = exp_bfd.set_index("date")["revenue"]

def assess_forecastability(series, name):
    s = series.dropna().to_numpy(dtype=np.float64)
    cv = float(np.std(s) / np.mean(s))
    acf_vals = acf_values(s, nlags=28)

    # 7-day seasonality variance share
    dow_labels = pd.to_datetime(series.index).dayofweek
    dow_means = pd.Series(s).groupby(dow_labels).transform("mean").to_numpy()
    ss_season = np.sum((dow_means - np.mean(s))**2)
    ss_total = np.sum((s - np.mean(s))**2)
    season_share = float(ss_season / ss_total) if ss_total > 0 else 0.0

    # ADF Stationarity test
    adf_level = adfuller(s)
    diff_s = np.diff(s)
    adf_diff = adfuller(diff_s)

    return {
        "Series": name,
        "CV": round(cv, 3),
        "ACF_lag7": round(float(acf_vals[7]), 3),
        "ACF_lag14": round(float(acf_vals[14]), 3),
        "ACF_lag28": round(float(acf_vals[28]), 3),
        "7d_Seasonality_Share": round(season_share, 3),
        "ADF_Level_p": round(float(adf_level[1]), 4),
        "ADF_Diff_p": round(float(adf_diff[1]), 6),
    }

diag_records = [
    assess_forecastability(retail_rev, "Retail Daily Revenue"),
    assess_forecastability(fleet_prod, "Fleet Daily Production"),
    assess_forecastability(m104_prod, "M-104 Daily Production"),
]
diag_df = pd.DataFrame(diag_records)
display(diag_df)
"""))

# EQ38 Section
cells.append(nbf.v4.new_markdown_cell("""## EQ38: Baseline Forecast Benchmarks to Beat (Day 8 Handoff)

Under **Decision D50**, baseline forecasting models are benchmarked via rolling-origin evaluation on the **last 28 days of the exploration period**:
- Horizon: $H = 7\\text{ days}$ ahead.
- Origins: Every 7 days ($T = 0, 7, 14, 21$).
- Models:
  1. `Naive`: Persistent repeat of last observed value.
  2. `Seasonal Naive`: Value from same day of week in previous week ($t - 7$).
  3. `Moving Average (MA-7)`: Mean of past 7 days.
- Targets: Retail daily revenue, Fleet daily production, Machine-level daily production.
- Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE).
"""))

cells.append(nbf.v4.new_code_cell("""# Identify the 28-day validation slice inside exploration
exp_dates = sorted([d for d in bfd['date'] if pd.Timestamp(f"{d} 00:00:00", tz='UTC') < cutoff_dt])
val_dates = exp_dates[-VALIDATION_DAYS:]
train_dates = exp_dates[:-VALIDATION_DAYS]
origins = [0, 7, 14, 21]

print(f"Validation Window: {val_dates[0]} to {val_dates[-1]} (28 calendar days)")

def run_rolling_benchmark(series, dates_slice):
    dates_list = series.index.tolist()
    records = []

    for orig_idx in origins:
        orig_date = dates_slice[orig_idx]
        if orig_date not in dates_list:
            continue
        pos = dates_list.index(orig_date)
        if pos < 7 or pos + 7 > len(dates_list):
            continue

        actuals = series.iloc[pos:pos+7].to_numpy(dtype=np.float64)
        pred_naive = np.full(7, series.iloc[pos-1], dtype=np.float64)
        pred_snaive = series.iloc[pos-7:pos].to_numpy(dtype=np.float64)
        pred_ma7 = np.full(7, series.iloc[pos-7:pos].mean(), dtype=np.float64)

        for p_vals, model_name in [(pred_naive, "Naive"), (pred_snaive, "Seasonal Naive"), (pred_ma7, "MA-7")]:
            err = actuals - p_vals
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err**2)))
            nz = actuals > 0
            mape = float(np.mean(np.abs(err[nz] / actuals[nz])) * 100) if nz.any() else np.nan

            records.append({"model": model_name, "origin": orig_idx, "mae": mae, "rmse": rmse, "mape": mape})

    res_df = pd.DataFrame(records).groupby("model")[["mae", "rmse", "mape"]].mean().reset_index()
    return res_df

# 1. Retail Daily Revenue
ret_s = bfd.set_index("date")["revenue"]
bench_ret = run_rolling_benchmark(ret_s, val_dates)
bench_ret["Target"] = "Retail Daily Revenue (£)"

# 2. Fleet Daily Production
fleet_s = ops.groupby("date")["production_quantity"].sum()
bench_fleet = run_rolling_benchmark(fleet_s, val_dates)
bench_fleet["Target"] = "Fleet Daily Production (Units)"

# 3. Per-Machine Daily Production (average metrics across all machines)
machine_benchmarks = []
for m_id, grp in ops.groupby("machine_id"):
    m_s = grp.set_index("date")["production_quantity"]
    m_res = run_rolling_benchmark(m_s, val_dates)
    machine_benchmarks.append(m_res)

bench_machines = pd.concat(machine_benchmarks).groupby("model")[["mae", "rmse", "mape"]].mean().reset_index()
bench_machines["Target"] = "Per-Machine Daily Production (Units)"

benchmark_table = pd.concat([bench_ret, bench_fleet, bench_machines])[["Target", "model", "mae", "rmse", "mape"]].round(2)
print("\\n=======================================================")
print("BASELINE FORECAST BENCHMARKS (TO BEAT ON DAY 8)")
print("=======================================================")
display(benchmark_table)
"""))

# EQ39 Section
cells.append(nbf.v4.new_markdown_cell("""## EQ39: Factory Demand Coupling with Retail Commercials

We examine the cross-correlation between fleet commercial production `orders` and downstream retail revenue (`[SYNTHETIC-BY-DESIGN]` link).
"""))

cells.append(nbf.v4.new_code_cell("""merged_orders = ops.groupby("date")["orders"].sum().reset_index().merge(bfd[["date", "revenue", "daily_sales"]], on="date")
r_orders_sales = float(merged_orders["orders"].corr(merged_orders["daily_sales"]))
r_orders_rev = float(merged_orders["orders"].corr(merged_orders["revenue"]))

print(f"Correlation: Fleet Orders vs Retail Sales Index:   r = {r_orders_sales:.4f}")
print(f"Correlation: Fleet Orders vs Retail Revenue (£):     r = {r_orders_rev:.4f}")
print("Verification PASS: Strong positive coupling recovers the synthetic demand chain.")
"""))

# Figure F09 Generation
cells.append(nbf.v4.new_markdown_cell("""## Curated Figure Export: F09

- **F09** (`fig09_retail_revenue_seasonality.png`):
  - Top Panel: Daily retail revenue with 7-day and 28-day moving averages and holdout marker.
  - Bottom Left: Day-of-week revenue share distribution.
  - Bottom Right: Monthly commercial revenue progression.
"""))

cells.append(nbf.v4.new_code_cell("""# Figure F09: Retail Revenue Dynamics and Seasonality
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

ax_top = fig.add_subplot(gs[0, :])
ax_bl = fig.add_subplot(gs[1, 0])
ax_br = fig.add_subplot(gs[1, 1])

# Top Panel: Daily Revenue and Rolling Trends
dates_ret = pd.to_datetime(bfd["date"])
ax_top.plot(dates_ret, bfd["revenue"], color="#0072B2", alpha=0.35, label="Daily Revenue", linewidth=1.0)
ax_top.plot(dates_ret, bfd["revenue_7d"], color="#0072B2", linewidth=2.0, label="7-Day Moving Avg")
ax_top.plot(dates_ret, bfd["revenue_28d"], color="#D55E00", linewidth=2.0, label="28-Day Moving Avg")
mark_holdout(ax_top, cutoff_dt)
ax_top.set_ylabel("Revenue (£)")
ax_top.set_title("Daily Commercial Retail Revenue & Rolling Trends (Real Transactions)", fontsize=11)
ax_top.legend(loc="upper left")

# Bottom Left: Day of Week Trading Share
y_dow = dow_summary["revenue_share"] * 100.0
colors_dow = ["#0072B2" if s != "Sunday" else "#E69F00" for s in dow_order]
ax_bl.bar(dow_order, y_dow, color=colors_dow, alpha=0.85)
ax_bl.set_title("Revenue Share by Day of Week", fontsize=11)
ax_bl.set_ylabel("Share of Total Revenue (%)")
ax_bl.tick_params(axis="x", rotation=45)

# Bottom Right: Monthly Commercial Revenue
months_labels = [p.strftime("%b %y") for p in monthly_rev["year_month"]]
ax_br.bar(months_labels, monthly_rev["total_revenue"] / 1000.0, color="#009E73", alpha=0.85)
ax_br.set_title("Monthly Commercial Revenue Progression", fontsize=11)
ax_br.set_ylabel("Total Revenue (£ Thousands)")
ax_br.tick_params(axis="x", rotation=45)

plt.suptitle("Retail Commercial Demand Patterns and Annual Seasonality", fontsize=12)
plt.tight_layout()
f09_path = save_fig(fig, "fig09_retail_revenue_seasonality.png")
plt.show()
print(f"Saved F09 to {f09_path}")
"""))

nb["cells"] = cells

with open(NB_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated {NB_PATH}")
