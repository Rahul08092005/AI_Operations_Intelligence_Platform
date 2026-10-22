"""Script to construct ml/notebooks/02_production_downtime_energy.ipynb with nbformat."""

import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "ml" / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)
NB_PATH = NB_DIR / "02_production_downtime_energy.ipynb"

nb = nbf.v4.new_notebook()
cells = []

# Title & intro
cells.append(nbf.v4.new_markdown_cell("""# 02: Production, Downtime, and Energy Efficiency

**Project:** AI Operations Intelligence Platform  
**Notebook:** 02 of 06  
**Audience:** Operations & Data Science  
**Prerequisites:** Pipeline complete, `operations_dataset.parquet` populated.

### Purpose
This notebook analyzes factory-floor manufacturing throughput and operational losses:
- Production level, coefficient of variation, and 28-day trends by machine and type (EQ6).
- Weekday vs. weekend production ratios and day-of-week profiles (EQ7).
- Operational target attainment and efficiency distributions (EQ8).
- Downtime drivers classified by root cause proxy (EQ9).
- Empirical production cost of downtime minutes vs theoretical planned slope (EQ10).
- Inventory, order fulfillment, and stock-out proxy evaluation (EQ11).
- Electrical energy efficiency and kWh per unit dynamics (EQ12).
- Curated figures: **F01** (`fig01_fleet_production_downtime.png`) and **F02** (`fig02_machine_efficiency_downtime.png`).
"""))

# Cell 1: Parameters
cells.append(nbf.v4.new_code_cell("""# Parameters
EXPLORATION_FRACTION = 0.7
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

from ml.eda.config import EXPLORATION_FRACTION, SEED, PALETTE, ROOT
from ml.eda.load import load_ops, load_maintenance, load_machines
from ml.eda.splits import is_exploration, exploration_cutoff
from ml.eda.plotting import apply_style, save_fig, mark_holdout
from ml.eda.stats import block_bootstrap_ci, bootstrap_ci, ratio_with_ci

apply_style()
print("Setup complete for Notebook 02.")
"""))

# EQ6 Section
cells.append(nbf.v4.new_markdown_cell("""## EQ6: Production Level, Variability, and Trends by Machine and Type

We evaluate daily production quantities across the exploration period (first 70% of days per Decision D44):
- Mean and median production units.
- Coefficient of variation ($CV = \\sigma / \\mu$).
- 28-day trend slope with moving-block bootstrap 95% confidence intervals (block length = 7 days).
- Fleet daily total production vs. target time series.
"""))

cells.append(nbf.v4.new_code_cell("""ops = load_ops()
exp_ops = ops[is_exploration(ops['date'])].copy()

# Table of production summary per machine
eq6_records = []
for m_id, grp in exp_ops.groupby("machine_id"):
    prod = grp["production_quantity"].to_numpy(dtype=np.float64)
    m_type = grp["machine_type"].iloc[0]
    mean_val = float(np.mean(prod))
    median_val = float(np.median(prod))
    cv_val = float(np.std(prod) / mean_val) if mean_val > 0 else 0.0

    # Trend slope over time
    x = np.arange(len(prod))
    slope = float(np.polyfit(x, prod, 1)[0])
    ci = block_bootstrap_ci(prod, block_len=7, stat=lambda s: np.polyfit(np.arange(len(s)), s, 1)[0], n=500, seed=SEED)

    eq6_records.append({
        "machine_id": m_id,
        "machine_type": m_type,
        "mean_units": round(mean_val, 1),
        "median_units": round(median_val, 1),
        "cv": round(cv_val, 3),
        "trend_slope": round(slope, 3),
        "slope_ci_lower": round(ci[0], 3),
        "slope_ci_upper": round(ci[1], 3),
    })

eq6_df = pd.DataFrame(eq6_records)
display(eq6_df)

# Type summary
type_summary = eq6_df.groupby("machine_type").agg({
    "mean_units": ["mean", "min", "max"],
    "cv": "mean",
    "trend_slope": "mean",
}).round(2)
display(type_summary)
"""))

# EQ7 Section
cells.append(nbf.v4.new_markdown_cell("""## EQ7: Weekday vs Weekend Effects

Industrial schedules often vary by day of week. We compute:
- Weekend-to-weekday production ratio per machine with 95% bootstrap confidence intervals.
- Day-of-week production profile across the fleet.
"""))

cells.append(nbf.v4.new_code_cell("""exp_ops['day_name'] = pd.to_datetime(exp_ops['date']).dt.day_name()
exp_ops['is_weekend'] = pd.to_datetime(exp_ops['date']).dt.dayofweek >= 5

eq7_records = []
for m_id, grp in exp_ops.groupby("machine_id"):
    w_units = grp.loc[grp['is_weekend'], 'production_quantity'].to_numpy(dtype=np.float64)
    wd_units = grp.loc[~grp['is_weekend'], 'production_quantity'].to_numpy(dtype=np.float64)

    ratio, ci_low, ci_high = ratio_with_ci(w_units, wd_units, n=1000, seed=SEED)
    eq7_records.append({
        "machine_id": m_id,
        "machine_type": grp["machine_type"].iloc[0],
        "weekend_mean": round(float(np.mean(w_units)), 1),
        "weekday_mean": round(float(np.mean(wd_units)), 1),
        "weekend_weekday_ratio": round(ratio, 3),
        "ratio_ci_lower": round(ci_low, 3),
        "ratio_ci_upper": round(ci_high, 3),
    })

eq7_df = pd.DataFrame(eq7_records)
display(eq7_df)

dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_profile = exp_ops.groupby("day_name")["production_quantity"].mean().reindex(dow_order)
print("\\nDay of Week Fleet Mean Production:")
print(dow_profile.round(1))
"""))

# EQ8 Section
cells.append(nbf.v4.new_markdown_cell(r"""## EQ8: Operational Target Attainment and Efficiency

We analyze the distribution of `efficiency` ($units / target$) across the fleet:
- Proportion of machine-days achieving $< 80\%$ efficiency.
- Identification of machines experiencing chronic underperformance.
"""))

cells.append(nbf.v4.new_code_cell("""machines = load_machines()
exp_ops['efficiency_pct'] = exp_ops['efficiency'] * 100.0

low_eff_days = exp_ops[exp_ops['efficiency_pct'] < 80.0]
total_exp_days = len(exp_ops)
print(f"Total Machine-Days in Exploration: {total_exp_days}")
print(f"Machine-Days with Efficiency < 80%: {len(low_eff_days)} ({len(low_eff_days)/total_exp_days:.2%})\\n")

low_eff_summary = low_eff_days.groupby("machine_id").agg(
    low_eff_days=("efficiency", "count"),
    mean_efficiency=("efficiency_pct", "mean"),
    min_efficiency=("efficiency_pct", "min"),
).reset_index().merge(machines[["machine_id", "scenario", "machine_type"]], on="machine_id")

display(low_eff_summary.sort_values("low_eff_days", ascending=False))
"""))

# EQ9 Section
cells.append(nbf.v4.new_markdown_cell("""## EQ9: Downtime Drivers and Cause Proxy Classification

We classify each machine-day into an operational cause proxy:
1. `maintenance_day`: Scheduled or performed maintenance event logged that day.
2. `post_failure`: Machine-day immediately following an unscheduled corrective breakdown.
3. `other`: Unscheduled operational idle time, minor stops, or sensor-related pauses.
"""))

cells.append(nbf.v4.new_code_cell("""maint = load_maintenance()
corr_events = maint[maint['maintenance_type'] == 'Corrective']
post_failure_keys = set()
for _, r in corr_events.iterrows():
    p_date = pd.to_datetime(r['performed_date']).date()
    post_failure_keys.add((r['machine_id'], p_date + pd.Timedelta(days=1)))

def get_cause_proxy(row):
    if row.get('maintenance_event_today', False) or (row.get('downtime_minutes', 0) >= 720 and row.get('maintenance_cost_day', 0) > 0):
        return 'maintenance_day'
    if (row['machine_id'], row['date']) in post_failure_keys:
        return 'post_failure'
    return 'other'

ops['cause_proxy'] = ops.apply(get_cause_proxy, axis=1)

downtime_by_cause = ops.groupby("cause_proxy")["downtime_minutes"].agg(
    total_minutes="sum",
    total_hours=lambda x: round(x.sum() / 60.0, 1),
    share=lambda x: round(x.sum() / ops["downtime_minutes"].sum(), 4)
).reset_index()

print("Fleet Total Downtime by Cause Proxy (Full Timeline):")
display(downtime_by_cause)
"""))

# EQ10 Section
cells.append(nbf.v4.new_markdown_cell("""## EQ10: Production Cost of Downtime

Under the Day 2 data generator design:
$$units = target \\times \\left(1 - \\frac{downtime}{1440}\\right) \\times efficiency$$
We run linear regressions of `production_quantity` on `downtime_minutes` per machine type and compare empirical slopes against the planned theoretical slopes:
- Air Compressor: $-450 / 1440 = -0.3125$ units/min
- Hydraulic Press: $-900 / 1440 = -0.6250$ units/min
- Conveyor Drive: $-1800 / 1440 = -1.2500$ units/min
"""))

cells.append(nbf.v4.new_code_cell("""planned_slopes = {
    "Air Compressor": -450.0 / 1440.0,
    "Hydraulic Press": -900.0 / 1440.0,
    "Conveyor Drive": -1800.0 / 1440.0,
}

eq10_records = []
for m_type, grp in exp_ops.groupby("machine_type"):
    x = grp["downtime_minutes"].to_numpy(dtype=np.float64)
    y = grp["production_quantity"].to_numpy(dtype=np.float64)

    slope, intercept = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1])
    planned = planned_slopes[m_type]

    # Block bootstrap CI on regression slope
    ci = block_bootstrap_ci(
        y, block_len=7,
        stat=lambda s: np.polyfit(x[:len(s)], s, 1)[0],
        n=500, seed=SEED
    )

    eq10_records.append({
        "machine_type": m_type,
        "empirical_slope": round(slope, 4),
        "slope_ci_95": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
        "planned_slope": round(planned, 4),
        "correlation_r": round(corr, 4),
        "provenance": "[SYNTHETIC-BY-DESIGN]",
    })

eq10_df = pd.DataFrame(eq10_records)
display(eq10_df)
"""))

# EQ11 Section
cells.append(nbf.v4.new_markdown_cell("""## EQ11: Production, Orders, and Inventory Balance

We examine factory fleet production vs. incoming commercial orders and factory inventory:
- Daily time series of fleet production, customer orders, and finished goods inventory.
- Stock-out proxy rate (days where orders exceeded available inventory stock).
"""))

cells.append(nbf.v4.new_code_cell("""fleet_daily = ops.groupby("date").agg(
    fleet_production=("production_quantity", "sum"),
    fleet_orders=("orders", "sum"),
    fleet_inventory=("inventory", "sum"),
).reset_index()

stockout_days = (fleet_daily["fleet_orders"] > fleet_daily["fleet_inventory"]).sum()
stockout_share = stockout_days / len(fleet_daily)

print(f"Total Days Analyzed: {len(fleet_daily)}")
print(f"Stock-Out Event Days (Orders > Inventory): {stockout_days} ({stockout_share:.2%})")
print(f"Mean Daily Production: {fleet_daily['fleet_production'].mean():,.1f} units")
print(f"Mean Daily Orders:     {fleet_daily['fleet_orders'].mean():,.1f} units")
print(f"Mean Inventory Buffer: {fleet_daily['fleet_inventory'].mean():,.1f} units")
"""))

# EQ12 Section
cells.append(nbf.v4.new_markdown_cell("""## EQ12: Electrical Energy Consumption and Efficiency per Unit

Energy is calculated directly from empirical motor current (`[REAL-DERIVED]`):
$$energy\\_kwh = \\sqrt{3} \\times 400V \\times I \\times \\cos(\\phi) \\times \\frac{5}{60}$$
We compute the distribution of `energy_per_unit` ($kWh / unit$) and test for drift across the exploration period.
"""))

cells.append(nbf.v4.new_code_cell("""eq12_records = []
for m_id, grp in exp_ops.groupby("machine_id"):
    epu = grp["energy_per_unit"].dropna().to_numpy(dtype=np.float64)
    m_type = grp["machine_type"].iloc[0]

    mean_epu = float(np.mean(epu))
    median_epu = float(np.median(epu))
    p95_epu = float(np.percentile(epu, 95))

    x = np.arange(len(epu))
    slope = float(np.polyfit(x, epu, 1)[0])

    eq12_records.append({
        "machine_id": m_id,
        "machine_type": m_type,
        "mean_kwh_per_unit": round(mean_epu, 4),
        "median_kwh_per_unit": round(median_epu, 4),
        "p95_kwh_per_unit": round(p95_epu, 4),
        "daily_drift_slope": round(slope, 6),
        "provenance": "[REAL-DERIVED]",
    })

eq12_df = pd.DataFrame(eq12_records)
display(eq12_df)
"""))

# Figure Generation: F01 and F02
cells.append(nbf.v4.new_markdown_cell("""## Curated Figures Export: F01 & F02

In accordance with Section 8:
- **F01** (`fig01_fleet_production_downtime.png`): Fleet daily production vs target (with 7-day rolling mean) and daily downtime hours (stacked by cause proxy).
- **F02** (`fig02_machine_efficiency_downtime.png`): Per-machine efficiency distribution (box plot) and mean downtime split by maintenance days vs other days.
"""))

cells.append(nbf.v4.new_code_cell("""# Figure F01: Fleet Production and Downtime
cutoff_dt = exploration_cutoff()
dates_full = pd.to_datetime(fleet_daily["date"])

# Compute fleet downtime by cause proxy
ops_daily_cause = ops.pivot_table(
    index="date", columns="cause_proxy", values="downtime_minutes", aggfunc="sum", fill_value=0
) / 60.0  # convert to hours

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)

# Panel 1: Production vs Target
ax1.plot(dates_full, fleet_daily["fleet_production"], label="Daily Fleet Production", color="#0072B2", alpha=0.55, linewidth=1.2)
roll_7d = fleet_daily["fleet_production"].rolling(7, min_periods=1).mean()
ax1.plot(dates_full, roll_7d, label="7-Day Rolling Mean", color="#0072B2", linewidth=2.2)
fleet_target = 12 * 1050  # weighted aggregate target
ax1.axhline(fleet_target, color="#D55E00", linestyle=":", label=f"Fleet Daily Target ({fleet_target:,} units)")
ax1.set_ylabel("Production (Units)")
ax1.set_title("Fleet Daily Production vs. Target and Downtime Loss (Full Window)", fontsize=13)
mark_holdout(ax1, cutoff_dt)
ax1.legend(loc="upper right", framealpha=0.9)

# Panel 2: Stacked Downtime Hours by Cause Proxy
proxy_cols = [c for c in ["maintenance_day", "post_failure", "other"] if c in ops_daily_cause.columns]
colors_proxy = {"maintenance_day": "#E69F00", "post_failure": "#D55E00", "other": "#56B4E9"}
ax2.stackplot(
    pd.to_datetime(ops_daily_cause.index),
    [ops_daily_cause[c] for c in proxy_cols],
    labels=[c.replace("_", " ").title() for c in proxy_cols],
    colors=[colors_proxy.get(c, "#999999") for c in proxy_cols],
    alpha=0.85
)
ax2.set_ylabel("Downtime (Hours)")
ax2.set_xlabel("Date")
mark_holdout(ax2, cutoff_dt)
ax2.legend(loc="upper right", framealpha=0.9)

plt.tight_layout()
f01_path = save_fig(fig, "fig01_fleet_production_downtime.png")
plt.show()
print(f"Saved F01 to {f01_path}")
"""))

cells.append(nbf.v4.new_code_cell("""# Figure F02: Machine Efficiency and Downtime
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Panel 1: Per-machine efficiency box plot
order_machines = sorted(ops["machine_id"].unique())
sns.boxplot(data=ops, x="machine_id", y="efficiency", order=order_machines, palette=PALETTE, ax=ax1)
ax1.axhline(1.0, color="#D55E00", linestyle=":", label="100% Target")
ax1.axhline(0.8, color="#E69F00", linestyle="--", label="80% Threshold")
ax1.set_title("Machine Efficiency Distribution", fontsize=11)
ax1.set_xlabel("Machine ID")
ax1.set_ylabel("Efficiency (Produced / Target)")
ax1.tick_params(axis="x", rotation=45)
ax1.legend(loc="lower right")

# Panel 2: Mean downtime split by maintenance days vs other days
ops["is_maint_day"] = ops["cause_proxy"] == "maintenance_day"
dt_split = ops.groupby(["machine_id", "is_maint_day"])["downtime_minutes"].mean().unstack() / 60.0
dt_split.columns = ["Non-Maintenance Days", "Maintenance Days"]

dt_split.plot(kind="bar", stacked=False, ax=ax2, color=["#56B4E9", "#E69F00"], alpha=0.9)
ax2.set_title("Mean Downtime: Maintenance vs Non-Maintenance", fontsize=11)
ax2.set_xlabel("Machine ID")
ax2.set_ylabel("Mean Downtime (Hours / Day)")
ax2.tick_params(axis="x", rotation=45)
ax2.legend(loc="upper right")

plt.suptitle("Machine-Level Efficiency & Operational Downtime Impact", fontsize=13)
plt.tight_layout()
f02_path = save_fig(fig, "fig02_machine_efficiency_downtime.png")
plt.show()
print(f"Saved F02 to {f02_path}")
"""))

nb["cells"] = cells

with open(NB_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated {NB_PATH}")
