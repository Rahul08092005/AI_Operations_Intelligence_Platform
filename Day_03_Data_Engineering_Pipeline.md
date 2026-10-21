# Day 3: Data Engineering Pipeline

**Project:** AI Operations Intelligence Platform
**Week:** 1 (Data Engineering + ML Foundation)
**Branch:** `feature/day-03-data-pipeline`
**Audience:** an AI coding agent (Antigravity) working inside the repository, supervised by the project owner.
**Prerequisite:** Day 1 (project skeleton) and Day 2 (raw unified datasets, ORM models) are merged into `develop`.
**Deliverable:** a reproducible pipeline that turns the raw datasets into a **cleaned, validated, integrated and feature-rich operational dataset**, including `operations_dataset.csv`, and loads the cleaned data into PostgreSQL.

---

## 0. How the Agent Must Work

1. **Read first.** Read this file completely. Then read `docs/data_dictionary.md`, `docs/data_mapping.md`, `docs/day02_walkthrough.md`, `data/raw/unified/timeline_metadata.json` (inspect the **actual keys**), `ml/preprocessing/config.py` and the existing `ml/preprocessing/` modules. Do not assume anything not written there or here.
2. **Plan before coding.** Produce a short implementation plan (use a plan / task-list artifact if your environment supports one) listing tasks T1 to T13, the files each creates, and open questions. Pause for human input only on a blocking issue (section 3).
3. **Execute tasks in order.** After each task run its acceptance checks and commit (section 13).
4. **Leakage is the main risk today.** Every feature must use information available at its own timestamp. Section 6 defines the rules and T11 tests them. When in doubt, choose the conservative option and document it.
5. **Do not delete real anomalies.** Cleaning must preserve the real MetroPT failure behavior and the intentional degradation episodes. T5 has an automated check for this.
6. **Finish with a walkthrough report** (`docs/day03_walkthrough.md`): files created, row counts in and out per table, quality-report highlights, acceptance results, deviations from this spec with reasons, open questions.

### Prompt to paste into the agent (short form)

> Read `Day_03_Data_Engineering_Pipeline.md` fully. Work on branch `feature/day-03-data-pipeline`. Follow sections 0, 2 and 6 and execute tasks T1 to T13 in order, running each task's acceptance checks and committing after each. Day 2 outputs in `data/raw/unified/` are the inputs. Never modify raw files. Do not remove real anomalies. Ground-truth labels must never appear in feature tables. Finish with `docs/day03_walkthrough.md`.

---

## 1. Context and Objective

### Pipeline (from the project plan)

```text
Data Ingestion
      |
Data Validation
      |
Missing Value Handling
      |
Outlier Handling
      |
Data Integration
      |
Feature Engineering
      |
operations_dataset.csv  (+ Parquet tables, PostgreSQL load)
```

### Inputs (from Day 2, all in `data/raw/unified/`)

| File | Content |
|---|---|
| `machines.csv` | 12 machines, scenarios, statuses |
| `sensors_raw.csv` | 5-minute sensor readings per machine (M-104 real MetroPT signals, others derived) |
| `maintenance_raw.csv` | Maintenance history |
| `production_raw.csv` | Daily production, targets, downtime per machine |
| `business_daily_raw.csv` | Daily machine-level business metrics |
| `business_transactions_raw.csv` | Online Retail transactions (uncleaned) |
| `timeline_metadata.json` | Time shifts, event timestamps, running threshold `I_OFF`, transform parameters |
| `machine_ai4i_state_raw.csv` | Optional (Day 2 stretch). Not used today |
| `ml/preprocessing/reference/metropt_failure_windows.csv` | Real failure windows (original, unshifted times) |

### Outputs (summary)

| # | Output | Location |
|---|---|---|
| 1 | Pipeline modules (ingest, validate, clean, integrate, features) | `ml/preprocessing/` |
| 2 | Clean base tables (Parquet) | `data/processed/*.parquet` |
| 3 | `sensor_features.parquet` (5-minute, feature-rich) | `data/processed/` |
| 4 | **`operations_dataset.csv`** and `.parquet` (daily, per machine) | `data/processed/` |
| 5 | Business feature tables | `data/processed/` |
| 6 | Ground-truth labels for anomaly evaluation (separate file) | `data/processed/anomaly_eval_labels.parquet` |
| 7 | Data quality report (JSON and Markdown) | `data/processed/quality_report.json`, `docs/data_quality_report.md` |
| 8 | Feature catalog | `docs/feature_catalog.md` and `data/processed/feature_catalog.csv` |
| 9 | Database loader and loaded PostgreSQL tables | `scripts/load_to_db.py` |
| 10 | Tests, Makefile targets, documentation | `tests/`, `Makefile`, `docs/` |

---

## 2. Non-Negotiable Rules

1. **Reproducibility.** The pipeline is deterministic. Running it twice produces identical Parquet content (compare data hashes, not file bytes, if the Parquet writer embeds metadata).
2. **Idempotent and safe.** Raw files are read-only. Processed outputs are overwritten on each run. The database loader replaces data inside a transaction.
3. **No processed data in Git.** `data/processed/**` stays git-ignored (Day 1 rules). Only code, docs and the small `feature_catalog` Markdown are committed. `data/processed/feature_catalog.csv` is generated and ignored.
4. **Time zone.** UTC everywhere. Daily boundaries at 00:00 UTC.
5. **No silent data loss.** Every removed row or modified value must be counted by reason in the quality report. No `drop`, `fillna` or `clip` without logging.
6. **No labels in feature tables.** Failure windows and degradation episodes appear only in `anomaly_eval_labels.parquet`.
7. **Memory discipline.** About 0.8 million sensor rows and about 541,000 retail rows. Process machine by machine where practical, use `float32` for features, avoid unnecessary copies.
8. **Code quality.** Python 3.11, type hints, `pathlib`, `logging` (no bare `print` in library code), docstrings on public functions, small pure functions that are unit-testable. `ruff check .` and `ruff format .` pass.
9. **Stay in scope.** No EDA charts, no model training, no scaling or encoding for models, no API endpoints. Those are Days 4 to 7.
10. **Config-driven.** Thresholds, windows and rules live in `ml/preprocessing/config.py` and `ml/preprocessing/validation_rules.py`, not scattered through the code.

---

## 3. Human Prerequisites and Stop Conditions

### Prerequisites

- Day 2 is merged into `develop`. `python scripts/validate_raw_outputs.py` passes on the current machine (agent verifies in T1).
- A PostgreSQL database reachable at `DATABASE_URL` in `.env`, with the 8 tables created by `python scripts/init_db.py`. How the database runs does not matter (Docker container, native install or a hosted instance). If you use Docker for the database only: `docker compose up -d db`.

### Stop and ask the human when

- `validate_raw_outputs.py` fails on the Day 2 outputs (do not "fix" raw data inside Day 3 code).
- `timeline_metadata.json` lacks the information needed to build evaluation labels (T7 describes the needed items). Do not invent timestamps.
- The database is unreachable at T10 (the pipeline up to T9 can still be completed and committed).

### Do NOT stop for

- Small row-count differences from the approximate numbers in this file. Record the actual numbers.
- The optional Day 2 outputs (AI4I state table, quality-issue log) being absent.

---

## 4. Decisions for Day 3

Record these in `docs/architecture.md` under "Decision Log" (continue numbering after D28 from Day 2).

| # | Decision | Choice | Reason |
|---|---|---|---|
| D29 | Integrated granularities | **Two integrated datasets:** `sensor_features` at 5-minute resolution (for anomaly detection) and `operations_dataset` at daily per-machine resolution (for EDA, forecasting, dashboards) | Models and dashboards need different resolutions. One table would either bloat or lose information |
| D30 | Outlier policy | Physically impossible values become missing (and flagged). Statistical outliers are **flagged only, never altered or removed** | Real failure behavior and benign spikes are signal, not noise |
| D31 | Imputation policy | Gaps of at most **3 intervals (15 minutes)** are interpolated or forward-filled per machine. Longer gaps stay missing and are flagged | Avoids fabricating long stretches of data |
| D32 | Rolling windows | Windows of **1 hour (12 intervals), 6 hours (72) and 24 hours (288)**, trailing (past and current only), minimum 50 percent of the window available | Past-only prevents leakage. Integer windows on a regular grid are fast |
| D33 | Ground truth | Failure windows, precursor windows and degradation episodes go into a **separate file**, never into feature tables | Prevents label leakage into Day 5 |
| D34 | File formats | Parquet is canonical for all tables. CSV only for `operations_dataset.csv` (as specified in the plan) | Size and speed |
| D35 | Retail cleaning | Returns and cancellations are **kept and flagged**. Rows with invalid price are **rejected** into a separate file. Non-product stock codes and extreme quantities are **flagged and excluded from KPIs** | Keeps the data auditable while protecting KPIs |
| D36 | Database load | Replace mode inside one transaction, PostgreSQL `COPY` for speed, foreign-key-safe order | Idempotent and fast |
| D37 | Time reference | UTC, daily boundaries at 00:00 UTC | Consistency |
| D38 | Warm-up | The first 24 hours of each machine are flagged `is_warmup`, not dropped | Rolling features are incomplete there. Day 5 chooses what to do |
| D39 | Machine baselines | Per-machine baseline deviation features are **deferred to Day 5** | Baselines depend on the train/test split and would leak otherwise |
| D40 | Operations dataset window | Dates covered by the sensor window only, complete days only | Matches the available sensor data |
| D41 | Maintenance feature timing | Maintenance events dated `D` become visible to features from `D + 1` (strictly past) | Conservative leakage rule |

---

## 5. Shared Configuration

### T1 (part): additions to `ml/preprocessing/config.py`

```python
PROCESSED = ROOT / "data" / "processed"

MAX_IMPUTE_INTERVALS = 3                 # 15 minutes at 5-minute resolution
ROLLING_WINDOWS = {"1h": 12, "6h": 72, "24h": 288}   # in 5-minute intervals
ROLLING_MIN_FRACTION = 0.5
WARMUP_INTERVALS = 288                   # first 24 hours per machine
OVERDUE_INSPECTION_H = INSPECTION_INTERVAL_H         # 500
OVERDUE_PREVENTIVE_H = PREVENTIVE_INTERVAL_H         # 2000
FAILURE_LOOKBACK_DAYS = 90
STAT_OUTLIER_ROBUST_Z = 6.0              # flag threshold (median / MAD based)
RETAIL_EXTREME_QTY = 10_000              # verify against the data (T4 note)
```

### T1 (part): `ml/preprocessing/validation_rules.py`

Hard physical limits (values outside are **impossible**, not merely unusual). Start with these, then check them against `docs/data_dictionary.md`:

| Column | Hard min | Hard max | Note |
|---|---|---|---|
| `temperature` | -20 | 150 | degrees C. The hero degradation reaches about 90 and must stay valid |
| `vibration` | 0 | 30 | mm/s. Hero degradation reaches about 8 and must stay valid |
| `pressure` | 0 | 20 | bar |
| `rpm` | 0 | 4000 | |
| `energy_consumption` | 0 | 10 | kWh per 5-minute interval |
| `motor_current` | 0 | 100 | A |
| `operating_hours` | 0 | 200000 | |
| `comp_state` | 0 | 1 | |
| `production_quantity`, `target_quantity` | 0 | 20000 | |
| `downtime_minutes` | 0 | 1440 | |
| `maintenance_cost` | 0 | 100000 | GBP |

If more than 0.1 percent of rows in a column fall outside these limits on the Day 2 data, **widen the limit and document why** instead of removing rows. The limits must never be tight enough to invalidate real or intentionally degraded values.

---

## 6. Leakage Rules (apply to every feature)

| Rule | Meaning |
|---|---|
| L1: trailing only | A feature at time `t` uses data at or before `t`. No centered windows, no `shift(-n)`, no future fill |
| L2: strictly past events | Maintenance and failure features at time `t` use events dated before the date of `t` (D41) |
| L3: no global statistics | No feature uses a statistic computed over the whole dataset (global mean, global min-max). Per-row statistics from trailing windows only. Fitting transformations is Day 5 or 6 work |
| L4: no labels in features | Ground-truth windows and episode flags live only in `anomaly_eval_labels.parquet` |
| L5: forward fill only forward | Imputation may use earlier values and, for interpolation of short gaps, the next valid value within the 15-minute limit (documented, flagged with `is_imputed`) |

T11 contains an automated leakage test: perturbing future values must not change present features.

---

## 7. Tasks

### T1: Branch, verification and configuration (about 20 min)

1. `git checkout develop && git pull && git checkout -b feature/day-03-data-pipeline`
2. Activate the venv. `python scripts/check_env.py` passes.
3. `python scripts/validate_raw_outputs.py` passes. If not, stop (section 3).
4. Add the config constants and `validation_rules.py` (section 5). Create `data/processed/.gitkeep` if missing.
5. Inspect the real keys in `timeline_metadata.json` and write a short summary of them to `docs/day03_walkthrough.md` (draft). This is the contract for T7.

**Acceptance:** config modules import without error. The key summary exists.

---

### T2: Ingestion module (about 40 min)

**Create:** `ml/preprocessing/ingest.py`

Functions (one per raw table) returning typed DataFrames:
`load_machines()`, `load_sensors()`, `load_maintenance()`, `load_production()`, `load_business_daily()`, `load_transactions()`, `load_metadata()`, `load_failure_windows()`.

Requirements:
- Explicit `dtype` and `parse_dates` per column (no inference). Timestamps parsed as UTC (`utc=True`). Dates as `datetime64[ns]` (midnight UTC).
- Sensor numeric columns as `float32`, `machine_id` as `category`.
- Each loader logs: file, rows, columns, memory use, time taken.
- Fail with a clear message if columns are missing or unexpected (column sets from Day 2 section 7).
- Support `sample=True`: only machines `M-103` and `M-104`, and only the last 45 days of sensors/production (used by fast tests and development).
- Return a small `IngestReport` dataclass (table, rows, columns, seconds) collected by the orchestrator.

**Acceptance**
- Each loader returns a frame whose row count equals the raw CSV row count.
- Dtypes are as specified. No pandas inference warnings.
- Sample mode returns two machines.

---

### T3: Validation module (about 60 min)

**Create:** `ml/preprocessing/validate.py`

Validation **detects and reports**. It does not modify data (cleaning is T4 to T6). Each check returns a `ValidationResult(name, table, severity, count, details, passed)`. Severities: `error` (pipeline must handle before continuing), `warning` (reported only).

#### Checks per table

**Sensors**
- required columns present, dtypes correct
- duplicates on (`machine_id`, `timestamp`): count and whether duplicate rows are identical or conflicting
- timestamps strictly on the 5-minute grid; number and length distribution of missing intervals per machine (gaps)
- timestamp order per machine
- hard-range violations per column (section 5 table)
- missing values per column
- `operating_hours` monotonic non-decreasing per machine (count of decreases)
- foreign key: every `machine_id` exists in `machines`

**Production and business_daily**
- one row per (`machine_id`, date), no missing dates inside the window
- hard-range violations
- consistency: `business_daily.production == production.production_quantity` and `downtime == downtime_minutes`
- foreign key check

**Maintenance**
- unique `maintenance_id`
- allowed `maintenance_type` values (`Inspection`, `Preventive`, `Corrective`, `Predictive`)
- `failure_type` is `None` unless `Corrective`
- cost range, date range, foreign key

**Transactions**
- missing `customer_id`, missing `description`
- `unit_price <= 0` rows
- `quantity < 0` rows, cancellation rows (`invoice_no` starting with `C`), and whether the two sets agree
- `revenue == quantity * unit_price`
- non-product stock codes (see T6 for the rule)
- extreme quantities (|quantity| at or above `RETAIL_EXTREME_QTY`; also report the top 10 by absolute quantity)
- duplicate `transaction_id`

#### Output
`run_all_validations(tables) -> list[ValidationResult]`, plus a JSON-serializable summary. The orchestrator runs it **twice**: on raw data (before cleaning) and on cleaned data (after cleaning), so the quality report shows before/after.

**Acceptance**
- On the Day 2 data, the "before" run reports the expected known issues (real MetroPT gaps in M-104, retail cancellations, missing customers, non-positive prices). Record actual counts.
- Zero `error` results involving foreign keys or cross-table consistency (if any appear, stop: Day 2 problem).
- Unit tests exist for each check using tiny synthetic frames (T11).

---

### T4: Missing-value and duplicate handling (about 60 min)

**Create:** `ml/preprocessing/clean_sensors.py` (sensor part) and `ml/preprocessing/clean_other.py` (maintenance, production, business_daily).

#### Sensors (per machine, in this order)

1. **Duplicates:** exact duplicate rows are dropped. For conflicting duplicates on (`machine_id`, `timestamp`) keep the row with fewer missing values, then the first. Count each case.
2. **Regular grid:** reindex each machine to the full 5-minute grid between its first and last timestamp. Rows that did not exist in the raw data are created with all signals missing and `is_gap = True` (so gaps become explicit rather than silent).
3. **Hard-range violations:** set the violating value to missing and set `is_sensor_fault = True` for that row. (Policy D30.) Count per column.
4. **Short-gap imputation:** for each signal column independently, fill runs of missing values of **length at most `MAX_IMPUTE_INTERVALS`**:
   - `temperature`, `pressure`, `rpm`, `motor_current`, `vibration`: linear interpolation between the neighboring valid values
   - `comp_state`: forward fill
   - `energy_consumption`: recompute from the (possibly imputed) `motor_current` using the same formula as Day 2 (`0.59 * I * 5 / 60`)
   - `operating_hours`: forward fill only, never decreasing
   Mark `is_imputed = True` on every row with at least one imputed value. Longer runs are left missing.
5. **Recompute `running`:** `running = motor_current > I_OFF` (value from `timeline_metadata.json`). Missing current gives `running = False` and the row keeps its missing flags.
6. **Warm-up flag:** the first `WARMUP_INTERVALS` rows of each machine get `is_warmup = True`.
7. Return the cleaned frame plus a `CleaningLog` (counts by machine, column and reason).

#### Production and business_daily
- Expected to be complete. If a (`machine_id`, date) row is missing, **do not invent values**: report it as a warning and leave it absent.
- Out-of-range values become missing and are flagged in the log (do not clip).
- `business_daily` consistency with `production` is re-verified. Fix nothing silently: report.

#### Maintenance
- No imputation. Duplicate `maintenance_id` rows are an error (report). Normalize `failure_type` to the string `None` for non-corrective rows.

**Acceptance**
- After cleaning, no machine has duplicate timestamps, and the grid is regular (difference between consecutive timestamps is exactly 5 minutes for every machine).
- No imputed run is longer than 3 intervals (assert).
- `operating_hours` is non-decreasing per machine after cleaning.
- Counts of imputed values, removed duplicates and nullified faults are in the log.

---

### T5: Outlier handling and anomaly-preservation check (about 60 min)

**Create:** `ml/preprocessing/outliers.py`

#### Tier 1: impossible values (done in T4 step 3)
Already handled. Tier 1 is the only tier that changes values.

#### Tier 2: statistical outlier flags (flag only)
For each machine and each of `temperature`, `vibration`, `pressure`, `rpm`, `energy_consumption`, `motor_current`:
- compute a **trailing** robust z-score using the median and MAD of the previous 24 hours (288 intervals, excluding the current value), shifted by one interval so the current value is not in its own baseline;
- `is_outlier_flag = True` if the absolute robust z-score exceeds `STAT_OUTLIER_ROBUST_Z` for at least one signal;
- also store `outlier_signals` (comma-separated names of the triggering signals) for transparency.

**These flags never change values.** They are informational features and the quality-report input.

#### Retail outlier flags (see T6 for rules)
Applied in the retail module.

#### Anomaly-preservation check (automated, part of the quality report)
Using `anomaly_eval_labels` (T7) and the real failure windows:
1. **Zero real data destroyed:** for M-104 intervals inside any shifted failure window, the share of rows with `is_sensor_fault = True` must be 0. If not, the hard limits are too tight: widen and document.
2. **Peak preservation:** the maximum of M-104 `vibration` and `temperature` in the last 7 days is unchanged by cleaning (compare raw vs clean: the maxima must be identical).
3. **Benign spikes kept:** for M-109 (`benign_spikes`), the number of intervals with vibration above 1.5 times its rolling median is the same before and after cleaning.
4. Report the fraction of rows flagged by Tier 2 per machine. Expect the highest rate for M-104 and M-109. A very high rate (above 5 percent for a healthy machine) indicates a threshold problem: review `STAT_OUTLIER_ROBUST_Z`.

**Acceptance:** all four preservation checks pass and are included in the quality report.

---

### T6: Retail transactions cleaning and flags (about 50 min)

**Create:** `ml/preprocessing/clean_transactions.py`

Policy D35. Output two tables: `business_transactions` (kept) and `business_transactions_rejected` (with a `reject_reason` column).

#### Rejected (removed from the main table, saved separately)
- `unit_price <= 0` (reason `non_positive_price`; includes manual adjustments and free items)
- duplicate `transaction_id` beyond the first (reason `duplicate_id`)
- missing `timestamp` or `quantity` (reason `missing_required`)

#### Kept but flagged (new boolean columns)
| Flag | Rule |
|---|---|
| `is_cancellation` | `invoice_no` starts with `C` |
| `is_return` | `quantity < 0` (includes cancellations) |
| `is_guest` | `customer_id` is missing (keep the row, do not impute) |
| `is_non_product` | `product_id` does not look like a product code. Product codes in this dataset are normally 5 digits optionally followed by letters; service codes such as postage, fees, manual entries and samples do not follow that pattern. Inspect the actual distinct codes that do not match `^\d{5}[A-Za-z]*$` and **list them in the quality report** before applying the rule |
| `is_extreme_quantity` | absolute `quantity` at or above `RETAIL_EXTREME_QTY`. **Inspect the data first:** the dataset is known to contain a few very large order and cancellation pairs. Set the threshold so only those rows are caught, and document the value |

#### Derived columns
`date` (UTC date of `timestamp`), `revenue_clean` = `revenue` where the row is not flagged `is_non_product` and not `is_extreme_quantity`, otherwise 0 (used only by KPIs; the original `revenue` stays untouched).

**Acceptance**
- `rows_kept + rows_rejected == raw rows` (no row lost).
- Flag counts match the validation "before" counts.
- `revenue_clean` is never larger in absolute value than `revenue`.
- Rejected file contains a `reject_reason` for every row.

---

### T7: Ground-truth labels for anomaly evaluation (about 40 min)

**Create:** `ml/preprocessing/labels.py`. Output: `data/processed/anomaly_eval_labels.parquet`.

**Purpose:** give Day 5 an honest evaluation target without contaminating features.

Columns: `machine_id`, `timestamp`, `in_failure_window` (bool), `in_precursor_window` (bool), `failure_window_id` (nullable), `episode_type` (string, see below), `episode_id` (nullable), `label_source` (`real` or `synthetic`).

Sources of information:
- **Real failures (M-104):** from `metropt_failure_windows.csv`, shifted with the `shift_days` value from `timeline_metadata.json`. `in_failure_window` is true between start and end. `in_precursor_window` is the 72 hours before the start. `label_source = real`.
- **Synthetic episodes (all machines):** from the event timestamps and episode definitions stored by Day 2 in `timeline_metadata.json` (hero degradation of M-104 over the last 35 days, M-103 drift over the last 21 days, generic failure events for M-105, M-110, M-111 with their 3-day precursors and outages, M-106 pre-maintenance drift, M-109 spikes, M-112 maintenance outage). `episode_type` values: `real_failure`, `real_precursor`, `hero_degradation`, `mild_drift`, `synthetic_failure_precursor`, `synthetic_failure_outage`, `benign_spike`, `pre_maintenance_drift`, `planned_maintenance_outage`, `normal`. `label_source = synthetic`.
- If a needed timestamp is not available in the metadata, **stop and ask** (section 3). Do not infer episode boundaries from the sensor values themselves.

One row per (`machine_id`, `timestamp`) on the same clean 5-minute grid as `sensors.parquet`.

**Acceptance**
- Row count equals the clean sensor row count.
- M-104 has `in_failure_window = True` rows equal in number to the failure window durations (within one interval of rounding).
- Every `episode_type` other than `normal` can be traced to a documented source (metadata key or failure window file). List the mapping in the walkthrough.
- The file contains no sensor columns, and no feature table contains any column from it (checked in T11).

---

### T8: Sensor feature engineering (about 90 min)

**Create:** `ml/preprocessing/features_sensor.py`. Input: clean sensors (T4, T5) and maintenance. Output: `data/processed/sensor_features.parquet`.

All computations are **per machine on the regular 5-minute grid** using integer windows (`rolling(window=n, min_periods=ceil(0.5*n))`). Trailing windows only (leakage rule L1). Output feature columns as `float32`.

#### 8a. Base columns carried through
`timestamp`, `machine_id`, `temperature`, `vibration`, `pressure`, `rpm`, `energy_consumption`, `operating_hours`, `motor_current`, `comp_state`, `running`, `is_imputed`, `is_gap`, `is_sensor_fault`, `is_outlier_flag`, `outlier_signals`, `is_warmup`.

#### 8b. Rolling statistics
For signals `temperature`, `vibration`, `pressure`, `rpm`, `energy_consumption`, `motor_current` and windows `1h`, `6h`, `24h`:
- `{signal}_roll_mean_{window}`
- `{signal}_roll_std_{window}`

For `temperature` and `vibration` additionally `{signal}_roll_max_1h` and `{signal}_roll_max_24h`.

#### 8c. Changes and trends
- `temperature_change`, `vibration_change`, `pressure_change`, `energy_change`: current value minus the value 1 hour earlier (12 intervals)
- the same four with suffix `_24h` (24 hours earlier, 288 intervals)
- `vibration_slope_24h`, `temperature_slope_24h`: least-squares slope per hour over the trailing 24 hours (compute with vectorized rolling sums of `x`, `y`, `xy`, `x^2`, not a Python-level apply)

#### 8d. Operating-context features
- `running_ratio_1h`, `running_ratio_24h`: trailing mean of `running`
- `load_ratio_1h`: trailing mean of `comp_state`
- `vibration_above_warning` (vibration greater than `VIB_WARNING`), `vibration_above_critical` (greater than `VIB_CRITICAL`), `temperature_above_spec` (greater than `TEMP_MAX_SPEC`), each as 0/1 (missing when the signal is missing)
- `vibration_above_warning_ratio_24h`: trailing mean of `vibration_above_warning`

#### 8e. Maintenance context (as-of join, strictly past, rule L2 and D41)
For each (`machine_id`, `timestamp`), considering only maintenance records with `maintenance_date` **before** the date of `timestamp`:

| Feature | Definition |
|---|---|
| `days_since_maintenance` | days since the last record of any type |
| `days_since_inspection` | days since the last `Inspection` |
| `days_since_preventive` | days since the last `Preventive` |
| `operating_hours_since_inspection` | `operating_hours(t)` minus `operating_hours` at the end of the event date, when the event is inside the sensor window. For events before the window use `days_since * mean_daily_running_hours` of that machine (computed from the window) |
| `operating_hours_since_preventive` | same rule |
| `hours_since_estimated` | 1 if either hours feature used the estimate, else 0 |
| `inspection_overdue` | `operating_hours_since_inspection >= OVERDUE_INSPECTION_H` |
| `preventive_overdue` | `operating_hours_since_preventive >= OVERDUE_PREVENTIVE_H` |
| `maintenance_count` | cumulative count of past records (all history in the file) |
| `maintenance_count_90d` | records in the previous 90 days |
| `maintenance_cost_avg` | expanding mean cost of past records |
| `maintenance_cost_90d` | sum of costs in the previous 90 days |
| `failure_count` | cumulative count of past `Corrective` records |
| `days_since_failure` | days since the last `Corrective` record (missing if none) |
| `previous_failure` | 1 if a `Corrective` record exists within the previous `FAILURE_LOOKBACK_DAYS` days |

Implementation hint: compute these once **per day per machine** (date-level table), then map to the 5-minute rows by date. This is far faster than an interval-level join.

#### 8f. Feature catalog
Generate `data/processed/feature_catalog.csv` and `docs/feature_catalog.md` listing for every output column: `name`, `group`, `definition`, `unit`, `window`, `source_columns`, `leakage_rule`, and `doc_name` (the equivalent name used in the dataset document, for example `rolling_temperature_mean` maps to `temperature_roll_mean_1h`, `rolling_vibration_std` to `vibration_roll_std_1h`).

**Acceptance**
- Row count equals the clean sensor row count. No duplicate (`machine_id`, `timestamp`).
- Feature columns are `float32` (flags `bool` or `int8`).
- NaN rates are reported per feature in the quality report. Rolling features are missing only where the window is below the 50 percent minimum (warm-up and long gaps).
- Leakage test (T11) passes.
- Sanity checks: M-104 `vibration_roll_mean_24h` in the last 24 hours is the highest among all machines. M-104 `preventive_overdue` is 1 at the end. M-106 `days_since_preventive` is at most 6 at the end. M-107 `preventive_overdue` is 1 at the end.

---

### T9: Daily integration and business features (about 90 min)

**Create:** `ml/preprocessing/integrate.py` and `ml/preprocessing/features_business.py`. Outputs: `operations_dataset.csv`, `operations_dataset.parquet`, `business_features_daily.parquet`, `product_demand_daily.parquet`.

#### 9a. Retail business features (company level, daily, full retail date range)
From the cleaned transactions (excluding rejected rows; KPI columns use rows with `is_non_product = False` and `is_extreme_quantity = False`):

| Feature | Definition |
|---|---|
| `daily_sales` | units sold (positive quantities of non-cancelled lines) per day |
| `daily_returns_units` | absolute units of returned lines per day |
| `revenue` | `revenue_clean` summed per day (net of returns) |
| `daily_orders` | distinct non-cancellation invoices per day |
| `daily_customers` | distinct non-guest customers per day |
| `average_order_value` | gross revenue of non-cancelled lines divided by `daily_orders` |
| `weekly_sales` | trailing 7-day sum of `daily_sales` |
| `monthly_sales` | trailing 28-day sum of `daily_sales` |
| `sales_growth` | `weekly_sales` divided by the previous `weekly_sales` (7 days earlier) minus 1 |

Calendar days without transactions inside the range get zeros (and are counted in the quality report as "days with no sales", for example public holidays).

`product_demand_daily.parquet`: for the top 50 products by total units (non-product codes excluded), columns `date`, `product_id`, `description`, `units`, `revenue`. Missing product-days filled with 0.

#### 9b. Daily machine-level sensor aggregates
From `sensor_features.parquet`, per (`machine_id`, date), complete days only:
- for `temperature`, `vibration`, `pressure`, `rpm`: `{signal}_mean`, `{signal}_max`, `{signal}_std`
- `energy_kwh` (sum of `energy_consumption`), `motor_current_mean`
- `operating_hours_end` (last value of the day), `running_hours` (sum of `running` times 5/60)
- `vibration_above_warning_minutes`, `vibration_above_critical_minutes`, `temperature_above_spec_minutes` (count of intervals times 5)
- `sensor_coverage` (share of intervals with at least one valid signal), `imputed_share`, `outlier_flag_share`
- `vibration_roll_mean_24h_end`, `temperature_roll_mean_24h_end`, `vibration_slope_24h_end` (value at the last interval of the day)

#### 9c. Join into `operations_dataset`
Grain: one row per (`machine_id`, `date`). Join order:

1. Spine: all (`machine_id`, date) for complete days in the sensor window (D40).
2. `machines`: `machine_type`, `location`, `installation_date`, derived `machine_age_days`. **Do not include `status`, `scenario`, `data_origin`, `ai4i_type`** (the first two describe the final state or the story and would leak).
3. `production`: `production_quantity`, `target_quantity`, `downtime_minutes`.
4. `business_daily`: `orders`, `inventory`, `maintenance_cost`. Rename `maintenance_cost` to `maintenance_cost_day`.
5. Daily sensor aggregates (9b).
6. Maintenance features as of the **start of the day** (the date-level table from T8e): the same columns.
7. Same-day maintenance event context (clearly named, described as same-day information): `maintenance_event_today` (0/1), `maintenance_type_today` (string or `None`).
8. Retail context, prefixed `retail_`: `retail_daily_sales`, `retail_revenue`, `retail_daily_orders`, `retail_average_order_value`, `retail_weekly_sales`, `retail_monthly_sales`, `retail_sales_growth`.

#### 9d. Derived daily features
- `efficiency` = `production_quantity / target_quantity`
- `downtime_hours` = `downtime_minutes / 60`
- `energy_per_unit` = `energy_kwh / production_quantity` (missing when production is 0)
- `production_lag_1`, `production_lag_7` (values from 1 and 7 days earlier, same machine)
- `production_roll_mean_7d`, `downtime_roll_mean_7d` (trailing, **excluding** the current day, so they are safe predictors of the current day)
- `orders_lag_1`
- `day_of_week` (0 to 6), `is_weekend`, `month`

#### 9e. Write
- `data/processed/operations_dataset.csv` (UTF-8, ISO dates, `.` decimals, no index) and `.parquet` with the same content.
- Columns ordered by group: keys, machine static, production, business, sensor aggregates, maintenance, retail, derived.
- Row order sorted by `machine_id`, `date`.

**Acceptance**
- 12 machines times the number of complete days in the window (record the actual count). No duplicate (`machine_id`, `date`).
- No missing values in keys, production, `target_quantity`, `downtime_minutes`.
- Missing values elsewhere are limited to documented causes (first-day lags, `energy_per_unit` on zero production, M-104 gap days). The quality report lists NaN counts per column.
- Cross-checks: `production_quantity` equals `production.parquet`; `energy_kwh` equals the sum from `sensor_features` for a random sample of 20 machine-days.
- `operations_dataset.csv` can be re-read with pandas and is equal to the Parquet version (after dtype alignment).
- The columns `status`, `scenario`, `data_origin`, `in_failure_window`, `episode_type` do **not** exist in this dataset.

---

### T10: Load cleaned data into PostgreSQL (about 60 min)

**Create:** `scripts/load_to_db.py` (uses `backend/app/database/session.py` settings and the ORM tables from Day 2).

Arguments: `--mode replace` (default), `--dry-run` (validate and print counts only), `--tables <list>`.

#### Tables loaded

| Parquet | Table | Notes |
|---|---|---|
| `machines.parquet` | `machines` | all 12 rows, includes the Day 2 extension columns |
| `sensors.parquet` | `sensor_readings` | only the Day 2 table columns (`timestamp`, `machine_id`, `temperature`, `vibration`, `pressure`, `rpm`, `energy_consumption`, `operating_hours`, `motor_current`, `comp_state`). **Skip rows where all five main signals are missing** (pure gap rows). Quality flags stay in Parquet only |
| `production.parquet` | `production` | |
| `maintenance.parquet` | `maintenance` | |
| `business_transactions.parquet` | `business_transactions` | cleaned table, not the rejected one. Flags are not loaded |
| (none) | `predictions`, `anomalies`, `ai_insights` | stay empty |

#### Method
1. Check connectivity (clear error: "database not reachable, check DATABASE_URL and that PostgreSQL is running").
2. In **one transaction**: truncate in reverse dependency order (child tables first), load in dependency order (`machines` first), commit. On error roll back and leave the old data untouched.
3. Use PostgreSQL `COPY` via `psycopg` (`cursor.copy`) streaming from the DataFrame in chunks of 100,000 rows (do not use row-by-row inserts). NaN becomes NULL. Timestamps are timezone-aware UTC.
4. After loading, verify per table: row count equals the expected count from the Parquet (after the gap-row rule), `min` and `max` timestamps match, and no foreign key violations. Print a summary table.
5. Run `ANALYZE` on the loaded tables.

**Acceptance**
- `python scripts/load_to_db.py` loads all tables. A second run produces identical counts (idempotent).
- Verification summary printed and saved into the quality report.
- Spot-check SQL (include as a test or in the walkthrough): latest `sensor_readings` row for `M-104` shows vibration above 7.0, and `SELECT machine_id, count(*) FROM sensor_readings GROUP BY 1` returns 12 rows.
- `--dry-run` touches no data.

---

### T11: Quality report, orchestrator and tests (about 90 min)

#### 11a. Orchestrator

**Create:** `ml/preprocessing/pipeline.py` and the entry point `scripts/run_pipeline.py`.

Order: ingest, validate (before), clean sensors and other tables, outliers, transactions, labels, sensor features, daily integration and business features, validate (after), write all outputs, write reports, optional database load.

Options: `--sample` (fast mode, two machines and 45 days), `--skip-db`, `--only <step>` (for debugging), `--log-level`.

The orchestrator must:
- write outputs only after the step succeeds (write to a temporary name, then rename),
- print a final summary table (table, rows in, rows out, seconds),
- exit non-zero on any `error` severity result remaining in the "after" validation.

#### 11b. Quality report

**Create:** `ml/preprocessing/report.py`. Outputs: `data/processed/quality_report.json` and `docs/data_quality_report.md` (generated; the Markdown file is committed).

Contents:
1. Row counts per table: raw, removed (by reason), final.
2. Values set to missing by column (Tier 1), values imputed by column, longest gap per machine.
3. Duplicates removed.
4. Tier 2 outlier flag rates per machine.
5. **Anomaly-preservation results** (T5).
6. Retail: rejected rows by reason, flagged rows by flag, list of non-product codes found, extreme-quantity rows.
7. Validation results before and after, side by side.
8. NaN rates per feature in `sensor_features` and per column in `operations_dataset`.
9. Row-level leakage checks result (T11c).
10. Database load verification (if run).
11. If `data/raw/unified/quality_issue_log.csv` exists (Day 2 optional injection): recovery metrics (share of injected blanks imputed, injected spikes flagged by Tier 2, duplicates removed). Otherwise skip.

#### 11c. Tests

**Create** `tests/test_pipeline.py` (and extend `tests/test_data.py`). Use tiny synthetic frames, not the real datasets, unless marked `integration`.

Required tests:

| Test | Assertion |
|---|---|
| Validation catches issues | each check returns `passed = False` for a crafted bad frame and `True` for a good one |
| Grid reindexing | missing intervals become explicit gap rows |
| Imputation limit | gap of 3 intervals is filled, gap of 4 is not |
| Tier 1 | out-of-range value becomes NaN and `is_sensor_fault` is set |
| Tier 2 is flag-only | values are byte-identical before and after the outlier step |
| Operating hours | never decrease after cleaning |
| **Leakage: rolling features** | modify all values **after** time `t` in a synthetic series; features at `t` and before are identical |
| **Leakage: maintenance features** | adding a maintenance event dated `D` does not change features for timestamps on or before `D` |
| **Leakage: labels** | no column of `anomaly_eval_labels` appears in `sensor_features` or `operations_dataset` |
| Retail rules | cancellation, guest, non-product and extreme-quantity flags and rejection reasons are correct on a crafted frame. Kept plus rejected equals input |
| Daily aggregation | `energy_kwh` equals the manual sum for a crafted day |
| Business KPIs | `average_order_value` and `sales_growth` correct on a crafted example |
| Idempotence (integration) | running the pipeline in `--sample` mode twice yields identical data (compare `pandas.util.hash_pandas_object` sums) |
| Schema (integration) | `operations_dataset` has unique keys and the forbidden columns are absent |

Integration tests run on the real data in `--sample` mode and are skipped automatically if `data/raw/unified/` is empty.

#### 11d. Makefile (TAB-indented recipes)

```makefile
pipeline:
	python scripts/run_pipeline.py --skip-db

pipeline-sample:
	python scripts/run_pipeline.py --sample --skip-db

db-load:
	python scripts/load_to_db.py

data: 
	python scripts/prepare_data.py
	python scripts/run_pipeline.py
```

Update `.PHONY`. The `data` target now runs the complete flow (raw generation, pipeline, database load) and replaces the earlier Day 1 placeholder.

**Acceptance**
- `pytest -v` passes (database-dependent tests skipped when no database).
- `ruff check .` passes.
- `python scripts/run_pipeline.py --sample --skip-db` completes in a few minutes at most and writes all outputs for two machines.
- `docs/data_quality_report.md` is generated and readable.

---

### T12: Documentation (about 40 min)

1. **`docs/data_pipeline.md`** (new): pipeline diagram, the step list with the policies D29 to D41, leakage rules, the list of all output tables with grain and row counts, and how to run it (`make pipeline`, `make db-load`, `make data`).
2. **`docs/data_dictionary.md`**: add a "Processed tables" section for every Parquet table and `operations_dataset.csv` (columns, types, units, nullability).
3. **`docs/feature_catalog.md`**: generated by T8f. Add a short "How to read this" intro.
4. **`docs/architecture.md`**: append D29 to D41 to the Decision Log, replace "Next Steps" with the Day 4 handoff (section 11).
5. **`README.md`**: update the "Data" section with the one-command flow and a link to `docs/data_pipeline.md` and `docs/data_quality_report.md`.
6. **`docs/day03_walkthrough.md`**: final agent report (section 0).

---

### T13: Git, pull request and tag (about 20 min)

See section 13 for the commit sequence.

---

## 8. Output File Contract

All in `data/processed/` (Parquet unless noted). Timestamps are UTC.

| File | Grain | Key columns | Approx. rows |
|---|---|---|---|
| `machines.parquet` | machine | `machine_id` | 12 |
| `sensors.parquet` | machine times 5 minutes | `machine_id`, `timestamp` + signals + flags (`is_gap`, `is_imputed`, `is_sensor_fault`, `is_outlier_flag`, `outlier_signals`, `is_warmup`, `running`) | about 0.7 to 0.9 million (includes explicit gap rows) |
| `sensor_features.parquet` | machine times 5 minutes | same keys + about 90 feature columns (8a to 8e) | same as `sensors.parquet` |
| `maintenance.parquet` | maintenance event | `maintenance_id` | about 100 to 250 |
| `production.parquet` | machine times day | `machine_id`, `timestamp` | about 2,500 |
| `business_daily.parquet` | machine times day | `machine_id`, `date` | same as production |
| `business_transactions.parquet` | invoice line | `transaction_id` | about 540,000 |
| `business_transactions_rejected.parquet` | invoice line | `transaction_id`, `reject_reason` | small |
| `business_features_daily.parquet` | day (company) | `date` | about 300 to 380 |
| `product_demand_daily.parquet` | product times day | `date`, `product_id` | about 50 times days |
| `operations_dataset.parquet` and **`operations_dataset.csv`** | machine times day | `machine_id`, `date` | 12 times days in the window |
| `anomaly_eval_labels.parquet` | machine times 5 minutes | `machine_id`, `timestamp` | same as `sensors.parquet` |
| `feature_catalog.csv` | feature | `name` | about 100 to 130 |
| `quality_report.json` | n/a | n/a | n/a |

---

## 9. Definition of Done

Day 3 is complete when all of the following are true:

1. `python scripts/validate_raw_outputs.py` (Day 2) still passes, and the Day 3 pipeline ran from a clean `data/processed/` without manual edits.
2. `python scripts/run_pipeline.py` produces every file in section 8.
3. The "after" validation has zero `error` results. Every removed or changed value is counted in the quality report.
4. The four anomaly-preservation checks pass (real failure windows untouched, M-104 peaks preserved, benign spikes preserved, healthy machines not over-flagged).
5. `sensor_features.parquet` and `operations_dataset` pass the leakage tests. The forbidden columns are absent. `anomaly_eval_labels.parquet` is separate.
6. `operations_dataset.csv` exists, has unique (`machine_id`, `date`) keys and matches the Parquet version.
7. The cleaned data is loaded into PostgreSQL, counts verified, second load identical.
8. The fleet storyline survives integration: M-104 ends with the highest 24-hour vibration mean, overdue preventive maintenance and a recent inspection record; M-103 is in the warning band; M-106 shows recent maintenance; M-112 shows the maintenance outage.
9. `pytest` and `ruff check .` pass.
10. `docs/data_pipeline.md`, `docs/data_quality_report.md`, `docs/feature_catalog.md`, the updated dictionary, architecture decision log (D29 to D41) and README are written.
11. No processed or raw data files are tracked by Git (`git ls-files data | grep -v -E "\.gitkeep|README.md"` prints nothing).
12. `docs/day03_walkthrough.md` exists. The branch is merged into `develop` and tagged `day-03-complete`.

---

## 10. Challenges, Risks and Mitigations

| # | Challenge | Why it matters | Mitigation |
|---|---|---|---|
| 1 | **Cleaning destroys real anomalies** | Isolation Forest (Day 5) would have nothing to detect | Tier 1 only touches impossible values. Tier 2 flags only. Automated preservation checks (T5) |
| 2 | **Hard limits too tight** | The hero degradation (vibration about 8 mm/s, temperature about 90 degrees C) could be nullified | Limits set well above manual thresholds, verified against the data, widened-and-documented rather than removing rows |
| 3 | **Label leakage into features** | Inflated Day 5 and 6 results | Rules L1 to L5, separate label file, three leakage tests |
| 4 | **Rolling features on irregular data** | Wrong windows when rows are missing | Reindex to a regular grid first, then integer windows |
| 5 | **Joining maintenance in time** | Easy to leak future events | Strictly past (D41), compute at date level, test with crafted events |
| 6 | **Operating hours across gaps** | Cumulative hours can decrease or stall | Forward-fill only, assert monotonic, document small undercount in gaps |
| 7 | **Memory and speed** | About 0.8 million sensor rows with about 90 features, 541,000 retail rows | `float32`, per-machine processing, vectorized slope computation, date-level maintenance join, Parquet |
| 8 | **Retail quirks** | Large order and cancellation pairs, non-product codes, guest customers distort KPIs | Flags plus KPI exclusion rules, listed in the quality report, thresholds checked against the data |
| 9 | **Daily aggregation of partial days** | First and last days may be incomplete | Complete days only (D40), `sensor_coverage` column |
| 10 | **Time zones and daily boundaries** | Off-by-one-day errors in joins | UTC everywhere, dates derived from UTC timestamps, tests with crafted midnight data |
| 11 | **Imputation hides real problems** | Models learn from fabricated values | Maximum gap 15 minutes, `is_imputed` flag, imputed share reported per day and machine |
| 12 | **Production and business tables disagreeing** | Dashboard KPI mismatch | Consistency validation, fail on mismatch |
| 13 | **Database loading speed and atomicity** | Slow row inserts, partially loaded tables | `COPY`, single transaction, verification after load |
| 14 | **Feature explosion** | Too many correlated features slow Day 5 and obscure explanations | Feature catalog with groups. Day 5 selects a subset |
| 15 | **Scope creep** | Day 3 can absorb EDA and modeling | Strict out-of-scope list (section 12) |
| 16 | **Reproducibility** | Different outputs on re-run | No randomness in the pipeline, idempotence test |

---

## 11. Handoff to Day 4 (add to `docs/architecture.md` "Next Steps")

1. **Day 4 goal:** exploratory data analysis in `ml/notebooks/`: production trends, machine health, temperature, vibration, downtime, energy consumption, maintenance frequency, retail business patterns; behavior before real MetroPT failures; AI4I class imbalance and failure-mode analysis; candidate thresholds. Export 8 to 10 charts to `docs/figures/` and write `docs/eda_findings.md`.
2. **Inputs ready:** `operations_dataset.csv`, `sensor_features.parquet`, `anomaly_eval_labels.parquet` (use labels only for **interpreting** charts, never to choose features), `business_features_daily.parquet`, `product_demand_daily.parquet`, `docs/feature_catalog.md`, `docs/data_quality_report.md`.
3. **EDA rules:** use only the early part of the timeline for any statistic that will later drive modeling choices (decide the train/test time split on Day 5 before looking at test-period statistics). Separate findings from real data (M-104 and retail) from findings that merely reflect synthetic assumptions.
4. **Known caveats to carry forward:** vibration, RPM and energy are synthetic or derived; only M-104 has real failure events; per-machine baselines are computed on Day 5; AI4I has no time or machine dimension (open question OQ-1).
5. **Branch:** `feature/day-04-eda`.

---

## 12. What NOT to Do on Day 3

- Do not produce charts, notebooks or EDA conclusions (Day 4).
- Do not train, tune or evaluate any model (Days 5 and 6).
- Do not fit scalers, encoders or imputers on the dataset. Scaling belongs to model training.
- Do not compute global statistics and use them as features.
- Do not clip, winsorize or remove statistical outliers.
- Do not drop the first 24 hours (flag them instead).
- Do not put labels, `status`, `scenario` or `data_origin` into feature tables.
- Do not modify raw files or regenerate Day 2 data inside Day 3 code.
- Do not write API endpoints (Day 7).
- Do not load the optional AI4I replay table or generate RAG documents.

---

## 13. Verification Commands and Git Workflow

### Verification (run in order)

```bash
source .venv/bin/activate
python scripts/check_env.py
python scripts/validate_raw_outputs.py
python scripts/run_pipeline.py --sample --skip-db     # fast check
python scripts/run_pipeline.py --skip-db              # full run
python scripts/init_db.py                             # database must be running
python scripts/load_to_db.py --dry-run
python scripts/load_to_db.py
pytest -v
ruff check .
git status --short                                    # no data or .env files
git ls-files data | grep -v -E "\.gitkeep|README.md"  # must print nothing
```

### Commit sequence (Conventional Commits)

| After | Message |
|---|---|
| T1 | `chore(data): add pipeline configuration and validation rules` |
| T2 | `feat(data): add typed ingestion for unified raw datasets` |
| T3 | `feat(data): add validation checks and report structures` |
| T4 | `feat(data): add duplicate, gap and imputation handling for sensors` |
| T5 | `feat(data): add outlier flags and anomaly preservation checks` |
| T6 | `feat(data): clean and flag retail transactions` |
| T7 | `feat(data): add ground-truth labels for anomaly evaluation` |
| T8 | `feat(features): add sensor rolling, trend and maintenance features` |
| T9 | `feat(features): add daily integration, business features and operations dataset` |
| T10 | `feat(db): add PostgreSQL loader with verification` |
| T11 | `feat(data): add pipeline orchestrator, quality report and tests` |
| T12 | `docs: add pipeline docs, feature catalog and quality report` |
| End | `docs: add Day 3 walkthrough` |

### Merge and tag

```bash
git push -u origin feature/day-03-data-pipeline
gh pr create --base develop --head feature/day-03-data-pipeline \
  --title "Day 3: Data engineering pipeline and operations dataset" \
  --body "See docs/day03_walkthrough.md"
# after review and merge:
git checkout develop && git pull
git tag -a day-03-complete -m "Day 3: cleaned and integrated operations dataset"
git push origin day-03-complete
git checkout -b feature/day-04-eda
```

Before merging, check the PR diff for: no data files, no secrets, no hardcoded absolute paths, no leftover debugging output, no use of `shift(-n)` or centered rolling windows (search for them).

---

## 14. Suggested Time Plan (about 8 hours)

| Time | Task |
|---|---|
| 0:00 to 0:20 | T1 verification and configuration |
| 0:20 to 1:00 | T2 ingestion |
| 1:00 to 2:00 | T3 validation |
| 2:00 to 3:00 | T4 missing values and duplicates |
| 3:00 to 4:00 | T5 outliers and preservation checks, T6 retail cleaning (start) |
| 4:00 to 4:50 | T6 finish, T7 labels |
| 4:50 to 6:20 | T8 sensor features |
| 6:20 to 7:50 | T9 daily integration and business features |
| 7:50 to 8:50 | T10 database load |
| 8:50 to 10:20 | T11 orchestrator, report, tests |
| Remainder | T12 docs, T13 merge |

This is a heavy day. If time runs short, defer in this order: `vibration_slope_24h` and `temperature_slope_24h` (8c), `product_demand_daily.parquet`, the quality-issue recovery metrics (11b item 11), `hours_since_estimated` refinements. Never skip T3, T4, T5, the leakage tests, T9 or T10.
