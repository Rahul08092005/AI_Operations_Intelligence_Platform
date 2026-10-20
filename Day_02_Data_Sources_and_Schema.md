# Day 2: Data Sources, Unified Schema and Raw Dataset Generation

**Project:** AI Operations Intelligence Platform
**Week:** 1 (Data Engineering + ML Foundation)
**Branch:** `feature/day-02-data-sources`
**Audience:** an AI coding agent (Antigravity) working inside the repository created on Day 1, supervised by the project owner.
**Deliverable:** raw, schema-conformant datasets ready for Day 3 processing, a documented unified schema, ORM models for all tables, and fully reproducible generator scripts.

---

## 0. How the Agent Must Work

Follow this protocol in order.

1. **Read first.** Read this file completely. Then read `docs/architecture.md`, `README.md`, `pyproject.toml` and the folder tree. Do not assume anything not written here.
2. **Plan before coding.** Produce a short implementation plan (use a plan / task-list artifact if your environment supports one) listing the tasks T1 to T14 below, the files each will create, and any questions. Wait for human approval of the plan **only if** you found a blocking issue (see section 3). Otherwise proceed.
3. **Execute tasks in order** (T1 to T14). After each task: run its acceptance checks, then commit with a Conventional Commit message (section 14).
4. **Never fabricate data that must be real.** Where this file says a value must come from a source file (for example the MetroPT failure windows), read it from the source. If you cannot, stop and ask the human. Do not invent it.
5. **Finish with a walkthrough report** (`docs/day02_walkthrough.md`) listing: files created, row counts per output, acceptance check results, deviations from this spec with reasons, and open questions.

### Prompt to paste into the agent (short form)

> Read `Day_02_Data_Sources_and_Schema.md` fully. Work on branch `feature/day-02-data-sources`. Follow section 0 (protocol), 2 (rules) and execute tasks T1 to T14 in order, running each task's acceptance checks and committing after each. Raw datasets are downloaded manually by me (section 3). If a required file is missing or a value must come from a source file you cannot read, stop and ask me. Finish with `docs/day02_walkthrough.md`.

---

## 1. Context and Objective

### Project summary

An AI platform for a simulated smart-manufacturing company: sensor data, business data and documents feed ML models (anomaly detection, failure prediction, forecasting), an explanation layer (SHAP + LLM + RAG) and a React dashboard. Stack: FastAPI, PostgreSQL, React, scikit-learn, XGBoost, SHAP.

### Why Day 2 exists

The public datasets come from **unrelated domains** and have different schemas:

| Dataset | What it is | Gap versus the project schema |
|---|---|---|
| MetroPT-3 | Metro train air-compressor sensors, about 1.5 million rows, 15 features | No vibration, no RPM, no energy, no machine ID, single machine |
| AI4I 2020 | 10,000 synthetic predictive-maintenance rows | No timestamps, no machine ID, rows are independent |
| UCI Online Retail | About 541,909 retail transactions (Excel) | No machines, no production data |
| M5 Forecasting | Retail demand (Kaggle) | Retail domain, large |

Day 2 builds a **synthetic company/machine layer** that maps these sources into one internal schema, and generates the datasets that do not exist publicly (maintenance history, production, daily business metrics). Everything synthetic or derived must be **labeled as such**.

### Day 2 outputs (summary)

| # | Output | Location |
|---|---|---|
| 1 | Data dictionary for every source | `docs/data_dictionary.md` |
| 2 | Mapping and provenance document | `docs/data_mapping.md` |
| 3 | MetroPT failure-window reference file | `ml/preprocessing/reference/metropt_failure_windows.csv` |
| 4 | Fleet definition (12 machines, scenarios) | `ml/preprocessing/fleet.py` and `data/raw/unified/machines.csv` |
| 5 | Raw unified datasets | `data/raw/unified/*.csv` (section 7) |
| 6 | M5 subset | `data/raw/m5/m5_subset_long.parquet` |
| 7 | ORM models for 8 tables and `scripts/init_db.py` | `backend/app/models/`, `scripts/` |
| 8 | Generator and validation scripts, tests | `ml/preprocessing/`, `scripts/`, `tests/` |

---

## 2. Non-Negotiable Rules

1. **Reproducibility.** One global seed (`SEED = 42`). Every generator uses `numpy.random.default_rng` seeded from `SEED` and the machine index. Running the pipeline twice must produce byte-identical CSVs.
2. **Idempotent scripts.** Safe to re-run. Outputs are overwritten. Raw input files are never modified.
3. **No raw data in Git.** `data/raw/**` is git-ignored (Day 1 rules). Never `git add -f` datasets. The small reference file in `ml/preprocessing/reference/` **is** committed.
4. **No secrets, no network calls in scripts.** Datasets are provided manually. Scripts only read local files.
5. **Honest labeling.** Every derived/synthetic signal is documented in `docs/data_mapping.md` (provenance table) and the machines table has a `data_origin` column.
6. **Memory discipline.** MetroPT has about 1.5 million rows. Read only needed columns, use `float32`, avoid copying large frames, process per machine.
7. **Code quality.** Python 3.11, type hints, `pathlib`, `logging` (no bare `print` in library code), docstrings on public functions. `ruff check .` and `ruff format .` must pass.
8. **Do not start Day 3 work.** No cleaning, outlier removal, imputation, rolling features or model training. Do not load data into PostgreSQL today (only create the empty tables).
9. **Stay inside scope.** Do not add datasets beyond those listed. NASA C-MAPSS, SECOM, Gas Turbine and MIMII are out of scope.
10. **Existing structure only.** Use the Day 1 folder structure. New folders allowed: `data/raw/unified/`, `ml/preprocessing/reference/`.

---

## 3. Human Prerequisites (Manual Downloads) and Stop Conditions

The project owner downloads these **before** the agent starts. The agent must verify them in T1.

| Dataset | Place file(s) in | Expected filename(s) | Source |
|---|---|---|---|
| MetroPT-3 | `data/raw/metropt/` | `MetroPT3(AirCompressor).csv`, `Data Description_Metro.pdf` | https://archive.ics.uci.edu/dataset/791/metropt3%2Bdataset |
| AI4I 2020 | `data/raw/ai4i/` | `ai4i2020.csv` | https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset |
| UCI Online Retail | `data/raw/retail/` | `Online Retail.xlsx` | https://archive.ics.uci.edu/dataset/352/online%2Bretail |
| M5 (optional) | `data/raw/m5/` | `calendar.csv`, `sales_train_validation.csv`, `sell_prices.csv` | https://www.kaggle.com/competitions/m5-forecasting-accuracy/data (requires Kaggle login and accepting the competition rules) |

Filenames from the sources can differ slightly (for example case or spacing). The agent should locate files by pattern (`*.csv`, `*.xlsx`) and report the resolved names.

### Stop conditions: the agent must stop and ask the human when

- MetroPT-3 CSV, AI4I CSV or Online Retail file is missing (these are required).
- The MetroPT failure windows cannot be read from `Data Description_Metro.pdf` (T3). The agent must **not** guess dates.
- A file's columns differ materially from the expectations in section 6 (T2).

### Do NOT stop for

- Missing M5 files: skip T9, log a warning, continue.
- Minor naming differences, small row-count differences from the numbers quoted here. Record actual values in the data dictionary.

---

## 4. Decisions for Day 2

Record these in `docs/architecture.md` under "Decision Log" (continue numbering after D14).

| # | Decision | Choice | Reason |
|---|---|---|---|
| D15 | Fleet size | **12 machines**, `M-101` to `M-112` | Enough variety for a dashboard, small enough to generate quickly |
| D16 | Machine types | Air Compressor (M-101 to M-104), Hydraulic Press (M-105 to M-108), Conveyor Drive (M-109 to M-112) | Variety for the demo. Compressor signals are closest to real MetroPT behavior |
| D17 | Real vs derived signals | **M-104 uses the real MetroPT-3 time series (time-shifted).** Other machines are derived from MetroPT healthy days (day-level block resampling with per-machine scaling) | Gives one machine with genuine failure events for anomaly evaluation (Day 5) |
| D18 | Vibration, RPM, energy | **Not in MetroPT-3.** Generated synthetically from real motor current, load state and temperature, clearly labeled | The project schema and the M-104 story require them |
| D19 | Sensor resolution | **5-minute grid** (`SENSOR_FREQ = "5min"`) | Keeps the fleet around 0.75 million rows. Day 3 may aggregate further |
| D20 | Timeline | MetroPT is time-shifted by a whole number of weeks so its last timestamp lands on `TIMELINE_END = 2026-10-19`. Online Retail is shifted by whole weeks to end on or before the same date | Weekday patterns are preserved. The demo looks current. The shift is stored in `data/raw/unified/timeline_metadata.json` |
| D21 | Currency | All monetary values are in **GBP** (matches Online Retail) | Avoids mixed currencies |
| D22 | M-104 storyline | Real failures in history, a final ~5 weeks of synthetic degradation, inspection on 2026-09-20 recommending preventive maintenance, preventive maintenance overdue, vibration above 7.0 mm/s in the last 48 hours | Matches the project plan's demo scenario and the synthetic manual (normal vibration 0 to 4.5, warning 4.5 to 7.0, critical above 7.0 mm/s, operating temperature 20 to 85 degrees C, inspection every 500 operating hours) |
| D23 | Database loading | Tables are **created** today, **loaded** on Day 3 | Cleaning happens first |
| D24 | Failure model inputs for demo machines | Not decided today. See open question OQ-1 | AI4I has no machine or time dimension. The method is chosen on Day 6 |
| D25 | Single source of truth for production | `production_raw.csv` is generated first. `business_daily_raw.csv` is derived from it | Prevents inconsistent numbers across tables |
| D26 | Extension columns | Allowed but marked "extension" in the dictionary (`motor_current`, `comp_state` in sensors; `data_origin`, `scenario`, `ai4i_type` in machines) | Needed later for load-mode features and storytelling |
| D27 | Reference file location | `ml/preprocessing/reference/` (committed), not `data/external/` (git-ignored) | The failure windows are small and essential for reproducibility |
| D28 | Quality-issue injection | Optional stretch (T13), default **off** | Real MetroPT already contains issues. Avoid over-engineering |

### Open question (resolve on Day 6, record only)

**OQ-1:** AI4I rows have no machine identity or time. Two candidate approaches to feed the failure model for demo machines:
(a) **Replay:** assign AI4I rows to each machine per day according to its scenario (T10, stretch, builds this table).
(b) **Adapter:** quantile-map each machine's sensors onto AI4I feature ranges (risk of physically inconsistent mappings, for example temperature difference).
Document both in `docs/data_mapping.md` under "Open Questions". Do not implement (b).

---

## 5. Shared Configuration

### T0 (part of T1): `ml/preprocessing/config.py`

Create a module of constants used by all generators:

```python
from datetime import date
from pathlib import Path

SEED = 42
TIMELINE_END = date(2026, 10, 19)          # last day of generated data (inclusive)
SENSOR_FREQ = "5min"
CURRENCY = "GBP"

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
UNIFIED = RAW / "unified"
REFERENCE = Path(__file__).resolve().parent / "reference"

# Operating thresholds (from the synthetic machine manual, used for scenario design only)
VIB_WARNING = 4.5       # mm/s
VIB_CRITICAL = 7.0      # mm/s
TEMP_MAX_SPEC = 85.0    # degrees C
INSPECTION_INTERVAL_H = 500
PREVENTIVE_INTERVAL_H = 2000
```

Paths must resolve correctly whether scripts run from the repo root or elsewhere. `UNIFIED` is created on demand (`mkdir(parents=True, exist_ok=True)`).

---

## 6. Tasks

### T1: Branch, environment and source verification (about 20 min)

**Steps**

1. `git checkout develop && git pull && git checkout -b feature/day-02-data-sources`
2. Activate the venv. Run `python scripts/check_env.py` (must pass).
3. Create `ml/preprocessing/config.py` (section 5), `ml/preprocessing/reference/` folder, `data/raw/unified/.gitkeep`.
4. Create `scripts/verify_sources.py`: locates each required file (section 3), prints resolved path, size, and row/column count (MetroPT: count rows by chunked reading; Online Retail: sheet name and shape). Exits non-zero if a **required** file is missing. M5 absence is a warning only.

**Acceptance**
- `python scripts/verify_sources.py` lists MetroPT, AI4I, Online Retail (and M5 if present) with sizes and counts.
- Output is saved to `docs/source_verification.txt`.

---

### T2: Profile the sources and write the data dictionary (about 60 min)

**Create:** `scripts/inspect_raw_data.py` (profiling) and `docs/data_dictionary.md` (written from the profile output plus the notes below). The script must not load the full MetroPT into memory more than once (`usecols`, `float32`, `parse_dates`).

For **each source** the dictionary must contain: file, row count, column list with dtype, unit (where known), min / max / mean / missing count, time range and sampling interval (time-series sources), and "known quirks".

**Expected structure to verify (report differences, do not fail)**

| Source | Expected columns |
|---|---|
| MetroPT-3 | `timestamp`, `TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs`, `Oil_temperature`, `Motor_current`, `COMP`, plus other digital signals (`DV_eletric`, `Towers`, `MPG`, `LPS`, `Pressure_switch`, `Oil_level`, `Caudal_impulses`). The file may contain an unnamed index column |
| AI4I 2020 | `UDI`, `Product ID`, `Type` (L/M/H), `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF` |
| Online Retail | `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country` |
| M5 | `calendar.csv`, `sales_train_validation.csv`, `sell_prices.csv` structures |

**Source-specific items the profile must answer** (write the answers into the dictionary)

- **MetroPT-3:** time range and sampling interval; number and size of gaps; value ranges per signal; the **meaning of `COMP`** (read the PDF: which state corresponds to loaded vs unloaded); histogram-based **running threshold** for `Motor_current` (the value that separates "off" from "running"; record as `I_OFF`); fraction of time running.
- **AI4I:** total failures and the percentage (expected about 339, about 3.4%), counts per failure mode (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`), counts per `Type`, rows with more than one failure mode.
- **Online Retail:** date range, number of invoices, number of cancellations (`InvoiceNo` starting with `C`), rows with negative `Quantity`, rows with missing `CustomerID`, rows with `UnitPrice <= 0`, countries (top 5 by revenue). Cache the Excel read to `data/raw/retail/online_retail.parquet` (reading `.xlsx` is slow).
- **M5:** only structure and sizes (if present).

**Acceptance**
- `docs/data_dictionary.md` exists with one section per source and a final section "Unified internal schema" (copy from section 7 of this file).
- All numbers in the dictionary come from the profile output, not from this document.
- `I_OFF` and the `COMP` meaning are recorded.

---

### T3: MetroPT failure windows reference file (about 20 min)

**Create:** `ml/preprocessing/reference/metropt_failure_windows.csv`

```csv
failure_id,start_time,end_time,failure_type,source
```

**Rules**
- Read the failure reports from `data/raw/metropt/Data Description_Metro.pdf` (the document lists air-leak failures with start and end times).
- Use the **original, unshifted** timestamps. Use the same timezone/format as the CSV timestamps.
- `failure_type` = `Air Leak` unless the PDF specifies otherwise. `source` = `Data Description_Metro.pdf`.
- If the PDF cannot be parsed or the values are ambiguous, **stop and ask the human** to supply them. Do not estimate.
- Each window must lie within the data time range (assert this).

**Acceptance**
- The CSV exists, is committed, and contains every failure listed in the PDF (record the count).
- A short note in `docs/data_dictionary.md` explains how the windows will be used (Day 5: ground truth for anomaly detection evaluation).

---

### T4: Fleet definition (about 30 min)

**Create:** `ml/preprocessing/fleet.py`, containing a `FLEET` list of dataclasses and a function `build_machines_df()`. Output: `data/raw/unified/machines.csv`.

#### Machines table (output columns)

| Column | Type | Notes |
|---|---|---|
| `machine_id` | str | `M-101` to `M-112` |
| `machine_type` | str | See below |
| `location` | str | Plant and line |
| `installation_date` | date | Before the sensor window |
| `status` | str | `Running`, `Warning`, `Critical`, `Maintenance`, `Idle` (snapshot at `TIMELINE_END`) |
| `data_origin` | str | **Extension.** `metropt_real_signals` or `metropt_derived` |
| `scenario` | str | **Extension.** See scenario table |
| `ai4i_type` | str | **Extension.** `L`, `M` or `H` (AI4I product quality variant) |

#### Fleet definition

| machine_id | machine_type | location | installation_date | scenario | ai4i_type | status at end | data_origin |
|---|---|---|---|---|---|---|---|
| M-101 | Air Compressor | Plant A, Line 1 | 2021-03-15 | `healthy` | L | Running | metropt_derived |
| M-102 | Air Compressor | Plant A, Line 1 | 2021-09-01 | `healthy` | M | Running | metropt_derived |
| M-103 | Air Compressor | Plant A, Line 2 | 2020-11-10 | `mild_drift` | M | Warning | metropt_derived |
| M-104 | Air Compressor | Plant A, Line 2 | 2019-06-20 | `hero_degrading` | L | Critical | metropt_real_signals |
| M-105 | Hydraulic Press | Plant B, Line 1 | 2022-02-01 | `past_failure_old` | M | Running | metropt_derived |
| M-106 | Hydraulic Press | Plant B, Line 1 | 2021-07-12 | `recently_maintained` | H | Running | metropt_derived |
| M-107 | Hydraulic Press | Plant B, Line 2 | 2020-05-25 | `overdue_maintenance` | L | Running | metropt_derived |
| M-108 | Hydraulic Press | Plant B, Line 2 | 2022-08-30 | `healthy` | M | Running | metropt_derived |
| M-109 | Conveyor Drive | Plant C, Line 1 | 2021-01-18 | `benign_spikes` | L | Running | metropt_derived |
| M-110 | Conveyor Drive | Plant C, Line 1 | 2020-09-03 | `past_failure_mid` | M | Running | metropt_derived |
| M-111 | Conveyor Drive | Plant C, Line 2 | 2021-10-14 | `past_failure_recent` | H | Running | metropt_derived |
| M-112 | Conveyor Drive | Plant C, Line 2 | 2022-04-05 | `in_maintenance` | M | Maintenance | metropt_derived |

#### Scenario definitions (authoritative)

| Scenario | Behavior |
|---|---|
| `healthy` | No degradation. Vibration stays below 4.5 mm/s |
| `mild_drift` | Over the last 21 days vibration rises gradually from baseline to about 5.0 mm/s and temperature rises about 4 degrees C. Ends in the **warning** band |
| `hero_degrading` | Real MetroPT behavior for history (including real failure windows). Over the final 35 days an exponential-shaped synthetic degradation is added (details in T5). Ends **critical** |
| `past_failure_old` | One failure event about 120 days before the end (precursor 3 days, outage 1 day), healthy since |
| `past_failure_mid` | One failure event about 75 days before the end, healthy since |
| `past_failure_recent` | One failure event about 58 days before the end, healthy since |
| `recently_maintained` | Vibration drifts up to about 4.0 mm/s, then preventive maintenance 5 days before the end brings it back to baseline |
| `overdue_maintenance` | Sensors look normal, but the last preventive maintenance was at least 15 percent beyond the 2,000-hour interval |
| `benign_spikes` | About 6 short vibration spikes per month (up to 3 consecutive intervals, multiplier 1.8). No failure. Serves as a false-positive test for anomaly detection |
| `in_maintenance` | Machine is off for the last 36 hours (near-zero motor current, zero production), a preventive maintenance record is dated `TIMELINE_END` |

**Acceptance**
- `machines.csv` has 12 rows and the columns above, written by `build_machines_df()`.
- Installation dates are all earlier than the sensor window start.

---

### T5: Sensor generation (about 120 min, the largest task)

**Create:** `ml/preprocessing/generate_sensors.py`. Output: `data/raw/unified/sensors_raw.csv` and `data/raw/unified/timeline_metadata.json`.

#### Output columns

| Column | Type | Unit | Provenance |
|---|---|---|---|
| `timestamp` | datetime (UTC, ISO 8601) | | Shifted per D20 |
| `machine_id` | str | | |
| `temperature` | float | degrees C | M-104: real `Oil_temperature`. Others: derived from real |
| `vibration` | float | mm/s RMS | **Synthetic** (formula below) |
| `pressure` | float | bar | M-104: real `TP3`. Others: derived from real |
| `rpm` | float | rev/min | **Synthetic** |
| `energy_consumption` | float | kWh per 5-minute interval | **Derived** from real `Motor_current` |
| `operating_hours` | float | hours (cumulative lifetime) | **Derived** |
| `motor_current` | float | A | **Extension.** Real (M-104) or derived |
| `comp_state` | int (0/1) | | **Extension.** Real (M-104) or derived |

#### Step 1: Load and reduce MetroPT

- Read only `timestamp`, `TP3`, `Oil_temperature`, `Motor_current`, `COMP` (`float32`).
- Resample to the 5-minute grid with the mean (`COMP`: mean greater than or equal to 0.5 gives 1).
- Compute the **time shift**: `shift_days = 7 * floor((TIMELINE_END - max_date).days / 7)`; shifted data ends on or just before `TIMELINE_END`. Save `shift_days`, original start/end, new start/end into `timeline_metadata.json`.
- Apply the same shift to `metropt_failure_windows.csv` start/end times when used.
- Keep genuine gaps in M-104 as missing rows (do not fabricate real data). Log the number of missing intervals.

#### Step 2: Healthy-day pool (for the 11 derived machines)

- Mark each calendar day (original dates) as **healthy** if it has at least 90 percent coverage and no overlap with any failure window extended by 3 days before and 1 day after.
- Require at least 60 healthy days. If fewer, stop and ask.
- For each derived machine and each target date, choose a source day from the pool by seeded random draw, preferring the same weekday. Copy its 288 intervals of `Motor_current`, `COMP`, `TP3`, `Oil_temperature`.
- Apply per-machine fixed transforms (seeded): current scale in [0.93, 1.07], temperature scale in [0.95, 1.05] plus offset in [-4, +4] degrees C, pressure scale in [0.97, 1.03]. Document the drawn values in `timeline_metadata.json`.

#### Step 3: Derived signals (all machines)

Let `L_t = clip(motor_current / I_ref, 0, 1.2)` where `I_ref` is the 95th percentile of the machine's running motor current, and `running_t = motor_current > I_OFF` (from T2). Let `T_ref` = median temperature of the machine.

- **vibration**
  `v_t = 1.6 + 1.2 * L_t + 0.02 * (temperature_t - T_ref) + e_t`
  `e_t` is an AR(1) process, phi = 0.95, stationary standard deviation about 0.2. Clip at 0.3 minimum. When not running, vibration = 0.3 + small noise.
  Per-type base multiplier: Air Compressor 1.0, Hydraulic Press 1.1, Conveyor Drive 0.9.
- **rpm**
  Nominal: Air Compressor 1480, Hydraulic Press 1450, Conveyor Drive 960.
  `rpm_t = nominal * (1 - 0.02 * L_t) + Normal(0, 0.004 * nominal)` when running, else `0`.
- **energy_consumption** (kWh in the interval)
  `0.59 * motor_current * (5 / 60)` (kW per ampere factor for a three-phase motor at about 400 V; document the assumption). Missing current gives missing energy.
- **operating_hours** (cumulative)
  `initial_hours = (window_start - installation_date).days * 24 * util_m` with `util_m` drawn in [0.55, 0.80] (seeded) plus the running time accumulated at 5 / 60 hours per running interval. Monotonically non-decreasing.

#### Step 4: Scenario overlays (applied on top of Step 3 signals, then recompute dependent signals)

Implement one function per overlay, each taking the machine DataFrame and returning it modified.

**4a. Real failure windows (M-104 only).** For each (shifted) window `[ws, we]`:
- Precursor: over the 72 hours before `ws`, ramp a vibration multiplier from 1.0 to 1.6 and a temperature offset from 0 to +3 degrees C (linear).
- Inside the window: vibration multiplier 1.8 and an rpm reduction of 3 percent.
- Record `window_overlay_applied = true` in metadata. (The ground truth for Day 5 is the window itself.)

**4b. Hero degradation (M-104, final 35 days).** Let `s` go from 0 to 1 across the final 35 days and `k = 3`:
`ramp(s) = (exp(k * s) - 1) / (exp(k) - 1)`
- Vibration: `v = v_healthy + (7.8 - v_healthy_mean) * ramp(s)` (additive, so the end of the series approaches about 7.8 mm/s).
- Temperature: `+ 10 * ramp(s)` degrees C.
- Energy: `* (1 + 0.12 * ramp(s))`.
- Pressure: `- 0.15 * ramp(s)` bar (slow decline, small).
- RPM: `* (1 - 0.03 * ramp(s))`.
- Add noise growth: multiply the AR(1) noise by `(1 + 1.5 * ramp(s))`.

**4c. Mild drift (M-103, final 21 days).** Linear ramp: vibration adds up to the amount that brings the weekly mean to about 5.0 mm/s, temperature +4 degrees C.

**4d. Generic failure event** `apply_failure_event(df, center_date, precursor_days=3, outage_hours=24)`: ramp vibration multiplier 1.0 to 1.7 and temperature +3 degrees C during the precursor, then machine off (`motor_current` near zero, `comp_state` 0) for the outage. Used by M-105 (-120 d), M-110 (-75 d), M-111 (-58 d). After the outage, signals return to normal. Return the event timestamp for the maintenance generator.

**4e. Recently maintained (M-106).** Vibration drift up to about 4.0 mm/s over the 30 days before the maintenance date (end minus 5 days), then a drop to baseline immediately after maintenance.

**4f. Benign spikes (M-109).** About 6 spikes per 30 days at seeded random times. Each lasts 1 to 3 intervals, vibration multiplier 1.8.

**4g. In maintenance (M-112).** Last 36 hours: `motor_current` near 0, `comp_state` 0, vibration 0.3, rpm 0, energy near 0.

**4h. Overdue maintenance (M-107).** No sensor overlay.

After overlays, recompute `energy_consumption` (if current changed), `running`, and `operating_hours`.

#### Step 5: Assemble, order, write

- Concatenate machines, sort by `machine_id`, `timestamp`. Write CSV with ISO timestamps, UTC.
- Round: temperature 2, vibration 3, pressure 3, rpm 1, energy 4, operating hours 2, motor current 3 decimals.
- Save to `data/raw/unified/sensors_raw.csv`.

**Acceptance (checked by `validate_raw_outputs.py`, T12)**
- 12 machines, 5-minute grid, last timestamp for each derived machine is `TIMELINE_END` 23:55 UTC.
- For machines `healthy`, `past_failure_*` (outside the failure episode), `overdue_maintenance`, `benign_spikes`: at least 99.5 percent of vibration readings are below 4.5 mm/s.
- M-103: mean vibration of the final 7 days is between 4.5 and 5.8.
- M-104: mean vibration of the last 24 hours is at least 7.0; the first time vibration (hourly mean) crosses 4.5 is between 7 and 14 days before the end; the last 24 hours mean temperature exceeds the machine's healthy 95th percentile by at least 8 degrees C.
- `operating_hours` is non-decreasing per machine.
- Rerun produces identical file hashes.

---

### T6: Maintenance history generation (about 60 min)

**Create:** `ml/preprocessing/generate_maintenance.py`. Output: `data/raw/unified/maintenance_raw.csv`.

#### Output columns

| Column | Type | Notes |
|---|---|---|
| `maintenance_id` | str | `MNT-000001`, sequential after sorting by date |
| `machine_id` | str | |
| `maintenance_date` | date | |
| `maintenance_type` | str | `Inspection`, `Preventive`, `Corrective`, `Predictive` |
| `failure_type` | str | Only for `Corrective`. Otherwise `None` (string) |
| `maintenance_cost` | float | GBP |

#### Rules

1. **Inspections** every 500 operating hours (+/- 10 percent jitter), driven by the machine's cumulative `operating_hours` series. The first event after window start is offset by a seeded random initial state so machines are not synchronized.
2. **Preventive** every 2,000 operating hours (+/- 10 percent jitter), with scenario exceptions:
   - `overdue_maintenance` (M-107): the last preventive was at least 2,300 operating hours before the end.
   - `hero_degrading` (M-104): last preventive at least 2,100 operating hours before the end (overdue) and **none** after that.
   - `recently_maintained` (M-106): a preventive dated `TIMELINE_END - 5 days`.
   - `in_maintenance` (M-112): a preventive dated `TIMELINE_END`.
3. **Corrective** events:
   - M-104: one per real MetroPT failure window (dated the day after the window end), `failure_type = "Air Leak"`.
   - M-105, M-110, M-111: dated the day after the generic failure event outage. Choose `failure_type` by sampling the AI4I failure-mode frequencies (`Tool Wear Failure`, `Heat Dissipation Failure`, `Power Failure`, `Overstrain Failure`, `Random Failure`), seeded.
4. **Inspection on 2026-09-20 for M-104** (fixed record, regardless of the hours rule). This supports the Day 17 inspection-report document.
5. **Costs (GBP, lognormal, seeded, rounded to 2 decimals):**

| Type | Median | Range (approx. P5 to P95) |
|---|---|---|
| Inspection | 250 | 150 to 400 |
| Preventive | 1,500 | 800 to 2,500 |
| Predictive | 900 | 500 to 1,500 |
| Corrective | 6,000 | 2,500 to 15,000 |

6. Records must lie between `window_start - 180 days` (history before the window is allowed so `days_since_maintenance` is defined from the first day) and `TIMELINE_END`.
7. History before the window: simulate backwards using the same rules for 180 days so every machine has a prior maintenance record. Mark nothing special; just dates.

**Acceptance**
- Every machine has at least one maintenance record before the window start (or in the first week of it).
- M-104: exactly one inspection dated 2026-09-20, number of corrective records equals the number of failure windows, no preventive in the last 2,100 operating hours.
- M-107: the last preventive is at least 2,300 operating hours before the end.
- No `failure_type` other than `None` on non-corrective rows.
- Costs are positive and within plausible bounds.

---

### T7: Production and business daily data (about 60 min)

**Create:** `ml/preprocessing/generate_production.py` and `ml/preprocessing/generate_business.py`. Outputs: `production_raw.csv`, `business_daily_raw.csv`.

#### `production_raw.csv` (unified schema)

| Column | Type | Notes |
|---|---|---|
| `timestamp` | date at 00:00 UTC | One row per machine per day, for the sensor window |
| `machine_id` | str | |
| `production_quantity` | int | |
| `target_quantity` | int | |
| `downtime_minutes` | int | 0 to 1,440 |

Generation logic:
- Nominal daily target by type: Air Compressor 1,200, Hydraulic Press 900, Conveyor Drive 1,500; multiply by a per-machine factor in [0.9, 1.1]. Weekend factor 0.6.
- `downtime_minutes` = sum of:
  - base random downtime (exponential, mean 20 minutes, capped at 90),
  - off-time derived from the sensor data (intervals with `running = false` during planned operating windows are not counted as downtime unless they belong to an outage; the outage and failure-window intervals are),
  - failure outage durations (for M-104 real windows: overlap hours of the window with the day, as minutes),
  - maintenance days: Inspection +60 minutes, Preventive +240 minutes, Corrective +480 minutes (the day after failure also counted), Predictive +120 minutes.
  Cap at 1,440.
- `production_quantity = round(target * efficiency * (1 - downtime_minutes / 1440) * noise)` where `efficiency` is 0.93 for healthy, reduced by up to 10 percent during degradation periods (for example M-104 in the last 14 days, M-103 in the last 10 days), and `noise ~ Normal(1, 0.03)`. Never negative.

#### `business_daily_raw.csv` (plan's Business Dataset)

Columns: `date`, `machine_id`, `production`, `orders`, `inventory`, `downtime`, `maintenance_cost`.

- `production` = `production_quantity` from `production_raw.csv`; `downtime` = `downtime_minutes`.
- `maintenance_cost` = sum of that machine's maintenance costs on that date (0 if none).
- `orders` per machine/day = `round(target_quantity * demand_index_d * noise)` where `demand_index_d` is the **Online Retail daily revenue** (after the date shift in T8) divided by its mean, smoothed with a 7-day centered moving average and clipped to [0.6, 1.6]; noise `Normal(1, 0.05)`.
- `inventory` (end-of-day finished goods) follows a stock-flow: `shipped = min(orders, inventory_prev + production)`, `inventory = inventory_prev + production - shipped`. Start inventory = 2,000 per machine. Never negative.

**Acceptance**
- Row count = 12 machines times number of days in the sensor window.
- `business_daily_raw.production` equals `production_raw.production_quantity` for every (machine, date); `downtime` equals `downtime_minutes`.
- Sum of `maintenance_cost` by machine equals the sum of `maintenance_raw.maintenance_cost` inside the window.
- M-104 has materially higher downtime on days within or right after each real failure window than its median day.
- Inventory is never negative; production is never negative.

---

### T8: Online Retail standardization (about 40 min)

**Create:** `ml/preprocessing/standardize_retail.py`. Output: `data/raw/unified/business_transactions_raw.csv`.

Light standardization only. **Do not clean** (cancellations, negative quantities, missing customers stay for Day 3).

| Output column | Source / rule |
|---|---|
| `transaction_id` | `InvoiceNo` + `-` + line number within the invoice (1-based, original row order) |
| `timestamp` | `InvoiceDate`, shifted by `retail_shift_days` (a multiple of 7, chosen so the last date is on or before `TIMELINE_END`) |
| `product_id` | `StockCode` (as string) |
| `quantity` | `Quantity` |
| `unit_price` | `UnitPrice` (GBP) |
| `revenue` | `Quantity * UnitPrice` (negative for returns) |
| `customer_id` | `CustomerID` as string, empty when missing |
| `invoice_no` | **Extension.** `InvoiceNo` (string) |
| `description` | **Extension.** `Description` |
| `country` | **Extension.** `Country` |

Save `retail_shift_days` and original/new date range to `timeline_metadata.json`. Use the cached Parquet from T2 if it exists.

**Acceptance**
- Row count equals the source row count (about 541,909).
- Same number of cancellation rows as in the source (`invoice_no` starting with `C`).
- Weekday of each timestamp equals the original weekday.
- `revenue == quantity * unit_price` for all rows.

---

### T9: M5 subset (optional, about 30 min)

**Create:** `ml/preprocessing/prepare_m5.py`. Skip with a logged warning if M5 files are missing.

- Select store `CA_1`, then the top 30 items by total units sold in `sales_train_validation.csv`.
- Melt the `d_1 ... d_1913` columns into long format, join `calendar.csv` (date, weekday, month, `event_name_1`, `snap_CA`) and `sell_prices.csv` (price by store, item, `wm_yr_wk`).
- Output `data/raw/m5/m5_subset_long.parquet` with columns: `item_id`, `dept_id`, `cat_id`, `store_id`, `date`, `units`, `sell_price`, `event_name_1`, `snap_CA`.
- **Do not time-shift** M5. It is a separate demand-forecasting demo dataset. Note this in the dictionary.

**Acceptance**
- 30 items times 1,913 days = 57,390 rows (record the actual count).
- No melt errors (no `d_` columns left).

---

### T10: AI4I state replay table (stretch, about 40 min, skip if behind)

**Purpose:** provide, for each machine, daily AI4I-style operating snapshots for the last 30 days so the Day 6/7 failure model has a scenario-consistent input. This implements approach (a) of OQ-1 and must be documented as a **replay**, not a derivation from the sensors.

**Create:** `ml/preprocessing/generate_ai4i_state.py`. Output: `data/raw/unified/machine_ai4i_state_raw.csv`.

Columns: `machine_id`, `snapshot_date`, `ai4i_udi`, `type`, `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`.

Buckets of AI4I rows (computed from the CSV):
- **low:** `Machine failure = 0` and below the 75th percentile of both torque and tool wear
- **stressed:** `Machine failure = 0` and above the 75th percentile of both torque and tool wear
- **high:** `Machine failure = 1`

For each machine and each of the last 30 days, sample one row (seeded, with the machine's `type` when possible) from a bucket mixture:

| Scenario | Mixture on the final day | Behavior over the 30 days |
|---|---|---|
| `hero_degrading` | 5 percent low, 25 percent stressed, 70 percent high | Linear change from the healthy mix (94 / 5 / 1 percent low / stressed / high) on day 1 to the final-day mix |
| `mild_drift` | 10 percent high, 40 percent stressed, 50 percent low | Linear from healthy mix |
| `overdue_maintenance` | 3 percent high, 25 percent stressed, 72 percent low | constant |
| All others | 1 percent high, 5 percent stressed, 94 percent low | constant |

Record the mixtures actually used in `docs/data_mapping.md`.

**Acceptance**
- 12 machines times 30 days = 360 rows.
- Every row maps to a valid `UDI` in the AI4I source.
- The fraction of `high` rows for M-104 in the last 7 days is higher than for any other machine.

---

### T11: ORM models and database initialization (about 60 min)

**Create** one model per table in `backend/app/models/` (one file per table, imported in `backend/app/models/__init__.py` so they register on `Base`) and `scripts/init_db.py`.

Use SQLAlchemy 2.0 style (`Mapped`, `mapped_column`), `Base` from `app.database.session`. Timestamps use `DateTime(timezone=True)`.

| Table | Columns | Keys and indexes |
|---|---|---|
| `machines` | `machine_id` (String, PK), `machine_type`, `location`, `installation_date` (Date), `status`, `data_origin`, `scenario`, `ai4i_type` | PK `machine_id` |
| `sensor_readings` | `timestamp`, `machine_id` (FK), `temperature`, `vibration`, `pressure`, `rpm`, `energy_consumption`, `operating_hours`, `motor_current`, `comp_state` | Composite PK (`machine_id`, `timestamp`). Index on `timestamp` |
| `production` | `timestamp` (date), `machine_id` (FK), `production_quantity`, `target_quantity`, `downtime_minutes` | Composite PK (`machine_id`, `timestamp`) |
| `maintenance` | `maintenance_id` (String, PK), `machine_id` (FK), `maintenance_date` (Date), `maintenance_type`, `failure_type`, `maintenance_cost` | Index on (`machine_id`, `maintenance_date`) |
| `business_transactions` | `transaction_id` (String, PK), `timestamp`, `product_id`, `quantity`, `unit_price`, `revenue`, `customer_id` (nullable), `invoice_no`, `description`, `country` | Indexes on `timestamp`, `product_id` |
| `predictions` | `prediction_id` (BigInteger, autoincrement PK), `machine_id` (FK), `timestamp`, `failure_probability` (Float), `risk_level` (String), `model_version` (String) | Index on (`machine_id`, `timestamp`) |
| `anomalies` | `anomaly_id` (BigInteger, autoincrement PK), `machine_id` (FK), `timestamp`, `anomaly_score` (Float), `anomaly_type` (String), `severity` (String) | Index on (`machine_id`, `timestamp`) |
| `ai_insights` | `insight_id` (BigInteger, autoincrement PK), `timestamp`, `machine_id` (FK, nullable), `insight_type`, `explanation` (Text), `recommendation` (Text), `severity` | Index on `timestamp` |

Notes:
- Business-level tables without a machine (`business_transactions`) carry no FK.
- The CSV `date`-only `production.timestamp` is stored as `Date`. Keep the name `timestamp` to match the unified schema.
- Do **not** add Alembic. Use `Base.metadata.create_all()`.

**`scripts/init_db.py`** requirements:
- Arguments: `--drop` (drops all tables first; requires typing `yes` unless `--force` is given).
- Reads the database URL from settings.
- Creates all tables, then uses the SQLAlchemy inspector to print each table and its columns.
- Exits non-zero if the database is unreachable (with a clear message to run `docker compose up -d db`).

**Acceptance**
- `docker compose up -d db && python scripts/init_db.py` creates 8 tables.
- Running it twice does not fail. `--drop --force` recreates them.
- `tests/test_db_schema.py` (skipped automatically if the DB is not reachable) asserts that all 8 tables exist with the expected key columns.
- `ruff check .` passes. `pytest` passes.

---

### T12: Orchestrator, validation, tests, tooling, documentation (about 90 min)

#### 12a. Orchestrator

`scripts/prepare_data.py`: runs in order T4 (machines), T5 (sensors), T6 (maintenance), T7 (production, then business), T8 (retail), T9 (M5, if available), T10 (if implemented). Options: `--only <step>`, `--skip-m5`. Logs the time and row count for each step. Fails fast on errors.

Note on ordering: maintenance depends on sensor `operating_hours` and on failure-event timestamps returned by the sensor step. Persist these event timestamps in `timeline_metadata.json` so the maintenance step does not recompute them.

#### 12b. Validation

`scripts/validate_raw_outputs.py`: runs all acceptance checks listed in T4 to T10, prints a PASS/FAIL table, exits non-zero on any FAIL. Include:

- file existence and column sets (exactly as section 7)
- row-count expectations
- foreign-key integrity (every `machine_id` in sensors, production, maintenance, business_daily exists in `machines.csv`)
- timestamp bounds (no timestamp after `TIMELINE_END`)
- scenario property checks (T5 acceptance list)
- cross-table consistency (T7)
- determinism check: run the generators twice into a temp directory and compare SHA-256 hashes of the sensor and maintenance outputs (slow, behind a `--determinism` flag)

#### 12c. Tests

`tests/test_data.py` (pytest): small, fast tests that run the **generator functions on a tiny synthetic input** (do not require the real datasets) for: AR(1) noise statistics, hero ramp function values (`ramp(0) = 0`, `ramp(1) = 1`, monotonic), failure overlay behavior, maintenance interval logic, stock-flow non-negativity, date-shift weekday preservation. Additional tests that need generated outputs are marked `skipif` when `data/raw/unified/` is empty.

#### 12d. Makefile (TAB-indented recipes)

Add and update targets:

```makefile
verify-sources:
	python scripts/verify_sources.py

inspect-data:
	python scripts/inspect_raw_data.py

data-raw:
	python scripts/prepare_data.py

validate-raw:
	python scripts/validate_raw_outputs.py

db-init:
	python scripts/init_db.py
```

Keep the existing `data:` target (Day 3 will extend it). Add the new names to `.PHONY`.

#### 12e. Documentation

1. **`docs/data_dictionary.md`** (T2) final pass: add all unified output tables (section 7).
2. **`docs/data_mapping.md`**, containing:
   - **Source to unified mapping tables** (below).
   - **Provenance table:** for each column in `sensors_raw.csv`, whether it is real, derived or synthetic.
   - **Timeline shifts** (values from `timeline_metadata.json`).
   - **Fleet and scenarios** (copy the tables from T4).
   - **Open Questions:** OQ-1 (section 4).
   - **Limitations:** public datasets are unrelated; vibration, RPM and most machine behavior are synthetic; only M-104 contains real failure events; AI4I labels are rule-generated.
3. Update `docs/architecture.md`: append D15 to D28 to the Decision Log and replace "Next Steps" with the Day 3 handoff (section 13).
4. Update `README.md` "Data" section with a short description of the synthetic company layer and a link to `docs/data_mapping.md`.
5. Create `docs/day02_walkthrough.md` (agent report).

**Mapping tables to include in `docs/data_mapping.md`**

MetroPT-3 to `sensors_raw`:

| Unified column | MetroPT source | Transformation | Provenance |
|---|---|---|---|
| `timestamp` | `timestamp` | Shift by whole weeks, 5-minute mean grid | real (shifted) |
| `temperature` | `Oil_temperature` | None (M-104). Scale and offset (others) | real / derived |
| `pressure` | `TP3` | None (M-104). Scale (others) | real / derived |
| `motor_current` | `Motor_current` | None (M-104). Scale (others) | real / derived |
| `comp_state` | `COMP` | Threshold 0.5 on the interval mean | real / derived |
| `energy_consumption` | `Motor_current` | `0.59 * I * 5/60` | derived |
| `vibration` | none | Formula in T5 | synthetic |
| `rpm` | none | Formula in T5 | synthetic |
| `operating_hours` | `Motor_current` (running state) | Cumulative running time plus initial hours | derived |

AI4I 2020 (reference for later days):

| AI4I column | Planned use |
|---|---|
| `Type` | `machines.ai4i_type` |
| `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]` | Failure-model features (Day 6) and replay table (T10) |
| `Machine failure` | Failure-model label (Day 6) |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | **Excluded from features** (they compose the label). Used only to sample `failure_type` values for corrective maintenance |

Online Retail to `business_transactions_raw`: as in T8.

---

### T13: Quality-issue injection (stretch, default off, about 30 min)

Only if everything else is done and validated.

**Create:** `ml/preprocessing/inject_quality_issues.py`, run with `python scripts/prepare_data.py --inject-quality-issues`. Seeded. Applies to the **derived** machines only (never M-104's real signals):

- randomly blank 0.5 percent of sensor values (per column independently),
- insert 0.1 percent sensor spikes (value multiplied by 5 to 10),
- duplicate 0.02 percent of rows,
- write every injected issue to `data/raw/unified/quality_issue_log.csv` (`table`, `machine_id`, `timestamp`, `column`, `issue_type`, `original_value`).

The default run (flag off) must not touch this logic. Document the flag in the dictionary. Day 3 can use the log to measure how well its cleaning recovers the original data.

---

### T14: Git, pull request and tag (about 20 min)

See section 14 for the commit sequence.

---

## 7. Output File Contract

All files in `data/raw/unified/` unless stated. CSV, UTF-8, header row, ISO 8601 timestamps in UTC, `.` as decimal separator, no index column.

| File | Columns (in order) | Approx. rows |
|---|---|---|
| `machines.csv` | `machine_id, machine_type, location, installation_date, status, data_origin, scenario, ai4i_type` | 12 |
| `sensors_raw.csv` | `timestamp, machine_id, temperature, vibration, pressure, rpm, energy_consumption, operating_hours, motor_current, comp_state` | about 12 times the number of 5-minute intervals in the window (about 0.7 to 0.9 million) |
| `maintenance_raw.csv` | `maintenance_id, machine_id, maintenance_date, maintenance_type, failure_type, maintenance_cost` | about 100 to 250 |
| `production_raw.csv` | `timestamp, machine_id, production_quantity, target_quantity, downtime_minutes` | 12 times days in the window (about 2,500) |
| `business_daily_raw.csv` | `date, machine_id, production, orders, inventory, downtime, maintenance_cost` | same as production |
| `business_transactions_raw.csv` | `transaction_id, timestamp, product_id, quantity, unit_price, revenue, customer_id, invoice_no, description, country` | about 541,909 |
| `machine_ai4i_state_raw.csv` (stretch) | `machine_id, snapshot_date, ai4i_udi, type, air_temperature_k, process_temperature_k, rotational_speed_rpm, torque_nm, tool_wear_min` | 360 |
| `timeline_metadata.json` | shifts, original/new ranges, per-machine transform parameters, failure event timestamps, `I_OFF` | n/a |
| `data/raw/m5/m5_subset_long.parquet` (optional) | see T9 | about 57,390 |
| `quality_issue_log.csv` (stretch, flag on) | see T13 | varies |

---

## 8. Definition of Done

Day 2 is complete when all of the following are true:

1. `python scripts/verify_sources.py` passes and its output is saved.
2. `docs/data_dictionary.md` documents every source with real numbers from the profile, `COMP` meaning and `I_OFF`.
3. `metropt_failure_windows.csv` exists, matches the PDF, and is committed.
4. `python scripts/prepare_data.py` generates all required files in `data/raw/unified/` from a clean state, with no manual edits.
5. `python scripts/validate_raw_outputs.py` passes every check, and the determinism check passes.
6. M-104 shows the full storyline: real failure windows, final-35-day degradation, vibration above 7.0 in the last 24 hours, preventive maintenance overdue, inspection on 2026-09-20.
7. The fleet snapshot matches the plan's dashboard story: M-101 and M-102 normal, M-103 warning, M-104 critical.
8. `python scripts/init_db.py` creates the 8 tables in PostgreSQL. No data is loaded.
9. `docs/data_mapping.md` contains the mapping, provenance, timeline shifts, fleet, open questions and limitations.
10. `ruff check .` passes. `pytest` passes (DB-dependent tests skipped when no DB).
11. No raw or generated data files are tracked by Git (`git ls-files data | grep -v -E "\.gitkeep|README.md"` returns nothing).
12. `docs/day02_walkthrough.md` exists. The branch is merged into `develop` and tagged `day-02-complete`.

---

## 9. Challenges, Risks and Mitigations

| # | Challenge | Why it matters | Mitigation |
|---|---|---|---|
| 1 | **MetroPT-3 lacks vibration, RPM and energy** | The schema and the M-104 story need them | Synthetic formulas anchored to real motor current, load and temperature. Provenance documented. Never presented as measured |
| 2 | **Only one real machine** | Only M-104 has genuine failure events | Derived machines use healthy-day resampling. Failure episodes are synthetic and labeled. Day 5 evaluates on M-104's real windows only |
| 3 | **Failure windows contaminating derived machines** | Healthy machines would show real failures | Healthy-day pool excludes failure windows plus buffer |
| 4 | **Failure windows must come from the PDF** | Invented dates would invalidate the anomaly evaluation | T3 stop condition. Human supplies if the PDF cannot be parsed |
| 5 | **Meaning of `COMP` and the running threshold** | Wrong assumption breaks operating hours and load features | Determined from the PDF and the histogram in T2, recorded in the dictionary |
| 6 | **Memory and speed (1.5 million rows, Excel with 541,909 rows)** | Crashes or slow runs | Column subsetting, `float32`, Parquet cache for the Excel file, per-machine processing |
| 7 | **Time shifting breaks seasonality** | Weekday patterns carry meaning for production and retail | Shift by whole weeks only. Test weekday preservation |
| 8 | **Synthetic data that looks random** | A dashboard on noise tells no story | Causal chain: degradation to downtime to production loss to maintenance cost. Scenario overlays and consistency tests |
| 9 | **Inconsistent numbers across tables** | Dashboard KPIs disagree | Single source of truth (D25) and cross-table validation |
| 10 | **Operating-hours / maintenance circularity** | Maintenance depends on hours, hours depend on sensors, failure events depend on both | Fixed generation order. Persist event timestamps in metadata |
| 11 | **AI4I has no time or machine dimension** | Failure model cannot learn "failure in next 7 days" | Record OQ-1. Reframe on Day 6 as risk for the current operating state |
| 12 | **AI4I leakage trap** | `TWF`, `HDF`, `PWF`, `OSF`, `RNF` define `Machine failure` | Documented now as excluded features, enforced Day 6 |
| 13 | **Reproducibility drift** | Different results on re-run | Single seed, per-machine RNG, determinism test |
| 14 | **Large files in Git** | Repo bloat, license issues | `.gitignore` rules from Day 1, final `git ls-files` check |
| 15 | **Currency mixing** | Online Retail is GBP | All monetary values in GBP (D21) |
| 16 | **Scope creep** | Day 2 can absorb the whole week | Stretch tasks T10 and T13 are optional. No cleaning or DB loading today |

---

## 10. Suggested Time Plan (about 7 hours)

| Time | Task |
|---|---|
| 0:00 to 0:20 | T1 verify sources and environment |
| 0:20 to 1:20 | T2 profile and dictionary |
| 1:20 to 1:40 | T3 failure windows |
| 1:40 to 2:10 | T4 fleet |
| 2:10 to 4:10 | T5 sensors (largest task) |
| 4:10 to 5:10 | T6 maintenance |
| 5:10 to 6:10 | T7 production and business, T8 retail |
| 6:10 to 7:00 | T9 M5, T11 ORM and init_db |
| Remainder | T12 validation, tests, docs, T14 merge. T10 and T13 only if time remains |

If time runs short, defer in this order: T13, T10, T9, quality of documentation polish. Never skip T3, T5, T6, T7, T11, T12 validation.

---

## 11. Verification Commands (run in order)

```bash
source .venv/bin/activate
python scripts/check_env.py
python scripts/verify_sources.py
python scripts/inspect_raw_data.py
python scripts/prepare_data.py
python scripts/validate_raw_outputs.py --determinism
docker compose up -d db
python scripts/init_db.py
pytest -v
ruff check .
git status --short                     # no data or .env files listed
git ls-files data | grep -v -E "\.gitkeep|README.md"   # must print nothing
```

---

## 12. What NOT to Do on Day 2

- Do not clean, impute, remove outliers or resample for modeling (Day 3).
- Do not compute rolling or lag features (Day 3).
- Do not load data into PostgreSQL (Day 3).
- Do not train any model or write evaluation code (Days 5 and 6).
- Do not write real API endpoints (Day 7).
- Do not create the RAG documents (Day 17).
- Do not download other datasets (C-MAPSS, SECOM, Gas Turbine, MIMII).
- Do not modify raw source files.
- Do not guess the MetroPT failure windows.
- Do not hardcode dataset statistics quoted in this document. Use the profile output.

---

## 13. Handoff to Day 3

Add to `docs/architecture.md` under "Next Steps":

1. **Day 3 goal:** build the data engineering pipeline in `ml/preprocessing/`: ingestion, validation (schema, ranges, duplicates, timestamp order), missing-value handling, outlier handling that **keeps real anomalies**, integration of sensors, maintenance and production, feature engineering (rolling statistics, deltas, `days_since_maintenance`, `maintenance_count`, `failure_count`, `previous_failure`), output of `operations_dataset.csv` and Parquet files in `data/processed/`, and loading into PostgreSQL.
2. **Inputs ready:** everything in `data/raw/unified/` plus `timeline_metadata.json`, `metropt_failure_windows.csv` and the documented `I_OFF`.
3. **Known cleaning targets:** gaps in M-104, cancellations and negative quantities and missing customers in the retail transactions, duplicate rows (only if T13 was run).
4. **Branch:** `feature/day-03-data-pipeline`.

---

## 14. Git Workflow

### Commit sequence (Conventional Commits)

| After | Message |
|---|---|
| T1 | `chore(data): add config, source verification script and reference folder` |
| T2 | `docs(data): add raw data profiling script and data dictionary` |
| T3 | `feat(data): add MetroPT failure window reference file` |
| T4 | `feat(data): define 12-machine fleet and scenarios` |
| T5 | `feat(data): generate unified sensor dataset with scenario overlays` |
| T6 | `feat(data): generate maintenance history` |
| T7 | `feat(data): generate production and daily business datasets` |
| T8 | `feat(data): standardize online retail transactions` |
| T9 | `feat(data): add M5 subset preparation` (skip if not run) |
| T10 | `feat(data): add AI4I state replay table` (skip if not run) |
| T11 | `feat(db): add ORM models and database init script` |
| T12 | `test(data): add validation script and unit tests`, then `docs: add data mapping, update architecture and README` |
| T13 | `feat(data): add optional quality issue injection` (skip if not run) |
| End | `docs: add Day 2 walkthrough` |

### Merge and tag

```bash
git push -u origin feature/day-02-data-sources
gh pr create --base develop --head feature/day-02-data-sources \
  --title "Day 2: Data sources, unified schema and raw datasets" \
  --body "See docs/day02_walkthrough.md"
# after review and merge:
git checkout develop && git pull
git tag -a day-02-complete -m "Day 2: raw datasets and unified schema"
git push origin day-02-complete
git checkout -b feature/day-03-data-pipeline
```

Before merging, check the PR diff for: no data files, no secrets, no hardcoded absolute paths, no leftover debugging output.
