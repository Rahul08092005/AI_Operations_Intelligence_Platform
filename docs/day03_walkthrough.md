# Day 03: Data Engineering Pipeline & Feature Store — Walkthrough Report

**Project:** AI Operations Intelligence Platform  
**Branch:** `feature/day-03-data-pipeline`  
**Execution Date:** October 2026  
**Pipeline Status:** COMPLETE & VERIFIED (Zero Validation Errors, 100% Anomaly Preservation)  

---

## 1. Executive Summary

Day 3 delivered the production-grade Data Engineering Pipeline and Feature Store for the AI Operations Intelligence Platform. The pipeline transforms raw industrial telemetry (MetroPT air compressor real signals and physical derivations), enterprise maintenance logs, factory floor daily production records, and UCI Online Retail transactions into two integrated tabular datasets:
1. `sensor_features.parquet`: High-frequency (5-minute resolution) feature store with 89 engineered temporal features (trailing rolling stats, least-squares slopes, operating context, and $D+1$ strictly past maintenance indicators) designed for unsupervised anomaly detection (Day 5).
2. `operations_dataset.parquet` and `operations_dataset.csv`: Daily machine-level operational dataset combining equipment sensor profiles, maintenance status, production yield/scrap, and retail demand signals designed for exploratory data analysis (Day 4) and dashboard visualization (Day 9).

All transformations strictly enforce temporal leakage prevention rules (L1–L5), non-destructive outlier handling (D30), bounded short-gap imputation ($\le 15$ min, D31), ground-truth label isolation (D33), atomic Parquet persistence (D34), and single-transaction database loading (D36).

---

## 2. Deliverables & Created Files

### 2.1 Core Pipeline Modules (`ml/preprocessing/`)
- `config.py`: Centralized configuration constants (`ROLLING_WINDOWS`, `STAT_OUTLIER_ROBUST_Z=6.0`, `MAX_IMPUTE_INTERVALS=3`, `WARMUP_INTERVALS=288`, `RETAIL_EXTREME_QTY=10000`, `PROCESSED`).
- `validation_rules.py`: Typed physical limit boundaries (`SENSOR_LIMITS`, `PRODUCTION_LIMITS`, `MAINTENANCE_LIMITS`) and non-product stock code regex.
- `ingest.py`: Memory-tracked, typed ingestion functions for all 7 unified datasets with `--sample` mode support.
- `validate.py`: Validation suite for pre-cleaning and post-cleaning verification (checking duplicates, bounds, monotonicity, and foreign keys).
- `clean_sensors.py`: 5-minute grid reindexing, Tier 1 physically impossible nullification, bounded short-gap imputation, and first 24h warmup flagging.
- `clean_other.py`: Cleaning and normalization for machines catalog, maintenance records, production daily metrics, and business daily totals.
- `outliers.py`: Tier 2 trailing 24h robust $z$-score flagging and automated 4-point anomaly preservation test suite.
- `clean_transactions.py`: Retail transaction cleaning, non-product line and extreme quantity flagging, and separate rejection table generation.
- `labels.py`: Ground-truth evaluation label generator (`anomaly_eval_labels.parquet`) isolating failure windows and precursor periods from feature tables.
- `features_sensor.py`: 89 engineered features across rolling statistics (1h, 6h, 24h), rate-of-change, least-squares slopes, and strictly past maintenance as-of joins ($D+1$).
- `features_business.py`: Daily retail KPIs (sales, orders, customers, average order value, sales growth) and product demand across top 50 SKUs.
- `integrate.py`: Complete-day (288-interval) daily sensor aggregation and cross-domain integration into `operations_dataset.parquet` and `operations_dataset.csv`.
- `report.py`: Automated quality audit report generator outputting structured JSON and Markdown documents.
- `pipeline.py`: Main pipeline orchestrator executing all 11 stages with atomic file writes, timing tracking, and failure assertion.

### 2.2 Entry Points & Tooling
- `scripts/run_pipeline.py`: Production CLI entry point supporting `--sample`, `--skip-db`, `--only <step>`, and `--log-level`.
- `scripts/load_to_db.py`: PostgreSQL `COPY` loader inside a single atomic transaction with `--dry-run` and count verification.
- `tests/test_pipeline.py`: Comprehensive test suite containing 14 unit and integration tests (including temporal leakage, idempotence, and schema validation).
- `Makefile`: Standard developer targets (`pipeline`, `pipeline-sample`, `db-load`, `data`, `test`, `lint`, `format`).

### 2.3 Documentation
- `docs/data_pipeline.md`: End-to-end pipeline architecture specification, Mermaid workflow diagram, and leakage rules.
- `docs/data_quality_report.md`: Automated audit report documenting table row counts, imputation counts, outlier rates, and validation results.
- `docs/feature_catalog.md`: Complete reference catalog of 89 engineered features, mathematical formulations, and leakage classifications.
- `docs/data_dictionary.md`: Section 6 appended documenting schemas, data types, and nullability contracts for all processed files.
- `docs/architecture.md`: Architecture Decision Records updated with D29 through D41 and Day 4 handoff.
- `README.md`: Updated with one-command data flow and links to Day 3 specifications.

---

## 3. Pipeline Execution Metrics & Row Counts

The full fleet pipeline executed across 12 machines and 214 calendar days in **54.56 seconds**:

| Step | Target Table | Rows In | Rows Out | Time (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1_ingest** | `sensors_raw.csv` | 736,128 | 736,128 | 2.06s | Loaded 81.85 MB retail, 28.78 MB sensors |
| **2_validate_before** | All raw tables | 736,128 | 736,128 | 0.86s | 25/25 checks passed |
| **3_clean_sensors** | `sensors.parquet` | 736,128 | 736,128 | 0.62s | 1 short gap imputed, 3,456 warmup rows |
| **3b_clean_other** | Maintenance, Production, Biz | 2,556 | 2,556 | 0.01s | Normalized failure types & production bounds |
| **4_outliers** | `sensors_flagged` | 736,128 | 736,128 | 3.46s | 240,418 rows flagged (|Z| > 6.0), 0 modified |
| **5_transactions** | `business_transactions` | 536,636 | 536,636 | 0.17s | 8,706 cancellations flagged, 4 extreme qty |
| **6_labels** | `anomaly_eval_labels` | 736,128 | 736,128 | 1.27s | Isolated targets for M-104 and synthetic episodes |
| **7_sensor_features** | `sensor_features.parquet` | 736,128 | 736,128 | 12.17s | 89 features computed (float32) |
| **8_operations_dataset** | `operations_dataset` | 736,128 | 2,556 | 15.82s | Daily aggregation across complete days (D40) |
| **9_validate_after** | All cleaned tables | 736,128 | 736,128 | 2.14s | 25/25 checks passed, 0 errors remaining |
| **10_atomic_writes** | 12 Parquet & CSV files | 736,128 | 736,128 | 15.25s | Atomic rename via .tmp buffers |
| **11_reports** | `quality_report` | 2,556 | 2,556 | 0.73s | JSON & Markdown quality reports written |
| **TOTAL** | **Full Fleet Pipeline** | **736,128** | **2,556** | **54.56s** | **SUCCESS** |

---

## 4. Key Quality & Audit Highlights

1. **Non-Destructive Cleaning Verification (Preservation Suite):**
   - **Zero Real Data Destroyed:** In failure windows for `M-104`, exactly 0 sensor readings were nullified (`in_fw_faults = 0`).
   - **Peak Temperature Preservation:** `raw_max = 89.05 °C`, `clean_max = 89.05 °C` (difference: 0.000 °C).
   - **Peak Vibration Preservation:** `raw_max = 8.167 mm/s`, `clean_max = 8.167 mm/s` (difference: 0.000 mm/s).
   - **Benign Spikes Preserved:** Benign vibration surges on `M-109` match identically between raw and cleaned frames.
2. **Retail Audit Breakdown:**
   - 536,636 raw transactions ingested.
   - 8,706 cancellations and returns flagged.
   - 131,626 guest checkout records flagged (`customer_id` is null).
   - 207 administrative non-product lines flagged (`POST`, `D`, `CRUK`, etc.).
   - Exactly 4 extreme quantity transactions flagged ($\ge 10,000$ units), matching the known wholesale order pairs (Invoices 541431/C541433 and 581483/C581484).
3. **Temporal Leakage Audit Results:**
   - `labels_in_sensor_features`: **PASSED** (0 leaked label columns).
   - `labels_in_operations_dataset`: **PASSED** (0 leaked label columns).
   - `forbidden_metadata_in_operations`: **PASSED** (`status`, `scenario`, `data_origin`, `ai4i_type` absent).
   - `operations_dataset_unique_keys`: **PASSED** (0 duplicate `(machine_id, date)` keys).
   - `maintenance_strictly_past`: **PASSED** (all valid `days_since_maintenance` $\ge 1.0$).

---

## 5. Fleet Storyline Sanity Check Results

Snapshot of fleet health and narrative anchors on the final timeline day (`2026-10-19`):

| Machine ID | Scenario / Role | 24h Vibration Mean | Operating Hours | Days Since PM | Overdue Flag | Storyline Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M-104** | Hero Degrading (Air Leak) | **7.75 mm/s** | 2,348.5 h | 29.0 d | **1.0 (Overdue)** | **CRITICAL** (Active air leak degradation, inspection anchor 29d prior) |
| **M-103** | Warning Band | **4.78 mm/s** | 2,120.4 h | 53.0 d | 0.0 | **WARNING** (Vibration in warning threshold [4.5, 7.0]) |
| **M-105** | Hydraulic Press Drift | 2.21 mm/s | 2,050.2 h | 48.0 d | 0.0 | Drifting baseline |
| **M-108** | Hydraulic Press Healthy | 1.85 mm/s | 1,980.0 h | 52.0 d | 0.0 | Normal operation |
| **M-106** | Recently Maintained | **1.75 mm/s** | 1,840.1 h | **14.0 d** | 0.0 | **HEALTHY** (Overhauled on 2026-10-05) |
| **M-102** | Air Compressor Healthy | 1.61 mm/s | 1,910.3 h | 57.0 d | 0.0 | Normal operation |
| **M-110** | Conveyor Drive Past Fail | 1.35 mm/s | 1,750.0 h | 49.0 d | 0.0 | Stable post-repair |
| **M-101** | Air Compressor Baseline | 1.24 mm/s | 1,890.5 h | 53.0 d | 0.0 | Reference healthy asset |
| **M-107** | Hydraulic Press Overdue | 1.19 mm/s | 2,410.8 h | 56.0 d | **1.0 (Overdue)** | Cumulative hours > 2,000h PM threshold |
| **M-111** | Conveyor Drive Healthy | 0.85 mm/s | 1,620.0 h | 52.0 d | 0.0 | Normal operation |
| **M-109** | Conveyor Drive Spikes | 0.83 mm/s | 1,690.4 h | 52.0 d | 0.0 | Benign operational spikes preserved |
| **M-112** | Overhaul Outage | **0.30 mm/s** | 1,510.0 h | **1.0 d** | 0.0 | **MAINTENANCE** (1,440 min downtime, overhaul in progress) |

---

## 6. Test Suite & Verification Results

All automated verification commands passed without warnings or regressions:

1. **Raw Validation Suite:**
   ```bash
   python scripts/validate_raw_outputs.py
   # Result: 37/37 checks PASSED
   ```
2. **Automated Unit & Integration Tests:**
   ```bash
   pytest -v
   # Result: 21 passed, 1 skipped (PostgreSQL service not running) in 41.46s
   ```
3. **Code Style & Linting:**
   ```bash
   ruff check .
   # Result: All checks passed!
   ```
4. **Git Hygiene & Zero Data in Git:**
   ```bash
   git status --short
   git ls-files data | Select-String -NotMatch "\.gitkeep|README\.md"
   # Result: Zero data files tracked. data/processed/* strictly ignored.
   ```

---

## 7. Deviations from Specification & Rationale

1. **PostgreSQL Connectivity:** Docker Desktop was not active on the test machine. `load_to_db.py` gracefully tested and verified `--dry-run` logic and was configured with `--skip-db` as a first-class flag in the pipeline and Makefile.
2. **`is_sampled` Auto-Detection in `validate.py`:** Updated `validate.py` to evaluate `len(df_prod["machine_id"].unique()) < 12` so that `--sample` mode (2 machines) correctly skips the fleet-wide business daily consistency check without throwing false error flags.
3. **Retail Extreme Quantity Count:** Documented 4 extreme quantity rows ($\ge 10,000$) instead of 2. Investigation revealed two separate wholesale order/cancellation pairs in Online Retail (Invoices 541431/C541433 with 74,215 units, and Invoices 581483/C581484 with 80,995 units). All 4 rows are correctly flagged.

---

## 8. Handoff to Day 04

The feature store and integrated dataset are fully prepared for Day 4: Exploratory Data Analysis (EDA):
- Target Branch: `feature/day-04-eda`
- Core Data Artifact: `data/processed/operations_dataset.csv`
- Primary Goal: Author exploratory notebooks in `ml/notebooks/`, extract 8–10 diagnostic figures into `docs/figures/`, and document analytical findings in `docs/eda_findings.md`.
