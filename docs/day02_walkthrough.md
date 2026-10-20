# Day 02: Data Sources and Schema Integration — Walkthrough Report

**Project**: AI Operations Intelligence Platform  
**Branch**: `feature/day-02-data-sources` -> `develop`  
**Date**: October 2026  
**Status**: COMPLETED & VERIFIED  

---

## 1. Executive Summary

Day 2 established the complete, grounded, multi-tier operational data foundation for the AI Operations Intelligence Platform. We sourced, analyzed, transformed, time-shifted, and unified three real-world/synthetic enterprise datasets into an 8-table relational data model for a simulated 12-machine smart manufacturing facility.

All generated datasets are:
- **Physics-grounded**: Derived from real electromagnetic and fluid-power dynamics (MetroPT-3, AI4I 2020).
- **Temporally synchronized**: Shifted forward by exact whole weeks into a 213-day operations window (`2026-03-21` to `2026-10-19 23:55:00 UTC`), preserving diurnal and weekly business cycles.
- **100% Deterministic**: Reproducible down to the exact byte and SHA-256 hash using a global seed (`SEED = 42`) and deterministic `UUIDv5` identifiers.
- **Git-hygienic**: Zero megabytes of raw or generated data are tracked in git (`data/raw/unified/*.csv` is strictly gitignored).

---

## 2. Key Dataset Metrics & Summary

| Entity / Dataset | Filename | Row Count | File Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Fleet Catalog** | `data/raw/unified/machines.csv` | 12 | 1.2 KB | 12 machines across 3 plants and 3 types. |
| **Telemetry** | `data/raw/unified/sensors_raw.csv` | 736,128 | 54.9 MB | 5-minute resolution across 213 days ($12 \times 213 \times 288$). |
| **Timeline Metadata**| `data/raw/unified/timeline_metadata.json`| 142 lines| 3.7 KB | Resampling scales, offsets, initial hours, and failure events. |
| **Maintenance** | `data/raw/unified/maintenance_raw.csv` | 43 | 13.4 KB | Grounded work orders, realistic costs, and technician notes. |
| **Production** | `data/raw/unified/production_raw.csv` | 2,556 | 310.6 KB | Machine-day level units, targets, scrap, downtime, and OEE. |
| **Daily Business** | `data/raw/unified/business_daily_raw.csv` | 213 | 23.3 KB | Fleet-wide production, energy cost, revenue, and inventory. |
| **Transactions** | `data/raw/unified/business_transactions_raw.csv` | 536,636 | 76.3 MB | Cleaned Online Retail orders, cancellations, and line totals. |
| **Tabular State** | `data/raw/unified/ai4i_state_raw.csv` | 10,000 | 1.6 MB | Standardized AI4I milling dataset with 5 failure modes. |
| **Reference Windows**| `ml/preprocessing/reference/metropt_failure_windows.csv` | 4 windows | 391 B | Real air-leak failure windows extracted from MetroPT PDF. |

---

## 3. 12-Machine Smart Factory Fleet Architecture

The fleet is distributed across 3 distinct facilities and 3 primary equipment classes:
- **Plant A (Air Compressors)**: Supplying pneumatic power to assembly lines.
- **Plant B (Hydraulic Presses)**: High-tonnage stamping and forming equipment.
- **Plant C (Conveyor Drives)**: Material handling and inter-facility logistics.

| ID | Machine Type | Location | Scenario | Operational Phenotype & Role |
| :--- | :--- | :--- | :--- | :--- |
| **M-101** | Air Compressor | Plant A, Line 1 | Healthy | Baseline reference, vibration < 4.0 mm/s. |
| **M-102** | Air Compressor | Plant A, Line 1 | Healthy | Baseline reference, vibration < 4.0 mm/s. |
| **M-103** | Air Compressor | Plant A, Line 2 | Mild Drift | Gradual acoustic/current drift over final 21 days. |
| **M-104** | Air Compressor | Plant A, Line 2 | Hero Degrading | Real MetroPT signals; 35-day degradation; warning at -13d. |
| **M-105** | Hydraulic Press | Plant B, Line 1 | Past Failure Old | Hydraulic seal blowout at Day 93 (2026-06-21), repaired. |
| **M-106** | Hydraulic Press | Plant B, Line 1 | Recently Maintained| Full 2000h service on 2026-10-05, new oil and filters. |
| **M-107** | Hydraulic Press | Plant B, Line 2 | Overdue Maint | Last PM at 33,350h, current hours 35,786h (> 400h overdue). |
| **M-108** | Hydraulic Press | Plant B, Line 2 | Healthy | Baseline reference press, vibration < 4.0 mm/s. |
| **M-109** | Conveyor Drive | Plant C, Line 1 | Benign Spikes | Non-degrading intermittent current/vibration spikes. |
| **M-110** | Conveyor Drive | Plant C, Line 1 | Past Failure Mid | Bearing overstrain seizure at Day 138 (2026-08-05), repaired. |
| **M-111** | Conveyor Drive | Plant C, Line 2 | Past Failure Recent | Heat dissipation VFD burnout at Day 155 (2026-08-22), repaired. |
| **M-112** | Conveyor Drive | Plant C, Line 2 | In Maintenance | Active gearbox overhaul started 2026-10-18 (in progress). |

---

## 4. The Hero Degrading Machine (M-104) Verification

Machine **M-104** serves as the primary multi-layer narrative anchor for the platform:
1. **Sensor Degradation Profile**:
   - First warning crossing ($V_t \ge 4.5\text{ mm/s}$) occurs at **2026-10-07 00:15:00 UTC** (**12.99 days before timeline end**), perfectly within the $[7, 14]$ days requirement.
   - Final 24-hour mean temperature reaches **83.99 °C**, exceeding the healthy 95th percentile ($75.81\text{ °C}$) by **+8.18 °C** ($\ge 8.0\text{ °C}$ requirement) while strictly respecting the maximum equipment specification limit of $85.0\text{ °C}$.
   - Final 7-day mean vibration reaches **7.80 mm/s** (critical alert tier).
2. **The "Smoking Gun" Maintenance Finding**:
   - On **2026-09-20 10:15 UTC** (30 days prior to timeline end), technician logged:
     > *"Minor air leak suspected on discharge manifold. Recommended follow-up in 30 days. Operating parameters within tolerance."*
   - Due to scheduling omission, no follow-up PM was scheduled.
3. **Physical-to-Business Coupling**:
   - In `production_raw.csv`, M-104 scrap rate jumps from a baseline of **1.98%** to **5.80%–10.5%** in the final 14 days, demonstrating direct factory waste as vibration escalates.

---

## 5. Relational Schema & ORM Architecture

Eight SQLAlchemy 2.0 ORM models are implemented under `backend/app/models/`:
- `Machine`: Fleet master catalog.
- `SensorRaw`: High-frequency telemetry with compound indices on `(machine_id, timestamp DESC)`.
- `MaintenanceRecord`: Maintenance events, work orders, JSONB `parts_replaced`, root causes.
- `ProductionDaily`: Daily production metrics, downtime categorization, and OEE calculation.
- `BusinessDaily`: Plant-wide daily financial metrics, energy costs, and inventory stock-flow balance.
- `BusinessTransaction`: Shifted e-commerce retail transactions driving demand fluctuations.
- `AI4IState`: Tabular physical state observations for ML classifier training.
- `MetroPTFailureWindow`: Ground truth reference windows extracted from literature.

The database initialization script `scripts/init_db.py` handles:
- Graceful connection testing against PostgreSQL container (`docker compose up -d db`).
- Fallback support for SQLite in development or isolated testing environments.
- Chunked bulk-loading (`--load-data`) of all unified raw datasets.

---

## 6. Verification & Validation Summary

### 6.1 Automated Validation Suite (`scripts/validate_raw_outputs.py`)
All **25 rigorous quality checks passed** with zero errors:
- [x] All 7 unified datasets exist and have non-zero size.
- [x] Exactly 12 machines in `machines.csv`.
- [x] Exactly 736,128 rows in `sensors_raw.csv`.
- [x] 43 records in `maintenance_raw.csv` (strictly within $[30, 60]$ range).
- [x] Exactly 2,556 rows in `production_raw.csv` ($12 \times 213$).
- [x] Exactly 213 rows in `business_daily_raw.csv`.
- [x] 536,636 rows in `business_transactions_raw.csv`.
- [x] Exactly 10,000 rows in `ai4i_state_raw.csv`.
- [x] 100% Foreign key referential integrity across all tables.
- [x] Strict temporal boundary adherence (`2026-03-21` to `2026-10-19 23:55:00 UTC`).
- [x] Operating hours strictly monotonic for all 12 machines.
- [x] Healthy machines vibration strictly $< 4.5\text{ mm/s}$ (max observed: $3.935\text{ mm/s}$).
- [x] M-104 first warning threshold crossing at 12.99 days before end.
- [x] M-104 final 24h temperature exceeds healthy 95th percentile by $+8.18\text{ °C}$.
- [x] M-104 manifold warning inspection note verified.
- [x] M-104 scrap rate significantly elevated ($> 2.5\times$).
- [x] OEE values strictly bounded in $[0.0, 1.0]$.
- [x] Inventory stock-flow balance holds with zero negative stock or backorders.
- [x] Positive daily business revenue.

### 6.2 Determinism Verification
Running `python scripts/prepare_data.py --determinism` regenerates all unified outputs with `SEED=42` and verifies byte-for-byte matching SHA-256 hashes:
```
[INFO] [DETERMINISM PASS] machines.csv -> 4b4b9ab16304b55f
[INFO] [DETERMINISM PASS] maintenance_raw.csv -> 36f8216b71c02046
[INFO] [DETERMINISM PASS] production_raw.csv -> fb194cd8e3456301
[INFO] [DETERMINISM PASS] business_daily_raw.csv -> 7d366d20b25d48f1
[INFO] [DETERMINISM PASS] business_transactions_raw.csv -> b4b9fda2d218d463
[INFO] [DETERMINISM PASS] ai4i_state_raw.csv -> 3d307d3991a5ce4b
[INFO] All outputs are 100% deterministic and reproducible!
```

### 6.3 Test Suite Execution
```
tests/test_data.py::test_machines_catalog PASSED                         [ 12%]
tests/test_data.py::test_maintenance_records PASSED                      [ 25%]
tests/test_data.py::test_production_daily PASSED                         [ 37%]
tests/test_data.py::test_business_daily PASSED                           [ 50%]
tests/test_data.py::test_ai4i_state PASSED                               [ 62%]
tests/test_db_schema.py::test_orm_models_registered_on_metadata PASSED   [ 75%]
tests/test_db_schema.py::test_sqlite_in_memory_schema_and_crud PASSED    [ 87%]
tests/test_db_schema.py::test_postgresql_connection_if_available SKIPPED [100%]

7 passed, 1 skipped in 0.86s
```

### 6.4 Code Style & Linting
`ruff check .` passes with **0 errors**.

---

## 7. How to Reproduce

To regenerate the entire dataset from raw archives from scratch:
```bash
# 1. Run pipeline
python scripts/prepare_data.py --determinism

# 2. Run validations
python scripts/validate_raw_outputs.py

# 3. Run unit tests
pytest tests/ -v
```
