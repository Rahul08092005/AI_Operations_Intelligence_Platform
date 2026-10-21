# AI Operations Intelligence Platform

An enterprise-grade operational intelligence platform combining real-time anomaly detection, predictive maintenance, demand forecasting, automated root-cause analysis, and an interactive RAG-powered knowledge copilot.

---

## Architecture Overview

```
                               ┌────────────────────────────────┐
                               │  Smart Factory Fleet (12 M/Cs) │
                               └──────────────┬─────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
        ┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
        │   Plant A (Air Comp)  │ │ Plant B (Hyd. Press)  │ │ Plant C (Conv. Drive) │
        │   M-101 .. M-104      │ │ M-105 .. M-108        │ │ M-109 .. M-112        │
        └───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              ▼
                         ┌────────────────────────────────────────┐
                         │ Unified Operational Data Layer (Raw)   │
                         │  - machines.csv (12 fleet units)       │
                         │  - sensors_raw.csv (736,128 intervals) │
                         │  - maintenance_raw.csv (43 work orders)│
                         │  - production_raw.csv (2,556 days)     │
                         │  - business_daily_raw.csv (213 days)   │
                         │  - business_transactions_raw.csv       │
                         │  - ai4i_state_raw.csv (10k rows)       │
                         └────────────────────┬───────────────────┘
                                              ▼
                         ┌────────────────────────────────────────┐
                         │ Day 3 Feature Store & Cleaning Engine  │
                         │  - sensors.parquet (Tier 1 & Tier 2)   │
                         │  - sensor_features.parquet (89 feats)  │
                         │  - operations_dataset.parquet & .csv   │
                         │  - anomaly_eval_labels.parquet         │
                         │  - business_features_daily.parquet     │
                         │  - product_demand_daily.parquet        │
                         └────────────────────┬───────────────────┘
                                              ▼
                    ┌───────────────────────────────────────────────────┐
                    │ PostgreSQL 16 Relational Schema / SQLAlchemy 2.0  │
                    └───────────────────────────────────────────────────┘
```

---

## Quickstart & One-Command Data Flow

The complete data engineering and ingestion workflow can be executed with a single command:

```bash
# Complete end-to-end data build: prepare raw, run Day 3 pipeline, load database
make data
```

### Modular Pipeline Targets

```bash
# 1. Fast sample run (machines M-103 & M-104, trailing 45 days, ~10s runtime)
make pipeline-sample

# 2. Full fleet pipeline (12 machines, 214 days, outputs all 14 processed files)
make pipeline

# 3. Load processed data into PostgreSQL database via atomic COPY transaction
make db-load

# 4. Verify output raw datasets against integrity constraints
make validate-raw

# 5. Run full test suite (22 unit, integration, and leakage tests)
make test

# 6. Run linting and code quality checks
make lint
```

### Python CLI Invocations

```bash
# Activate Python 3.11+ virtual environment
# Windows: .\.venv\Scripts\Activate.ps1 | Linux/macOS: source .venv/bin/activate

# Run pipeline in sample mode
python scripts/run_pipeline.py --sample --skip-db

# Run full fleet pipeline
python scripts/run_pipeline.py --skip-db

# Database loader (dry-run mode)
python scripts/load_to_db.py --dry-run
```

---

## Documentation Index

- [Data Engineering Pipeline Specification](docs/data_pipeline.md): End-to-end pipeline architecture, leakage prevention rules (L1–L5), and step execution policies.
- [Data Quality & Audit Report](docs/data_quality_report.md): Automated quality metrics, missing value imputation logs, outlier rates, and anomaly preservation verification.
- [Feature Catalog](docs/feature_catalog.md): Complete catalog of 89 engineered sensor features, mathematical definitions, rolling windows, and leakage classifications.
- [Data Dictionary](docs/data_dictionary.md): Field-level descriptions, data types, units, and nullability contracts for raw and processed operational tables.
- [Data Mapping Specification](docs/data_mapping.md): Transformation rules, whole-week time shifts, and synthetic signal derivations.
- [Architecture Decision Records](docs/architecture.md): Formal design decisions (D01 through D41) and Day 4 handoff.
- [Day 2 Walkthrough Report](docs/day02_walkthrough.md): Comprehensive Day 2 deliverables, verification results, and acceptance checklists.
- [Day 3 Walkthrough Report](docs/day03_walkthrough.md): Comprehensive Day 3 data pipeline deliverables, sanity checks, and benchmark execution metrics.
