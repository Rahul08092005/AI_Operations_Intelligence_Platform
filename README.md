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
                    ┌───────────────────────────────────────────────────┐
                    │ PostgreSQL 16 Relational Schema / SQLAlchemy 2.0  │
                    └───────────────────────────────────────────────────┘
```

---

## Quickstart & Data Pipeline

### Prerequisites
- Python 3.11+ (Python 3.14 compatible)
- Docker & Docker Compose (optional for PostgreSQL; SQLite fallback supported)

### 1. Verify Source Archives
Inspect the presence and schema of source datasets:
```bash
python scripts/verify_sources.py
# or: make verify-sources
```

### 2. Run Master Data Preparation Pipeline
Generates all 7 standardized operational datasets with deterministic seed (`SEED=42`):
```bash
python scripts/prepare_data.py --determinism
# or: make data-raw
```

Available pipeline flags:
- `--target <step>`: Run a single step (`verify_sources`, `machines`, `sensors`, `maintenance`, `production`, `retail`, `business`, `ai4i`, `validate`, or `all`).
- `--skip-existing`: Skip generation if target outputs already exist.
- `--determinism`: Verifies byte-level SHA-256 reproducibility across runs.

### 3. Run Output Validation Suite
Asserts row counts, foreign key integrity, temporal bounds, and operational scenario constraints:
```bash
python scripts/validate_raw_outputs.py
# or: make validate-raw
```
A human-readable summary report is written to `docs/raw_data_validation_report.txt`.

### 4. Database Initialization
Initialize all 8 relational tables:
```bash
# PostgreSQL (requires running container: docker compose up -d db)
python scripts/init_db.py --load-data

# SQLite offline mode:
python scripts/init_db.py --sqlite-fallback --load-data
```

### 5. Run Test Suite
```bash
pytest tests/ -v
# or: make test
```

---

## Documentation Index

- [Data Dictionary](docs/data_dictionary.md): Field-level descriptions, units, and valid ranges for all operational tables.
- [Data Mapping Specification](docs/data_mapping.md): Transformation rules, whole-week time shifts, and synthetic signal derivations.
- [Architecture Decision Records](docs/architecture.md): Formal design decisions (D01 through D28).
- [Day 2 Walkthrough Report](docs/day02_walkthrough.md): Comprehensive Day 2 deliverables, verification results, and acceptance checklists.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
