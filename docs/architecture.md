# Architecture Decision Records (ADRs)

**Project**: AI Operations Intelligence Platform  
**Status**: APPROVED / ACTIVE  
**Last Updated**: October 2026  

---

## Record Index

| ADR ID | Title | Date | Status |
| :--- | :--- | :--- | :--- |
| **D01–D14** | Day 01: Project Foundation, Tech Stack & Repository Structure | 2026-10-19 | Accepted |
| **D15** | Unified Relational Schema Design (8 Core Tables) | 2026-10-20 | Accepted |
| **D16** | 12-Machine Smart Factory Fleet Architecture | 2026-10-20 | Accepted |
| **D17** | AI4I Equipment Failure Taxonomy Standardization | 2026-10-20 | Accepted |
| **D18** | MetroPT-3 Real Air Leak Failure Window Extraction | 2026-10-20 | Accepted |
| **D19** | Physical Sensor Synthesis (Vibration, RPM, Energy) | 2026-10-20 | Accepted |
| **D20** | Whole-Week Integer Timeline Shift Mechanics | 2026-10-20 | Accepted |
| **D21** | Hero Machine (M-104) Multi-Stage Degradation Arc | 2026-10-20 | Accepted |
| **D22** | Operating Hours-Grounded Maintenance Scheduling | 2026-10-20 | Accepted |
| **D23** | M-104 "Smoking Gun" Inspection Record & Human Error Anchor | 2026-10-20 | Accepted |
| **D24** | Physical Vibration to Production Scrap Linkage | 2026-10-20 | Accepted |
| **D25** | Stock-Flow Inventory Accounting Grounded in Retail Demand | 2026-10-20 | Accepted |
| **D26** | PostgreSQL 16 & SQLAlchemy 2.0 ORM Architecture | 2026-10-20 | Accepted |
| **D27** | Deterministic Reproducibility with Global Seed 42 & UUIDv5 | 2026-10-20 | Accepted |
| **D28** | Strict Zero-Data-in-Git Exclusion Policy | 2026-10-20 | Accepted |

---

## Detailed Decisions (Day 02: D15–D28)

### D15: Unified Relational Schema Design (8 Core Tables)
- **Context**: The platform requires cross-layer correlation across sensor telemetry, maintenance logs, production outputs, and business financials.
- **Decision**: Standardize on 8 relational tables: `machines`, `sensors_raw`, `maintenance_records`, `production_daily`, `business_daily`, `business_transactions`, `ai4i_state`, and `metropt_failure_windows`.
- **Consequences**: Enables SQL joins linking physical machine distress to work orders, scrap rates, and financial revenue impacts.

### D16: 12-Machine Smart Factory Fleet Architecture
- **Context**: Need a fleet large enough to represent realistic enterprise operations without overwhelming local development hardware.
- **Decision**: Define a 12-machine fleet spanning 3 physical plants (Plants A, B, C) and 3 equipment types (Air Compressors, Hydraulic Presses, Conveyor Drives).
- **Consequences**: Exact coverage of 10 distinct operational phenotypes (healthy, drifting, degrading, past failures, recently maintained, overdue, in-maintenance).

### D17: AI4I Equipment Failure Taxonomy Standardization
- **Context**: Need standard root-cause failure classifications across industrial assets.
- **Decision**: Adopt the 5 physical failure modes from AI4I (`Tool Wear Failure`, `Heat Dissipation Failure`, `Power Failure`, `Overstrain Failure`, `Random Failure`) along with MetroPT compressor modes (`Air Leak`, `Pressure Drop`).
- **Consequences**: Direct alignment between ML diagnostic models (Day 7) and maintenance root-cause tracking.

### D18: MetroPT-3 Real Air Leak Failure Window Extraction
- **Context**: Real failure dynamics are superior to purely synthetic signatures for diagnostic realism.
- **Decision**: Extract 4 real failure windows (`FAIL-001` through `FAIL-004`) from `Data Description_Metro.pdf` into a version-controlled reference table `metropt_failure_windows.csv`.
- **Consequences**: Preserves authentic compressor pressure decay and electrical recovery profiles.

### D19: Physical Sensor Synthesis (Vibration, RPM, Energy)
- **Context**: MetroPT lacks vibration and RPM channels; purely random numbers lack physics credibility.
- **Decision**: Synthesize vibration from electrical load ($L_t$) and oil temperature ($T_t$) via AR(1) autoregressive noise; derive RPM inversely proportional to load; derive three-phase energy ($E_t = \sqrt{3} V I \cos\phi$).
- **Consequences**: Telemetry adheres to electromagnetic and thermodynamic conservation laws.

### D20: Whole-Week Integer Timeline Shift Mechanics
- **Context**: Datasets from 2011 and 2020 must align with a modern operations timeline ending in October 2026 (`2026-10-19`).
- **Decision**: Shift dates exclusively by exact integer multiples of 7 days: MetroPT shifted by 320 weeks (+2,240 days); Online Retail shifted by 776 weeks (+5,432 days).
- **Consequences**: Strictly preserves day-of-week cadence, weekend shut-downs, and intra-week operational rhythms.

### D21: Hero Machine (M-104) Multi-Stage Degradation Arc
- **Context**: The core demonstration requires a machine actively degrading toward catastrophic failure at the end of the timeline.
- **Decision**: Assign real MetroPT signals to M-104, overlay real failure window ramps, and apply an exponential degradation ramp over the final 35 days. Vibration exceeds 4.5 mm/s exactly 12.99 days before timeline end; temperature in the final 24h exceeds the 95th percentile by >= 8 °C.
- **Consequences**: Creates a realistic diagnostic timeline for alerting, RAG retrieval, and predictive maintenance.

### D22: Operating Hours-Grounded Maintenance Scheduling
- **Context**: Maintenance intervals should reflect actual equipment run time rather than arbitrary calendar dates.
- **Decision**: Calculate cumulative operating hours directly from sensor motor current ($I_t > 1.55\text{ A}$). Schedule inspections every ~500–800 hours and overhauls every ~2,000 hours.
- **Consequences**: M-107 accumulates over 2,400 hours without an overhaul, naturally establishing its "overdue" condition.

### D23: M-104 "Smoking Gun" Inspection Record & Human Error Anchor
- **Context**: Diagnostic RAG agents need realistic unstructured textual evidence explaining why degradation went uncorrected.
- **Decision**: Place an inspection record for M-104 on `2026-09-20` noting: *"Minor air leak suspected on discharge manifold. Recommended follow-up in 30 days. Operating parameters within tolerance."* No follow-up PM was scheduled.
- **Consequences**: Provides the exact evidence the AI Root Cause Analysis agent will discover on Day 10.

### D24: Physical Vibration to Production Scrap Linkage
- **Context**: Industrial degradation directly degrades product manufacturing quality.
- **Decision**: Couple M-104's scrap rate to its degradation progress: scrap rises from 2.0% baseline up to 10.5% over the final 14 days.
- **Consequences**: Connects telemetry anomalies directly to daily scrap waste and lost factory yield.

### D25: Stock-Flow Inventory Accounting Grounded in Retail Demand
- **Context**: Business metrics must realistically fluctuate based on real consumer demand.
- **Decision**: Standardize Online Retail transactions to derive daily demand indices, driving factory dispatches, inventory stock-flow balance, and stockout backorders.
- **Consequences**: Disrupted production on outage days causes visible inventory drawdown and revenue loss.

### D26: PostgreSQL 16 & SQLAlchemy 2.0 ORM Architecture
- **Context**: Production database must support relational constraints, JSONB attributes, and high-performance querying.
- **Decision**: Deploy PostgreSQL via Docker Compose; implement ORM models using SQLAlchemy 2.0 `Mapped` and `mapped_column` with SQLite fallback for offline development.
- **Consequences**: Type-safe data access layer across FastAPI backend and data engineering scripts.

### D27: Deterministic Reproducibility with Global Seed 42 & UUIDv5
- **Context**: Data pipelines must be 100% reproducible across machines and CI runs.
- **Decision**: Enforce global seed `SEED = 42` (NumPy PCG64) and deterministic `UUIDv5` hashing for all generated primary keys.
- **Consequences**: End-to-end runs yield byte-identical CSVs and matching SHA-256 digests.

### D28: Strict Zero-Data-in-Git Exclusion Policy
- **Context**: Raw CSVs (several hundred megabytes) must never bloat git history.
- **Decision**: Enforce `.gitignore` on `data/raw/unified/*.csv` and source archives; track only reproducible generator scripts and small reference tables in `ml/preprocessing/reference/`.
- **Consequences**: Clean, lightweight repository cloneable in seconds.
