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
| **D29** | Dual Integrated Granularities (5-min & Daily) | 2026-10-21 | Accepted |
| **D30** | Two-Tier Outlier Policy (Physically Impossible vs. Statistical) | 2026-10-21 | Accepted |
| **D31** | Short-Run Bounded Imputation Policy (<= 15 minutes) | 2026-10-21 | Accepted |
| **D32** | Strictly Trailing Integer Rolling Windows (1h, 6h, 24h) | 2026-10-21 | Accepted |
| **D33** | Ground-Truth Label Isolation | 2026-10-21 | Accepted |
| **D34** | Parquet-First Storage Standard with Dual-Format Operations Dataset | 2026-10-21 | Accepted |
| **D35** | Retail Transactions Cleaning & Rejection Separation | 2026-10-21 | Accepted |
| **D36** | PostgreSQL COPY Load inside Single Atomic Transaction | 2026-10-21 | Accepted |
| **D37** | Strict UTC Time Reference Standard | 2026-10-21 | Accepted |
| **D38** | First 24-Hour Warmup Flagging | 2026-10-21 | Accepted |
| **D39** | Per-Machine Baseline Modeling Deferral | 2026-10-21 | Accepted |
| **D40** | Operations Dataset Complete-Day Filtering | 2026-10-21 | Accepted |
| **D41** | Strictly Past Maintenance Feature As-Of Timing (D + 1) | 2026-10-21 | Accepted |

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

---

## Detailed Decisions (Day 03: D29–D41)

### D29: Dual Integrated Granularities (5-min & Daily)
- **Context**: Industrial machine learning models and operational dashboards operate at fundamentally different temporal resolutions. High-frequency anomaly detection requires 5-minute telemetry, while financial correlation and demand planning require daily resolution.
- **Decision**: Produce two primary integrated datasets: `sensor_features.parquet` at 5-minute resolution and `operations_dataset.parquet` (and `.csv`) at daily per-machine resolution.
- **Consequences**: Prevents dataset bloating while avoiding information loss across disparate modeling and visualization layers.

### D30: Two-Tier Outlier Policy (Physically Impossible vs. Statistical)
- **Context**: Naive outlier removal destroys real failure dynamics (such as bearing overheating and severe air leak vibration).
- **Decision**: Implement a strict two-tier architecture:
  - **Tier 1 (Physical Impossibility):** Hard physical limit violations (e.g., $T > 150^\circ\text{C}$ or $P < 0$) are set to `NaN` and flagged with `is_sensor_fault = True`.
  - **Tier 2 (Statistical Extremes):** Trailing 24-hour robust $z$-scores ($|Z| > 6.0$) are flagged via `is_outlier_flag = True` and listed in `outlier_signals`, but numeric values are never modified or removed.
- **Consequences**: 100% of authentic degradation peaks and benign operational surges are preserved for machine learning detection.

### D31: Short-Run Bounded Imputation Policy (<= 15 minutes)
- **Context**: Sensor dropouts occur naturally, but unbounded interpolation fabricates artificial equipment states.
- **Decision**: Interpolate or forward-fill contiguous gaps of at most 3 intervals (15 minutes) and flag them with `is_imputed = True`. Gaps longer than 15 minutes remain `NaN` with explicit `is_gap = True` rows.
- **Consequences**: Maintains time-series continuity for short dropouts without masking actual communication outages or downtime.

### D32: Strictly Trailing Integer Rolling Windows (1h, 6h, 24h)
- **Context**: Temporal feature engineering is vulnerable to future leakage if centered windows or future shifts are applied.
- **Decision**: All rolling features use strictly trailing integer windows (12, 72, and 288 intervals) on a regular 5-minute grid, requiring at least 50% valid observations.
- **Consequences**: Guarantees zero look-ahead bias across all sliding-window statistics.

### D33: Ground-Truth Label Isolation
- **Context**: Inadvertent inclusion of failure indicators or scenario metadata in feature tables contaminates model evaluation.
- **Decision**: Store all ground-truth failure windows, precursor intervals, and episode IDs exclusively in `data/processed/anomaly_eval_labels.parquet`. Sensor feature and operations tables strictly exclude target labels.
- **Consequences**: Provides uncompromised ground-truth targets for Day 5 and Day 6 unsupervised and supervised model evaluation.

### D34: Parquet-First Storage Standard with Dual-Format Operations Dataset
- **Context**: Pipeline tables exceed 800,000 rows, making plain CSV storage slow and memory-intensive.
- **Decision**: Persist all processed tables as compressed Parquet files. Generate `operations_dataset.csv` alongside `operations_dataset.parquet` as the single canonical tabular file for cross-platform EDA.
- **Consequences**: Faster I/O throughput, reduced disk footprint, and automatic preservation of column data types.

### D35: Retail Transactions Cleaning & Rejection Separation
- **Context**: Retail transactions contain cancellations, non-product stock codes (e.g., `POST`), and unverified returns that distort financial KPIs.
- **Decision**: Split transactions into clean kept records (`business_transactions.parquet`) and invalid rejections (`business_transactions_rejected.parquet`). Flag non-product lines and extreme quantities ($\ge 10,000$), excluding them from revenue and order velocity aggregations.
- **Consequences**: Full financial auditability with non-destructive separation of anomalous and administrative records.

### D36: PostgreSQL COPY Load inside Single Atomic Transaction
- **Context**: Inserting hundreds of thousands of rows via standard SQL `INSERT` is prohibitively slow.
- **Decision**: Execute database ingestion via PostgreSQL `COPY` (using TSV in-memory buffers) inside a single atomic transaction with foreign-key-safe ordering and post-load verification.
- **Consequences**: Sub-minute database loading with complete rollback safety on failure.

### D37: Strict UTC Time Reference Standard
- **Context**: Mixing local time zones across plant locations causes off-by-one errors in daily aggregations.
- **Decision**: Enforce UTC timestamps across all tables, defining daily aggregation boundaries strictly at 00:00 UTC.
- **Consequences**: Consistent cross-table joins between high-frequency sensor readings and daily production logs.

### D38: First 24-Hour Warmup Flagging
- **Context**: Rolling features requiring trailing 24-hour windows are incomplete during the initial 288 intervals for each asset.
- **Decision**: Explicitly flag the initial 288 intervals of each machine with `is_warmup = True` rather than dropping rows.
- **Consequences**: Retains complete time series while allowing training pipelines to filter or adjust for initialization bias.

### D39: Per-Machine Baseline Modeling Deferral
- **Context**: Computing fleet-wide or per-machine normative baselines across the full timeline introduces data leakage.
- **Decision**: Defer per-machine normative baselines (e.g., healthy-state deviation) to Day 5, where baselines are fit strictly on the early training partition.
- **Consequences**: Fully complies with leakage prevention policy L3.

### D40: Operations Dataset Complete-Day Filtering
- **Context**: Partial days at the boundary of a dataset distort cumulative metrics like daily operating hours and energy.
- **Decision**: Include only calendar days with complete 288-interval coverage in `operations_dataset.parquet`.
- **Consequences**: Guarantees consistent daily totals for all machines across the active timeline.

### D41: Strictly Past Maintenance Feature As-Of Timing (D + 1)
- **Context**: Joining maintenance events on the day of maintenance can leak intervention details before work is completed.
- **Decision**: A maintenance event occurring on date $D$ becomes visible to rolling features strictly on date $D + 1$.
- **Consequences**: Ensures features reflect only historical, completed maintenance interventions.

### D42: Modular Notebook Layout & Helper Architecture
- **Context**: A single monolithic exploratory data analysis notebook becomes bloated, difficult to navigate, and prone to unreproducible execution states.
- **Decision**: Partition EDA into 6 focused notebooks (`01_overview_and_lineage.ipynb` through `06_ai4i_failure_data.ipynb`) backed by a shared helper package in `ml/eda/` (`config.py`, `load.py`, `splits.py`, `plotting.py`, `stats.py`).
- **Consequences**: Highly modular, maintainable, and independently executable analysis modules backed by automated unit tests in `tests/test_eda.py`.

### D43: Notebook Output Stripping via nbstripout
- **Context**: Committing rich notebook outputs (base64 image payloads, JSON metadata) creates massive Git diffs and bloats repository size.
- **Decision**: Install `nbstripout` as a Git pre-commit filter, committing notebooks in a clean state (code and markdown only), while exporting publication-quality figures directly to `docs/figures/`.
- **Consequences**: Clean Git history, minimal repository footprint, and reproducible, self-contained visual artifacts.

### D44: Chronological 70/30 Exploration vs. Holdout Partitioning
- **Context**: Selecting features, tuning anomaly thresholds, or deriving operational baselines on the full operational horizon introduces lookahead bias and overfits downstream evaluations.
- **Decision**: Establish an immutable chronological split: Exploration period = first 70% of days (first 150 days, cutoff `2026-08-17 00:00 UTC`), Holdout period = remaining 30% (last 64 days). All modeling choices for Days 5, 6, and 8 must be derived strictly from the exploration partition.
- **Consequences**: Completely shields the holdout period for uncompromised out-of-time evaluation in downstream modeling phases.

### D45: Evaluation Labels Strictly for Interpretation
- **Context**: Using supervised failure labels (`anomaly_eval_labels.parquet`) to rank or select features for unsupervised anomaly detection constitutes severe label leakage.
- **Decision**: Restrict evaluation labels exclusively to visual interpretation (event studies, timeline shading, and empirical base rate reporting). Under no circumstances may labels guide feature selection or threshold calibration.
- **Consequences**: Guarantees true unsupervised integrity for Day 5 anomaly models.

### D46: Stratified AI4I Train/Test Split Quarantining
- **Context**: Evaluating machine failure prediction models without a fixed, pre-allocated test set risks cross-validation leakage and data snooping.
- **Decision**: Generate an immutable 80/20 train/test stratified split index (`data/processed/ai4i_split_index.csv`, seed 42) based on `machine_failure`. EDA investigates the training partition only (8,000 rows); the test partition (2,000 rows) remains strictly quarantined for Day 6 final evaluation.
- **Consequences**: Prevents optimistic bias and establishes a rigorous, uncompromised test bed for failure classification.

### D47: Mandatory Provenance Tagging on All Findings
- **Context**: Blending real physical data (MetroPT compressor signals, Online Retail transactions) with algorithmic derivations creates risk of presenting synthetic artifacts as physical operational insights.
- **Decision**: Mandate explicit provenance tags on all findings and documentation: `[REAL]`, `[REAL-DERIVED]`, `[SYNTHETIC-BY-DESIGN]`, and `[MIXED]`.
- **Consequences**: Enforces intellectual honesty and clear industrial domain boundaries for all analytical conclusions.

### D48: Block Bootstrap Confidence Intervals for Autocorrelated Series
- **Context**: Industrial sensor streams and retail demand series exhibit strong temporal autocorrelation. Conventional i.i.d. bootstrap methods severely underestimate standard errors and produce overconfident confidence intervals.
- **Decision**: Employ moving-block bootstrap techniques (block lengths of 7 days for daily series, 24 hours for hourly series) to estimate uncertainty for all operational effect sizes and trend statistics.
- **Consequences**: Robust, statistically sound uncertainty intervals that account for temporal dependency.

### D49: Parquet-First Data Ingestion for Analytical Pipelines
- **Context**: Querying PostgreSQL for high-frequency analytical workflows introduces external infrastructure dependencies, connection overhead, and slower I/O.
- **Decision**: Direct all EDA notebooks and analytical loaders to read partitioned Parquet files directly from `data/processed/` using PyArrow column selection and memory mapping.
- **Consequences**: Blazing fast analytical read times (< 2 seconds per notebook) without requiring a running database server.

### D50: Rolling-Origin Baseline Benchmarking inside Exploration Slice
- **Context**: Time series forecasting models on Day 8 require realistic, unpeeking baseline benchmarks to beat.
- **Decision**: Compute naive, seasonal naive, and 7-day moving average baselines using rolling-origin 7-day evaluation across the final 28 days of the exploration partition (`2026-07-20` to `2026-08-16`).
- **Consequences**: Provides uncompromised target metrics (e.g., Fleet Production MA-7 MAE 1,026.8 units) without touching the holdout window.

### D51: Unified Plotting & Visualization Design Standards
- **Context**: Inconsistent figure aesthetics, unreadable font sizes, and inaccessible color palettes degrade the professional quality of operational reports.
- **Decision**: Enforce a standardized plotting style in `ml/eda/plotting.py` utilizing the Okabe-Ito colorblind-safe palette, 150 DPI resolution, consistent font hierarchies, and structured multi-panel layouts exported to `docs/figures/fig01_*.png` through `fig10_*.png`.
- **Consequences**: Cohesive, publication-grade figures suitable for executive dashboards and technical reports.

### D52: Immutability of Processed Data During Exploratory Analysis
- **Context**: Modifying or "fixing" processed data during EDA compromises pipeline determinism and breaks traceability.
- **Decision**: Treat `data/raw/` and `data/processed/` as strictly read-only during EDA (Rule R1). Data generator design deviations are logged in `docs/known_issues.md` rather than patched in-flight.
- **Consequences**: Complete reproducibility and strict separation of concerns between data engineering and data analysis.

---

## Next Steps (Handoff to Day 05: Anomaly Detection)

1. **Day 4 Accomplishments:**
   - Completed all 13 EDA tasks, delivering 6 reproducible notebooks in `ml/notebooks/` (outputs stripped via `nbstripout`).
   - Answered all 47 Exploratory Questions (EQ1 to EQ47) with quantitative evidence and bootstrap confidence intervals.
   - Generated 10 publication-quality curated figures (`fig01_*.png` through `fig10_*.png`) in `docs/figures/`.
   - Delivered `docs/eda_findings.md`, `docs/known_issues.md`, and `docs/day04_walkthrough.md`.
   - Helper package `ml/eda/` fully tested and verified by `pytest tests/test_eda.py`.

2. **Day 5 Modeling Scope (Isolation Forest Anomaly Detection):**
   - **Candidate Feature Shortlist:** `motor_current`, `temperature`, `pressure`, `rpm`, `energy_consumption`, `vibration`, along with 24-hour trailing rolling means, standard deviations, and least-squares slopes (DEC-01).
   - **Training Set Construction:** Fit Isolation Forest models strictly on the exploration period (first 70% of days), explicitly excluding known failure windows and scenario degradation episodes (DEC-02).
   - **Dual Evaluation Protocol on M-104 (Mandatory):** Evaluate M-104 anomaly scoring twice:
     1. On all features (including synthetic vibration and temperature overlay).
     2. On **real-only features** (`pressure`, `motor_current`, `comp_state`, and native MetroPT channels).
     *Note:* Modelers must report both sets of results honestly; (a) alone must never be claimed as real-world detection capability (DEC-03, Caveat C3).
   - **Operational Threshold Proposals:** Implement proposed warning thresholds derived from exploration-period healthy quantiles ($p99.9$): Air Compressor Vibration: 4.0 mm/s (warning), 7.0 mm/s (critical); Temperature: 80.0 °C (warning), 85.0 °C (critical) (Section 6, `docs/eda_findings.md`).
   - **Contamination Parameter:** Set Isolation Forest contamination parameter in the range **0.01 to 0.05 (recommended: 0.03)**. Do NOT set contamination to 32% (which reflects benign intraday duty switching, not physical faults) (DEC-02).
   - **False Positive Stress Test:** Use machine `M-109` (benign vibration spikes) as a benchmark; ensure critical alert rate on M-109 is $< 5\%$ (DEC-04).

3. **Downstream Directives (Days 6 & 8):**
   - **Day 6 (Failure Prediction):** Re-use `data/processed/ai4i_split_index.csv`. Strictly drop mode flags (`twf`, `hdf`, `pwf`, `osf`, `rnf`) and identifiers (`udi`, `product_id`). Report PR-AUC against baseline 0.0339 (DEC-05 to DEC-07).
   - **Day 8 (Forecasting):** Apply 7-day seasonal differencing ($D=1$). Target beating MA-7 baseline MAE $\le 1,026.8$ units on fleet production (DEC-08, DEC-09).

4. **Active Branch for Day 5:** `feature/day-05-anomaly-detection`.

