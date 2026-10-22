# Day 04: Exploratory Data Analysis — Walkthrough Report

**Project:** AI Operations Intelligence Platform  
**Branch:** `feature/day-04-eda`  
**Execution Date:** October 2026  
**Status:** COMPLETE & VERIFIED (All 13 Tasks Completed, 47/47 EQs Answered, 10 Figures Curated, 100% Tests Passing)

---

## 1. Executive Summary

Day 4 completed the end-to-end Exploratory Data Analysis (EDA) for the AI Operations Intelligence Platform. The analysis audited and characterized the multi-tiered processed dataset across 736,128 high-frequency sensor readings, 2,556 machine-days of manufacturing throughput, 43 maintenance work orders, 541,909 retail transactions, and 10,000 records of machine failure states.

Crucially, Day 4 established **intellectual honesty and domain discipline**:
1. **Provenance Tagging:** Delineated authentic physical phenomena (`[REAL]` in MetroPT compressor channels and Online Retail sales) from algorithmic artifacts (`[SYNTHETIC-BY-DESIGN]` in vibration overlays and production downtime math).
2. **Strict Time Partitioning (D44):** Derived all operational baseline thresholds, contamination guidelines, and benchmark forecasting models strictly from the **Exploration Partition** (first 70% of days; 150 days ending `2026-08-17 00:00 UTC`), preserving the remaining 30% holdout window for out-of-time evaluation in downstream phases.
3. **Quarantined Machine Learning Test Sets (D46):** Pre-allocated an immutable 80/20 stratified split (`data/processed/ai4i_split_index.csv`) for AI4I machine failure classification, restricting all exploratory investigations to the 8,000-row training partition and strictly sequestering the 2,000-row test partition for Day 6 final evaluation.
4. **Target Leakage Identification (EQ46):** Proved that failure mode flags (`twf`, `hdf`, `pwf`, `osf`, `rnf`) are causal sub-components of `machine_failure`, mandating their complete removal prior to predictive modeling on Day 6.
5. **Real-World Anomaly Precursor Discovery (EQ26, EQ27):** Proved that unmodified real compressor telemetry (`pressure`, `motor_current`, `comp_state`) shows flat baseline behavior prior to real air leak events, mandating that Day 5 evaluate anomaly models on real-only channels separately from synthetic overlay channels.

---

## 2. Deliverables Checklist

| Category | Deliverable | Path | Status |
| :--- | :--- | :--- | :--- |
| **Helper Package** | Configuration & constants | `ml/eda/config.py` | Complete & Tested |
| | Cached data loaders with PyArrow projection | `ml/eda/load.py` | Complete & Tested |
| | Chronological & stratified split utilities | `ml/eda/splits.py` | Complete & Tested |
| | Shared styling, threshold & episode shading | `ml/eda/plotting.py` | Complete & Tested |
| | Bootstrap CI, block bootstrap, robust z, event study | `ml/eda/stats.py` | Complete & Tested |
| | Package initializer | `ml/eda/__init__.py` | Complete |
| **Test Suite** | 9 comprehensive EDA helper tests | `tests/test_eda.py` | 9/9 Passed (1.84s) |
| **Split Generator** | AI4I 80/20 Stratified Split Generator | `scripts/generate_ai4i_split.py` | Complete |
| | Stratified Split Index (8,000 train / 2,000 test) | `data/processed/ai4i_split_index.csv` | Generated & Git-Ignored |
| **Notebooks** | 01: Overview, Quality & Planted Effects | `ml/notebooks/01_overview_and_lineage.ipynb` | Executed & Stripped |
| | 02: Production, Downtime & Energy | `ml/notebooks/02_production_downtime_energy.ipynb` | Executed & Stripped |
| | 03: Sensors & Machine Health | `ml/notebooks/03_sensors_machine_health.ipynb` | Executed & Stripped |
| | 04: Maintenance, Failures & Event Study | `ml/notebooks/04_maintenance_and_failures.ipynb` | Executed & Stripped |
| | 05: Retail Demand & Forecast Baselines | `ml/notebooks/05_retail_demand_and_baselines.ipynb` | Executed & Stripped |
| | 06: AI4I Failure Data & Imbalance | `ml/notebooks/06_ai4i_failure_data.ipynb` | Executed & Stripped |
| **Curated Figures** | Fig 01: Fleet Production & Downtime | `docs/figures/fig01_fleet_production_downtime.png` | 150 DPI, Captioned |
| | Fig 02: Machine Efficiency & Downtime | `docs/figures/fig02_machine_efficiency_downtime.png` | 150 DPI, Captioned |
| | Fig 03: Vibration ECDF by Machine | `docs/figures/fig03_vibration_ecdf_by_machine.png` | 150 DPI, Captioned |
| | Fig 04: Sensor Timelines Across Four Machines | `docs/figures/fig04_sensor_timelines_four_machines.png` | 150 DPI, Captioned |
| | Fig 05: Hourly Operating Profile | `docs/figures/fig05_hourly_operating_profile.png` | 150 DPI, Captioned |
| | Fig 06: Telemetry Signal Correlations | `docs/figures/fig06_signal_correlations.png` | 150 DPI, Captioned |
| | Fig 07: M-104 Compressor Failure Event Study | `docs/figures/fig07_m104_failure_event_study.png` | 150 DPI, Captioned |
| | Fig 08: Maintenance Frequency & Cost | `docs/figures/fig08_maintenance_frequency_cost.png` | 150 DPI, Captioned |
| | Fig 09: Retail Revenue Seasonality | `docs/figures/fig09_retail_revenue_seasonality.png` | 150 DPI, Captioned |
| | Fig 10: AI4I Imbalance & Predictive Features | `docs/figures/fig10_ai4i_imbalance_features.png` | 150 DPI, Captioned |
| **Documentation** | Comprehensive EDA Findings & Decision Register | `docs/eda_findings.md` | Complete (47 EQs, C1–C8) |
| | Known Issues & Lineage Log | `docs/known_issues.md` | Complete (KI-01 to KI-04) |
| | Architecture Decision Records (D42–D52) | `docs/architecture.md` | Updated with Day 5 Handoff |
| | Data Pipeline Specification | `docs/data_pipeline.md` | Section 8 Added |
| | Project Readme | `README.md` | Updated with EDA Section |
| | Developer Automation | `Makefile` | Added `eda-run`, `eda-test`, `eda-strip` |

---

## 3. Test & Verification Results

### 3.1 EDA Helper Test Suite (`tests/test_eda.py`)
Execution command: `pytest tests/test_eda.py -v`
```text
============================= test session starts =============================
platform win32 -- Python 3.14.6, pytest-9.1.1, pluggy-1.6.0
collected 9 items

tests/test_eda.py::test_split_masks PASSED                               [ 11%]
tests/test_eda.py::test_exploration_cutoff PASSED                        [ 22%]
tests/test_eda.py::test_ai4i_split PASSED                                [ 33%]
tests/test_eda.py::test_bootstrap_ci PASSED                              [ 44%]
tests/test_eda.py::test_block_bootstrap_autocorrelated PASSED            [ 55%]
tests/test_eda.py::test_ecdf PASSED                                      [ 66%]
tests/test_eda.py::test_robust_z_spike_and_no_leakage PASSED             [ 77%]
tests/test_eda.py::test_event_study PASSED                               [ 88%]
tests/test_eda.py::test_save_fig PASSED                                  [100%]

============================== 9 passed in 1.84s ==============================
```

### 3.2 Full Project Regression Suite
Execution command: `pytest -v`
```text
============================= test session starts =============================
collected 31 items

tests/test_data.py::test_machines_catalog PASSED                         [  3%]
tests/test_data.py::test_maintenance_records PASSED                      [  6%]
tests/test_data.py::test_production_daily PASSED                         [  9%]
tests/test_data.py::test_business_daily PASSED                           [ 12%]
tests/test_data.py::test_ai4i_state PASSED                               [ 16%]
tests/test_db_schema.py::test_orm_models_registered_on_metadata PASSED   [ 19%]
tests/test_db_schema.py::test_sqlite_in_memory_schema_and_crud PASSED    [ 22%]
tests/test_db_schema.py::test_postgresql_connection_if_available SKIPPED [ 25%]
tests/test_eda.py::test_split_masks PASSED                               [ 29%]
tests/test_eda.py::test_exploration_cutoff PASSED                        [ 32%]
tests/test_eda.py::test_ai4i_split PASSED                                [ 35%]
tests/test_eda.py::test_bootstrap_ci PASSED                              [ 38%]
tests/test_eda.py::test_block_bootstrap_autocorrelated PASSED            [ 41%]
tests/test_eda.py::test_ecdf PASSED                                      [ 45%]
tests/test_eda.py::test_robust_z_spike_and_no_leakage PASSED             [ 48%]
tests/test_eda.py::test_event_study PASSED                               [ 51%]
tests/test_eda.py::test_save_fig PASSED                                  [ 54%]
tests/test_pipeline.py::test_validation_catches_issues PASSED            [ 58%]
tests/test_pipeline.py::test_grid_reindexing PASSED                      [ 61%]
tests/test_pipeline.py::test_imputation_limit PASSED                     [ 64%]
tests/test_pipeline.py::test_tier_1_hard_limits PASSED                   [ 67%]
tests/test_pipeline.py::test_tier_2_is_flag_only PASSED                  [ 70%]
tests/test_pipeline.py::test_operating_hours_monotonic PASSED            [ 74%]
tests/test_pipeline.py::test_leakage_rolling_features PASSED             [ 77%]
tests/test_pipeline.py::test_leakage_maintenance_features PASSED         [ 80%]
tests/test_pipeline.py::test_leakage_labels_isolation PASSED             [ 83%]
tests/test_pipeline.py::test_retail_cleaning_rules PASSED                [ 87%]
tests/test_pipeline.py::test_daily_aggregation_energy PASSED             [ 90%]
tests/test_pipeline.py::test_business_kpis PASSED                        [ 93%]
tests/test_pipeline.py::test_pipeline_idempotence PASSED                 [ 96%]
tests/test_pipeline.py::test_operations_dataset_schema PASSED            [100%]

======================= 30 passed, 1 skipped in 17.06s ========================
```

### 3.3 Code Quality & Formatting
Execution command: `ruff check ml/eda tests/test_eda.py`
```text
All checks passed!
```

### 3.4 Git Pre-Commit Hygiene Checks
- `git ls-files data | grep -v -E "\.gitkeep|README.md"` $\rightarrow$ **Empty (Zero data files tracked)**.
- `grep -R "partition=\"test\"" ml/notebooks` $\rightarrow$ **Empty (Zero AI4I test partition references)**.
- `git diff --stat ml/notebooks/` $\rightarrow$ **Empty (All notebook outputs cleanly stripped via nbstripout)**.

---

## 4. Notebook Execution Confirmation

All 6 notebooks in `ml/notebooks/` execute cleanly top-to-bottom:
1. `ml/notebooks/01_overview_and_lineage.ipynb` (Executed in 4.2s, 0 errors).
2. `ml/notebooks/02_production_downtime_energy.ipynb` (Executed in 6.1s, generated `fig01`, `fig02`).
3. `ml/notebooks/03_sensors_machine_health.ipynb` (Executed in 18.4s, generated `fig03`, `fig04`, `fig05`, `fig06`).
4. `ml/notebooks/04_maintenance_and_failures.ipynb` (Executed in 8.3s, generated `fig07`, `fig08`).
5. `ml/notebooks/05_retail_demand_and_baselines.ipynb` (Executed in 11.2s, generated `fig09`).
6. `ml/notebooks/06_ai4i_failure_data.ipynb` (Executed in 5.8s, generated `fig10`).

---

## 5. Summary of Answers to All 47 Exploratory Questions

| # | Question Summary | Quantitative Finding & Evidence | Provenance |
| :--- | :--- | :--- | :--- |
| **EQ1** | Dataset temporal boundaries & coverage | 214 calendar days; Exploration = 150 days (514,944 intervals, 69.9%); Holdout = 64 days (221,184 intervals, 30.1%). Cutoff: `2026-08-17 00:00 UTC`. | `[REAL/SYNTHETIC]` |
| **EQ2** | Data quality, gaps, and missingness | Zero Tier 1 physical violations. 1 interval imputed across fleet (0.0001%). M-104 has zero missing 5-min intervals. | `[REAL/SYNTHETIC]` |
| **EQ3** | Telemetry lineage & caveats | 6 channels across 4 provenance tiers. Caveats C1–C8 fully documented in findings and architecture log. | `[MIXED]` |
| **EQ4** | Planted effects recovery verification | 5 effects pass strictly; 3 design deviations logged in `docs/known_issues.md` (KI-01 to KI-03). | `[SYNTHETIC-BY-DESIGN]` |
| **EQ5** | Cross-machine daily independence | Daily correlation off-diagonal median is near zero ($r = 0.0126$ vibration, $r = -0.0014$ temperature). Intraday waveforms share templates. | `[SYNTHETIC-BY-DESIGN]` |
| **EQ6** | Production throughput by type | Compressors $\mu = 313$ units ($CV=0.39$); Presses $\mu = 636$ units ($CV=0.37$); Conveyors $\mu = 1,235$ units ($CV=0.36$). | `[SYNTHETIC-BY-DESIGN]` |
| **EQ7** | Weekend vs. weekday production ratio | Continuous operations; median ratio = 0.957 [0.905, 1.094]. Weekday mean = 730 vs. Weekend mean = 698 units/day. | `[SYNTHETIC-BY-DESIGN]` |
| **EQ8** | Machine operational efficiency | Fleet efficiency averages 98.4%. Outages $< 80\%$ efficiency occur on $< 2.1\%$ of machine-days (53 days total). | `[SYNTHETIC-BY-DESIGN]` |
| **EQ9** | Downtime duration & drivers | Fleet incurred 28,891 downtime hours; maintenance work orders account for 61.8% (17,860h); idle time accounts for 38.2%. | `[SYNTHETIC-BY-DESIGN]` |
| **EQ10** | Production sensitivity to downtime | Linear loss ($r = -0.9991$): Compressors -0.410, Presses -0.821, Conveyors -1.641 units/min. | `[SYNTHETIC-BY-DESIGN]` |
| **EQ11** | Finished goods inventory buffer | Fleet production (8,689 units/day) vs. demand (714 units/day) yields stable 2.55M unit buffer; zero stock-outs. | `[SYNTHETIC-BY-DESIGN]` |
| **EQ12** | Energy efficiency per unit | Stationary across exploration: Conveyors 0.023, Presses 0.045, Compressors 0.095 kWh/unit. | `[REAL-DERIVED]` |
| **EQ13** | Healthy telemetry distributions | Healthy baseline quantiles established: Compressor vibration p50 = 2.06, p99 = 3.31, p99.9 = 3.59 mm/s; temp p99.9 = 82.97 °C. | `[REAL/SYNTHETIC]` |
| **EQ14** | Manual threshold headroom & false alarms | Zero healthy readings exceeded 4.5 mm/s or 85 °C in exploration (0.00% false alarms). M-104 hero degradation in holdout triggers 23,750 min $> 4.5$ mm/s. | `[SYNTHETIC-BY-DESIGN]` |
| **EQ15** | Hourly operating duty profiles | Diurnal duty cycle: running ratio drops to 40% at night (01:00–05:00 UTC) and rises to 85% during day shifts. Current swings 1.8A to 4.8A. | `[REAL/REAL-DERIVED]` |
| **EQ16** | Correlation among telemetry channels | Current vs. energy ($r = 1.000$); current vs. vibration ($r = 0.728$); temp vs. vibration ($r = 0.694$). | `[MIXED]` |
| **EQ17** | Baseline telemetry drift over time | Healthy units stationary ($< 0.5\%$/month). M-103 exhibits steady +1.4% monthly drift (+0.068 mm/s/month). | `[SYNTHETIC-BY-DESIGN]` |
| **EQ18** | Temporal autocorrelation structure | Pronounced 24-hour cycle: lag-24 ACF is 0.494 (temperature) and 0.393 (vibration); lag-168 is 0.231. | `[REAL]` |
| **EQ19** | Outlier rates & contamination guidance | Tier 1 violations = 0.00%. Tier 2 trailing robust z flag rate averages 32.7% (benign duty steps). Contamination recommended: 0.01–0.05. | `[REAL/SYNTHETIC]` |
| **EQ20** | M-104 missingness & gap distribution | Zero missing 5-minute intervals in cleaned series. All 4 real failure windows have 100% telemetry coverage. | `[REAL]` |
| **EQ21** | Maintenance spend & work order types | Fleet incurred 43 work orders totaling £78,450. Corrective = 3 (£13,150, $\mu = £4,383$); PM = 13 (£52,400); Inspection = 27 (£12,900). | `[SYNTHETIC-BY-DESIGN]` |
| **EQ22** | Service interval compliance | Median interval between inspections = 750.0h; PM interval = 550.4h (early anchor services in 7-month horizon). | `[SYNTHETIC-BY-DESIGN]` |
| **EQ23** | Maintenance overdue delinquency | M-104 (150 days) and M-107 (104 days) exhibit severe overdue status; serves as key predictive feature. | `[SYNTHETIC-BY-DESIGN]` |
| **EQ24** | Unscheduled breakdown history | Exactly 3 catastrophic breakdown outages: M-105 (Pressure Drop), M-110 (Overstrain), M-111 (Heat Dissipation). | `[SYNTHETIC-BY-DESIGN]` |
| **EQ25** | Operating hours vs. wear correlation | Normal equipment stationary ($r = -0.04$ on vibration vs. hours); wear is strictly scenario-driven. | `[SYNTHETIC-BY-DESIGN]` |
| **EQ26** | M-104 event study: real vs. overlay | Unmodified real channels (`pressure`, `motor_current`, `comp_state`) show flat normalized ratios (~1.0). Synthetic overlays ramp +60%. | `[REAL vs SYNTHETIC]` |
| **EQ27** | Precursor lead times | Zero lead time on real signals in exploration. Synthetic hero degradation in holdout window FAIL-004 provides 26.5h lead time. | `[MIXED]` |
| **EQ28** | Derived machine precursor signatures | M-105, M-110, and M-111 follow identical 72h linear ramps (+60% vibration) prior to 24h outages. | `[SYNTHETIC-BY-DESIGN]` |
| **EQ29** | Maintenance reset step response | M-106 planned PM on 2026-10-05 abruptly reduced mean vibration by -1.21 mm/s (-27.4%), confirming maintenance reset modeling. | `[SYNTHETIC-BY-DESIGN]` |
| **EQ30** | Native MetroPT precursor audit | Native discharge channels (`TP2`, `DV_pressure`) capture cyclic drying tower regeneration in authentic compressor dynamics. | `[REAL]` |
| **EQ31** | Retail revenue patterns & trading days | Commercial sales average £32,845/day on active days. Complete shutdown on Sundays (£0 revenue; 52 days). | `[REAL]` |
| **EQ32** | Annual retail sales seasonality | Strong Q4 holiday seasonality peaking in November at £214,500/month (2.6x peak-to-median ratio). | `[REAL]` |
| **EQ33** | Product SKU Pareto distribution | Extreme concentration: Top 10 SKUs generate 13.8% of revenue; Top 50 generate 31.4%; 482 SKUs generate 80%. | `[REAL]` |
| **EQ34** | Transaction data hygiene & rejections | Return rate = 1.62% of lines (8,706 lines). Guest checkouts = 24.5% of lines, 16.7% of revenue. UK accounts for 88.5%. | `[REAL]` |
| **EQ35** | Commercial customer concentration | Top 10% of registered accounts generate 60.4% of total registered sales volume. | `[REAL]` |
| **EQ36** | Non-trading day distribution | Exactly 69 zero-sales days: 52 regular Sunday closures and 17 holiday / inventory closure days. | `[REAL]` |
| **EQ37** | Time series stationarity & differencing | Fleet production ($p = 0.0001$) and retail revenue ($p = 0.0004$) are trend-stationary; 7-day seasonality explains 31.2% of variance. | `[REAL/SYNTHETIC]` |
| **EQ38** | Forecast benchmark baselines | 28-day exploration slice benchmarks: Fleet Production MA-7 MAE = 1,026.8 (11.5% MAPE); Retail MA-7 MAE = £17,379.6 (42.4% MAPE). | `[REAL/SYNTHETIC]` |
| **EQ39** | Factory order coupling to retail sales | Factory commercial orders correlate strongly with retail sales volume ($r = 0.8734$). | `[SYNTHETIC-BY-DESIGN]` |
| **EQ40** | AI4I failure target class imbalance | Prevalence in 8,000-row training set is exactly 3.39% (271 failures). Majority accuracy = 96.61%; PR-AUC baseline = 0.0339. | `[SYNTHETIC-BY-DESIGN]` |
| **EQ41** | Failure mode co-occurrence | Heat dissipation (`hdf`, 86) and power failure (`pwf`, 82) are dominant. 22 multi-mode rows; 7 unlabeled failures. | `[SYNTHETIC-BY-DESIGN]` |
| **EQ42** | AI4I feature effect sizes | Torque ($d = +1.48$ [1.32, 1.64]) and tool wear ($d = +0.89$ [0.76, 1.02]) have large positive effects; speed has negative effect ($d = -1.05$). | `[SYNTHETIC-BY-DESIGN]` |
| **EQ43** | Failure rate across machine types | Low-quality 'L' machines fail at 3.92% vs. 2.14% on 'H' machines. Top decile of torque has a 21.4% failure rate. | `[SYNTHETIC-BY-DESIGN]` |
| **EQ44** | Engineered physics features separation | Mechanical power ($ROC-AUC = 0.745$) and cumulative strain ($ROC-AUC = 0.692$) outperform raw temperature and speed. | `[SYNTHETIC-BY-DESIGN]` |
| **EQ45** | Decision rule-likeness diagnostic | Depth-2 decision tree on `strain` and `power_w` achieves 85.2% precision, confirming deterministic rule synthesis. | `[SYNTHETIC-BY-DESIGN]` |
| **EQ46** | Failure mode target leakage audit | Mode flags (`twf`, `hdf`, `pwf`, `osf`, `rnf`) determine target logically and must be dropped; `udi` and `product_id` must be dropped. | `[SYNTHETIC-BY-DESIGN]` |
| **EQ47** | Evaluation standards for imbalanced ML | Day 6 failure models must report PR-AUC, Recall @ 80% Precision, and Brier calibration scores. Accuracy is strictly prohibited. | `[SYNTHETIC-BY-DESIGN]` |

---

## 6. Planted-Effects Recovery Scorecard

| Planted Effect | Source Spec | Measured Value | Expected Value | Tolerance | Status | Diagnosis & Disposition |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Weekend production factor** | Day 2 Generator | **0.957** | 0.60 | $\pm 0.08$ | **FAIL** | Generator tied production to equipment running availability (~0.95) rather than artificial shutdown. Logged in `docs/known_issues.md` (KI-01). |
| **Healthy vibration threshold** | Day 2 Generator | **100.00%** | $\ge 99.5\%$ | $\ge 99.5\%$ | **PASS** | Complete absence of false alarms ($> 4.5$ mm/s) on healthy machines in exploration period. |
| **M-103 linear drift** | Day 2 Generator | **4.754 mm/s** | [4.5, 5.8] | [4.5, 5.8] | **PASS** | 21-day linear drift successfully elevated vibration into warning band. |
| **M-104 hero degradation** | Day 2 Generator | **7.746 mm/s** | $\ge 7.0$ | $\ge 7.0$ | **PASS** | Exponential degradation reaches 7.746 mm/s prior to timeline end. |
| **Routine inspection spacing** | Day 2 Generator | **750.0 hours** | 500 hours | $\pm 15\%$ (425–575h) | **FAIL** | Generator scheduled routine milestones at `[550h, 1300h]`, yielding a 750h interval. Logged in `docs/known_issues.md` (KI-02). |
| **Preventive service interval** | Day 2 Generator | **550.4 hours** | 2,000 hours | $\pm 15\%$ (1700–2300h) | **FAIL** | 7-month calendar horizon permits at most 1–2 PMs per unit; multi-PM intervals are sparse. Logged in `docs/known_issues.md` (KI-03). |
| **Retail orders coupling** | Day 2 Generator | **0.8734** | $\ge 0.50$ | $\ge 0.50$ | **PASS** | Strong synthetic coupling between factory production orders and commercial retail demand. |
| **Business daily equality** | Day 3 Pipeline | **0 diff** | Exact match | 0 diff | **PASS** | Exact row-level equality confirmed across all 2,556 machine-days. |

---

## 7. Curated Figure Gallery

All figures generated at 150 DPI with Okabe-Ito colorblind-safe palettes in `docs/figures/`:

1. **`fig01_fleet_production_downtime.png`:** Fleet daily manufacturing throughput vs. 12,600-unit target with 7-day moving average, paired with stacked daily downtime hours (maintenance, post-failure, operational).
2. **`fig02_machine_efficiency_downtime.png`:** Per-machine operational efficiency distributions vs. 80% and 100% target thresholds, and mean downtime hours split by maintenance vs. non-maintenance days.
3. **`fig03_vibration_ecdf_by_machine.png`:** Empirical cumulative distribution functions (ECDF) of vibration across all 12 machines comparing exploration (solid) vs. holdout (dashed) against 4.5 mm/s warning and 7.0 mm/s critical limits.
4. **`fig04_sensor_timelines_four_machines.png`:** Full 7-month telemetry timelines of daily mean and max vibration and temperature across four diagnostic regimes (M-101 healthy, M-103 linear drift, M-104 hero degrading, M-109 benign spikes).
5. **`fig05_hourly_operating_profile.png`:** 24-hour diurnal operating telemetry profiles (running ratio, current, vibration, temperature) comparing weekday vs. weekend duty cycles.
6. **`fig06_signal_correlations.png`:** Spearman rank correlation heatmap among 6 telemetry channels for healthy equipment in the exploration period, highlighting generator-governed collinearity.
7. **`fig07_m104_failure_event_study.png`:** M-104 compressor failure event study (-72h to +24h) contrasting unmodified real-world sensor channels (flat baseline) with synthetic overlay-affected channels (+60% ramp).
8. **`fig08_maintenance_frequency_cost.png`:** Fleet maintenance expenditure and work order volume by machine, and empirical distribution of operating hours between services vs. 500h inspection and 2,000h PM policy rules.
9. **`fig09_retail_revenue_seasonality.png`:** Daily commercial retail revenue with 7-day and 28-day rolling moving averages, weekday trading share bars highlighting Sunday closures, and monthly commercial revenue progression.
10. **`fig10_ai4i_imbalance_features.png`:** Failure mode frequencies and 3.39% class imbalance pie chart, and feature distribution comparisons between healthy and failing states across torque, tool wear, and temperature gradient.

---

## 8. Downstream Decisions & Directives

### For Day 5: Unsupervised Anomaly Detection (Isolation Forest)
- **Candidate Features (DEC-01):** Shortlist `motor_current`, `temperature`, `pressure`, `rpm`, `energy_consumption`, `vibration`, plus 24h rolling means, standard deviations, and least-squares slopes.
- **Normal Training Set (DEC-02):** Fit models strictly on the exploration period, explicitly excluding failure windows and scenario degradation episodes.
- **Dual Evaluation Protocol on M-104 (DEC-03, Mandatory):** Evaluate M-104 twice: (a) on all features, and (b) on **real-only features** (`pressure`, `motor_current`, `comp_state`). Both must be reported; (a) alone must not be claimed as real-world performance.
- **Operational Warning Thresholds (Section 6):** Air Compressor: 4.0 mm/s vibration, 80.0 °C temperature. Hydraulic Press: 4.2 mm/s vibration, 80.0 °C temperature. Conveyor Drive: 3.8 mm/s vibration, 78.0 °C temperature.
- **Contamination Parameter:** Set `contamination = 0.03` (Range: [0.01, 0.05]). Do NOT set contamination to 32% (which reflects normal diurnal load cycling).
- **False Positive Benchmark (DEC-04):** M-109 (benign vibration spikes) must achieve $< 5\%$ critical alert rate.

### For Day 6: Machine Failure Prediction (Classification)
- **Split Quarantining (DEC-05):** Re-use `data/processed/ai4i_split_index.csv`. Use training partition (8,000 rows) for model selection and cross-validation; test partition (2,000 rows) strictly reserved for final evaluation.
- **Leakage Dropping (DEC-06):** Strictly drop failure mode indicators (`twf`, `hdf`, `pwf`, `osf`, `rnf`) and identifiers (`udi`, `product_id`).
- **Feature Engineering:** Include physical interaction features: `power_w = 2 * pi * speed * torque / 60`, `temp_diff = process_temp - air_temp`, and `strain = torque * tool_wear`.
- **Evaluation Governance (DEC-07):** Report PR-AUC, Recall @ 80% Precision, and Brier score against prevalence baseline $0.0339$. Accuracy is strictly invalid.

### For Day 8: Operational & Demand Forecasting
- **Seasonality Modeling (DEC-08):** Incorporate 7-day seasonal differencing ($D=1$, period 7) for production throughput and commercial revenue.
- **Baselines to Beat (DEC-09):** Benchmark models against MA-7 baseline: Fleet Daily Production MAE $\le 1,026.8$ units (11.5% MAPE).

---

## 9. Open Questions & Items Flagged for Downstream Days

1. **OQ-1 (Day 6 Factory Linkage):** How should the AI4I classifier interact with the factory fleet? Recommendation: Treat the AI4I model as a pre-trained milling machine classifier; score incoming fleet machines (e.g. presses/compressors) only if feature spaces are harmonized, or deploy it as an asset-specific subsystem.
2. **Native MetroPT Channels (Day 5 Stretch):** Raw MetroPT contains 15 channels (including `TP2` discharge pressure and `DV_pressure` desiccant valve pressure). Testing Day 5 Isolation Forest on native channels can provide additional real-world benchmark validation.
3. **Sunday Retail Forecasting Mask (Day 8):** Retail sales are strictly £0 on Sundays (52 days). Models on Day 8 should employ calendar masking or a two-stage hurdle model rather than trying to fit smooth continuous distributions through exact zeros.
