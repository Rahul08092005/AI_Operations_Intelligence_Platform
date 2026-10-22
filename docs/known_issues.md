# Known Issues and Data Lineage Log

**Project:** AI Operations Intelligence Platform  
**Owner:** Data Engineering & EDA  
**Status:** Active  
**Last Updated:** October 2026 (Day 4 EDA)

---

## 1. Overview
This document tracks data, pipeline, and generator discrepancies identified during Exploratory Data Analysis (EDA). In accordance with Rule R9 (Day 4 Specification), pipeline fixes are exceptional and restricted to small Day 3 pipeline bugs. Issues originating from Day 2 data generator designs or external datasets are logged here with root-cause diagnoses, impact analyses, and downstream modeling implications.

---

## 2. Issues Register

| ID | Title | Component / Source | Evidence | Severity | Status | Downstream Owner / Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **KI-01** | **Weekend Production Factor Discrepancy** | `generate_production.py` (Day 2) | Per-machine weekend-to-weekday ratio median is **0.957** (range 0.905–1.094) vs. planned 0.60 ± 0.08 | Low | **Logged (By Design)** | Day 8 Forecasting: Model reflects true telemetry availability (~0.95 ratio) rather than an artificial 0.60 shutdown schedule. |
| **KI-02** | **Routine Inspection Spacing Delta** | `generate_maintenance.py` (Day 2) | Median operating hours between consecutive inspections is **750.0h** vs. planned 500h ± 15% (425–575h) | Low | **Logged (By Design)** | Day 5 Anomaly Detection / Day 7 Copilot: Generator scheduled milestone offsets at `[550h, 1300h]`, yielding a 750h interval. |
| **KI-03** | **Preventive Maintenance Interval Scarcity** | `generate_maintenance.py` (Day 2) | Only 2 machines had multiple PMs in 213 days; median interval was **550.4h** vs. planned 2,000h | Medium | **Logged (Horizon Constraint)** | Day 6 Failure Prediction: Total operating span (~3,400 running hours) provides sparse multi-PM observation. Use cumulative operating hours rather than PM cycle counts. |
| **KI-04** | **Absence of Precursor in Real Telemetry** | MetroPT-3 Real Compressor Telemetry (`M-104`) | Unmodified real channels (`pressure`, `motor_current`, `comp_state`) show zero statistically significant precursor lead time ($Z > 3$ for 6h) | High | **Architecture Finding** | Day 5 Anomaly Detection: Anomaly detection on real channels cannot rely on high-magnitude precursors; must evaluate against subtle multivariate drift and native discharge channels (`TP2`, `DV_pressure`). |

---

## 3. Detailed Diagnoses and Impact Analyses

### KI-01: Weekend Production Factor
- **Description:** Day 2 specification suggested a weekend production factor of 0.60. However, the generator calculated units produced directly from equipment running telemetry: `units = target * (operating_hours / 24.0) * efficiency`.
- **Root Cause:** MetroPT air compressors operate on continuous duty cycles including weekends (averaging ~15.5 operating hours on weekends vs. ~16.5 on weekdays). Consequently, equipment availability is ~65% on weekends vs. ~69% on weekdays, resulting in an empirical weekend-to-weekday ratio of 0.957.
- **Decision:** Preserve telemetry integrity. Do not inject artificial weekend downtime into processed data. Forecasting models on Day 8 will capture the empirical weekly pattern directly.

### KI-02: Routine Inspection Spacing
- **Description:** The median interval between consecutive routine inspections across machines was measured at 750.0 operating hours rather than the 500-hour nominal specification.
- **Root Cause:** In `ml/preprocessing/generate_maintenance.py`, inspection milestone offsets were initialized as `intervals = [550.0, 1300.0, 2050.0]`. The difference between milestone 1 (550h) and milestone 2 (1300h) is mathematically $1300 - 550 = 750\text{ hours}$.
- **Decision:** As this is an immutable characteristic of the synthetic maintenance log, `operations_dataset.parquet` tracks elapsed operating hours continuously.

### KI-03: Preventive Maintenance Interval Scarcity
- **Description:** A nominal 2,000-hour preventive maintenance rule implies only 1.7 PM cycles across a 3,400-hour timeline.
- **Root Cause:** A 7-month calendar horizon with ~65% machine duty yields at most one scheduled 2,000h PM per machine. Consecutive PM differences could only be evaluated on two machines with early anchor services.
- **Decision:** Feature store correctly tracks `operating_hours_since_preventive` and `preventive_overdue`.

### KI-04: Absence of Precursor in Real Compressor Telemetry
- **Description:** In the M-104 real failure window event study (EQ26, EQ27), unmodified sensor channels (`pressure`, `motor_current`, `comp_state`) showed no discernible change prior to failure start ($h=-72$ to $0$).
- **Root Cause:** Real-world compressor air leaks in the MetroPT-3 dataset developed rapidly or exhibited symptoms in native discharge channels (`TP2`, `DV_pressure`) that were not included in the primary 6-channel synthetic schema.
- **Decision:** Handed off to Day 5: Isolation Forest must be evaluated on both (a) all features (including synthetic overlay) and (b) **real-only features**. Modelers must not present (a) alone as real-world detection capability.
