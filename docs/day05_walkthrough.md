# Day 5 Walkthrough: Anomaly Detection with Isolation Forest

**Project:** AI Operations Intelligence Platform  
**Branch:** `feature/day-05-anomaly-detection`  
**Date:** November 2026  
**Audience:** Platform Engineering, Lead Data Scientist, Technical Evaluator  

---

## 1. EDA Handoff & Stop Condition Verifications

### 1.1 Decision Register Handoff (from `docs/eda_findings.md`)
- **DEC-01 (Anomaly Features):** Candidate feature sets defined (A: `plan_core`, B: `core_context`, C: `core_temporal`, D: `real_only`).
- **DEC-02 (Normal Training Set):** Train on exploration period normal rows only, strictly masking problem episodes plus 24-hour buffer.
- **DEC-03 (Dual Evaluation Protocol):** Mandatory side-by-side reporting of the full-feature model and the `real_only` comparator.
- **DEC-04 (False Alarm Stress Test):** M-109 (benign spikes) serves as the false-alarm stress test with 0 Critical alerts and Warning rate within budget.

### 1.2 Telemetry Base Rates and Contamination Range
- **Tier 1 Physical Limit Faults:** 0.00% across all machines.
- **Tier 2 Outlier Flag Rates ($|Z| > 6.0$):** Fleet average 32.7% (reflecting compressor load/unload cycles, not faults).
- **Contamination Directive:** Do **not** set contamination to 32%. Recommended range: $0.01$ to $0.05$ (true anomalies constitute 2% to 4% of intervals). Pre-registered choice: `contamination="auto"` for fitting, with calibrated percentile thresholds ($q_{\text{warn}}$, $q_{\text{crit}}$).

### 1.3 Threshold Proposals (from Exploration Healthy Equipment)
- **Air Compressors:** Healthy p99.9 vibration = 3.59 mm/s, temp = 82.97 °C. Warning: 4.0–4.5 mm/s, Critical: 7.0 mm/s.
- **Hydraulic Presses:** Healthy p99.9 vibration = 3.95 mm/s, temp = 81.40 °C. Warning: 4.2–4.5 mm/s, Critical: 7.0 mm/s.
- **Conveyor Drives:** Healthy p99.9 vibration = 3.23 mm/s, temp = 79.80 °C. Warning: 3.8–4.5 mm/s, Critical: 7.0 mm/s.
- **B1 Baseline Specification:** Vibration Warning at 4.5 mm/s, Critical at 7.0 mm/s; Temperature Warning at 85.0 °C.

### 1.4 Real Failure Windows by Period (M-104 MetroPT)
| Failure Window | Start Time (UTC) | End Time (UTC) | Partition | Hours |
|:---|:---|:---|:---|:---|
| **FAIL-001** | 2026-06-03 00:00:00 | 2026-06-06 23:55:00 | Exploration | 95.9h |
| **FAIL-002** | 2026-07-14 23:30:00 | 2026-07-18 06:00:00 | Exploration | 78.5h |
| **FAIL-003** | 2026-07-21 10:00:00 | 2026-07-26 14:30:00 | Exploration | 124.5h |
| **FAIL-004** | 2026-08-30 14:30:00 | 2026-09-02 19:00:00 | Holdout | 76.5h |

**Stop Conditions Check:** Exactly 3 real failure windows exist in exploration, 1 in holdout. Total = 4 ($\ge 2$). Stop condition passed.

---

## 2. Architecture & Decisions (D53 to D66)
- **D53 (Model Structure):** One global Isolation Forest per feature set, trained on pooled running normal rows from all machines, normalized via per-machine robust baselines (median and $1.4826 \times \text{MAD}$).
- **D54 (Operating State):** Only running rows with complete features scored. Non-running or gap rows receive state `Offline` or `NoData`.
- **D55 (Time Protocol):** 3-fold expanding-window CV inside exploration period with 24-hour safety gaps. Final refit on full exploration period. Single holdout evaluation.
- **D56 (Training Exclusions):** Exclude labeled problem episodes + 24-hour buffer. Retain `normal` and `benign_spike`.
- **D57 (Score Calibration):** Raw score mapping anchored to normal quantiles (median $\to 0.10$, $q_{\text{warn}} \to 0.50$, $q_{\text{crit}} \to 0.80$, saturation $\to 1.00$). 1h trailing smoothing. Status: Normal (<0.50), Warning (0.50–0.79), Critical ($\ge 0.80$).
- **D58 (Alert Episodes):** Contiguous Warning+ intervals $\ge 3$ periods (15 min), merging gaps $< 2\text{h}$.
- **D59 (Model Pool):** Feature sets A, B, C compete for primary; D (`real_only`) is a mandatory comparator.
- **D60 (Baselines):** B1 (manual thresholds) and B2 (robust z-score max).
- **D61 (Contributing Factors):** Top 3 model features by absolute robust z-score, with direction and magnitude.
- **D62 (Metrics):** Event-level detection, lead time, false alarms per machine per 30d, interval AP.
- **D63 (Experiment Budget):** Maximum 40 training runs.
- **D64 (Holdout Protocol):** Frozen config with SHA-256 artifact hashes. Single evaluation run.
- **D65 (Train/Serve Parity):** Shared feature engineering code via `ml.preprocessing.features_sensor`.
- **D66 (Database Schema):** Peak episode scores and types loaded into `anomalies` table.

---

*(Sections 3–8 will be populated as tasks T2–T14 are executed)*
