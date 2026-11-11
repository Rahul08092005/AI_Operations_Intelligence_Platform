# Methodology Specification: AI Operations Intelligence Platform

## 1. Overview & Architectural Principles

The AI Operations Intelligence Platform applies machine learning to industrial telemetry and operations data under strict reproducibility, temporal integrity, and scientific rigor.

### Key Governance Rules
1. **Rule R1 (Reproducibility):** Deterministic random seeds (`seed=42`) across all components.
2. **Rule R2 (Temporal Discipline):** Strict chronological partitioning. Model exploration, feature selection, and hyperparameter tuning occur exclusively within the **Exploration Partition** (first 70% of days; `2026-03-21` to `2026-08-17 00:00 UTC`). The **Holdout Partition** (`2026-08-17` to `2026-10-20`) is evaluated exactly once from cryptographically frozen configurations.
3. **Rule R3 (No Leakage):** Normalization baselines, score calibration curves, and feature scalers are fitted exclusively on training rows. Tampering with holdout data produces identical fitted models.
4. **Rule R9 (Intellectual Honesty):** Never tune models to fit demo expectations. Side-by-side reporting of full-feature models against real-signal-only comparators is mandatory.

---

## 2. Anomaly Detection Methodology (Day 5)

### 2.1 Model Architecture & Formulation (Decision D53)
The anomaly detection engine employs a single global **Isolation Forest** trained on pooled running intervals across all 12 fleet machines. To eliminate baseline discrepancies between asset types, features are normalized via per-machine robust statistics:
$$z_{m, f} = \frac{x_{m, f} - \text{median}_{m, f}}{\max(1.4826 \times \text{MAD}_{m, f}, \epsilon)}$$

### 2.2 Operating State Gating (Decision D54)
Telemetry scoring is gated by machine operating state:
- When `running == True` and features are complete $\to$ State `Running`, scored through Isolation Forest.
- When `running == False` $\to$ State `Offline`, score is `NaN`, status is `Offline`.
- When warm-up, data gap, sensor fault, or missing values exist $\to$ State `NoData`, score is `NaN`, status is `NoData`.

### 2.3 Score Calibration & Smoothing (Decision D57)
Raw tree anomaly scores $s_{\text{raw}} = -\text{score\_samples}(X)$ are calibrated via piecewise linear interpolation anchored to normal training quantiles:
- $\text{median} \to 0.10$ (nominal operating baseline)
- $q_{\text{warn}} (0.99) \to 0.50$ (Warning alert threshold)
- $q_{\text{crit}} (0.9999) \to 0.80$ (Critical alert threshold)
- $\text{saturation} \to 1.00$

Calibrated scores are smoothed using a causal trailing rolling mean (window 36 = 3 hours) with gap reset, preventing alert chatter while maintaining causal temporal integrity.

### 2.4 Event-Level Evaluation & False Alarm Budget (Decisions D58, D60, D62)
- **Episodes:** Warning and Critical intervals are grouped into episodes (minimum 3 intervals / 15 minutes, merged across gaps $< 2.0$ hours).
- **False Alarm Budget:** Evaluated outside problem periods: $\le 1.0$ Warning-or-higher episode / machine / 30 days, exactly $0$ Critical episodes outside problem periods, and $0$ Critical episodes on benign spike benchmark `M-109`.
- **Real-Window Metrics:** Evaluated against authentic failure windows on `M-104` with a 72-hour lead time horizon: Early ($[start - 72\text{h}, start)$), Late ($[start, end]$), or Missed.

---

## 3. Failure Prediction & Prognostics Methodology (Day 6 Handoff)

- Supervised classification using LightGBM / XGBoost on AI4I 2020 synthetic manufacturing telemetry (`data/processed/ai4i_split_index.csv`).
- Strict exclusion of failure mode indicators (`twf`, `hdf`, `pwf`, `osf`, `rnf`) and identifiers (`udi`, `product_id`).
- Primary metric: PR-AUC evaluated against the naive minority base rate ($0.0339$).

---

## 4. Time Series Demand & Production Forecasting Methodology (Day 8 Handoff)

- Hierarchical and SKU-level forecasting using 7-day seasonal differencing ($D=1$).
- Benchmark to beat: Fleet Production 7-Day Moving Average (MA-7) baseline MAE $\le 1,026.8$ units.
