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

## 3. Machine Failure Risk Prediction Methodology (Day 6)

### 3.1 Operating State Framing (Decision D67)
AI4I 2020 lacks temporal timestamps, run-to-failure trajectories, or machine identifiers. The failure prediction model is mathematically framed as estimating the **instantaneous failure risk of an operating state**, rather than a prognostic time-to-failure horizon ("failure within $N$ days").

### 3.2 Leakage Prevention & Feature Engineering (Decisions D69, D70)
- **Forbidden Columns:** Target-composing failure mode indicators (`twf`, `hdf`, `pwf`, `osf`, `rnf`) and identifiers (`udi`, `product_id`) are strictly excluded from the feature space. Mode flags are reserved exclusively for post-evaluation diagnostic recall breakdowns.
- **Continuous Physics Quantities (F1):** Thermodynamic and mechanical quantities are engineered to allow gradient boosted trees to identify boundary regions without hand-coding dataset decision rules:
  $$\text{temp\_diff\_k} = \text{process\_temperature\_k} - \text{air\_temperature\_k}$$
  $$\text{power\_w} = \text{torque\_nm} \times \left(\text{rotational\_speed\_rpm} \times \frac{2\pi}{60}\right)$$
  $$\text{strain\_nmin} = \text{tool\_wear\_min} \times \text{torque\_nm}$$
  $$\text{type} = \text{OrdinalCode}(L=0, M=1, H=2)$$

### 3.3 Evaluation Protocol & Model Selection (Decisions D71, D72, D73)
- **Repeated Stratified Cross-Validation:** Due to high class imbalance (~3.4% failure prevalence), candidate models are evaluated using Repeated Stratified K-Fold (5 folds $\times$ 2 repeats = 10 evaluations) on training rows only.
- **Pre-Registered Random Search:** 50 random hyperparameter draws + 3 ablations evaluated under pre-registered bounds.
- **Cascading Selection Rule:** Highest pooled out-of-fold PR-AUC, breaking ties ($\Delta \le 0.002$) by lowest Brier score, then lowest model complexity ($\text{max\_depth} \times \text{n\_estimators}$). Deterministically selected `XGB-PHYS-08` (depth=4, 600 estimators, PR-AUC=0.8821).

### 3.4 Calibration & Risk Categorization (Decisions D74, D75)
- **Nested Calibrator Selection:** Isotonic regression outperformed Platt sigmoid with a +5.49% lower nested Brier score and lower ECE (0.0023 vs 0.0029).
- **Derived Operational Thresholds:**
  - $t_{\text{med}} = 0.0460$: Target 95% recall on training out-of-fold predictions.
  - $t_{\text{high}} = 0.5000$: Maximum F1 score ($F1 = 0.8647$) on training out-of-fold predictions.
  - $t_{\text{crit}} = 0.5000$: Enforces monotonicity clamp ($t_{\text{crit}} \ge t_{\text{high}}$).

### 3.5 Holdout Discipline & Empirical Recall Ceiling (Decisions D68, D82)
- **Pre-Registration Freeze:** Artifact hashes recorded in `models/failure_config_frozen.json` before single holdout access.
- **Single-Run Test Evaluation (N=2,000, 68 failures):** Test PR-AUC = 0.8756 [0.8060, 0.9366] vs. prevalence baseline 0.0340; Test ROC-AUC = 0.9775; Test Recall = 80.88%; Test Precision = 93.22%.
- **Empirical Recall Ceiling:** Inherent uniform random noise failures (`rnf`) exhibit zero precursor signals, bounding the theoretical test recall ceiling at **97.06%** (66/68 physical failures).

### 3.6 Replay Architecture for Demo Machines (Decision D78)
Demo machine states for the 12 fleet assets are generated via scenario mixture sampling with replacement strictly from the quarantined test partition, flagged with `is_replay = True`.


---

## 4. Time Series Demand & Production Forecasting Methodology (Day 8 Handoff)

- Hierarchical and SKU-level forecasting using 7-day seasonal differencing ($D=1$).
- Benchmark to beat: Fleet Production 7-Day Moving Average (MA-7) baseline MAE $\le 1,026.8$ units.
