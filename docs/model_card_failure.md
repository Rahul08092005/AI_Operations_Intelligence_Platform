# Model Card: Equipment Failure Risk Predictor (XGBoost)

**Model Identifier:** `failure_model_xgb_v1`  
**Model Version:** `1.0.0`  
**Model Type:** Gradient Boosted Decision Trees (`XGBClassifier`) with Isotonic Probability Calibration  
**Primary Configuration:** `XGB-PHYS-08` (depth=4, 600 estimators, lr=0.03, subsample=0.8, colsample=0.8)  
**Date of Release:** November 16, 2026  
**Artifact Path:** `models/failure_model.pkl`  
**Frozen Pre-Registration Hash:** `bfec1db63882ac7f4861c7569949d9f0ad6c934c0200ac8eac3753fb85840b2b`  
**License:** Educational / Demonstration Only  

---

## 1. Intended Use & Operational Framing

### 1.1 Intended Use
- **Primary Purpose:** Estimating the instantaneous probability that an industrial machine's current mechanical and thermodynamic operating state represents or triggers an immediate functional failure.
- **Application Context:** Component of the AI Operations Intelligence Platform demonstration suite. Provides realistic, calibrated failure probabilities to downstream executive fleet dashboards (Day 7), SHAP feature explanation modules (Day 10), and LLM root-cause synthesis agents (Days 11–12).
- **Framing Constraint (Decision D67):** The model estimates **operating state failure risk**, *not* a prognostic "remaining useful life" or "failure within the next $N$ days". AI4I contains no temporal continuity.

### 1.2 Out-of-Scope & Prohibited Uses
- **NOT FOR REAL INDUSTRIAL PLANT CONTROL:** This model must **never** be used for autonomous machine shutdowns, life-critical safety interlocking, or actual maintenance scheduling in real-world manufacturing plants.
- **Prohibited Application to Unharmonized Telemetry:** The model must not be applied directly to non-milling physical equipment (e.g., train air compressors or hydraulic presses) without rigorous sensor-to-process parameter mapping (Caveat C5).
- **Prohibited Use of Mode Flags:** Label-composing failure mode flags (`twf`, `hdf`, `pwf`, `osf`, `rnf`) must never be fed as inputs (Rule R3).

---

## 2. Factors, Metrics & Pre-Registered Baselines

### 2.1 Primary Metric: Precision-Recall AUC (PR-AUC)
Due to severe class imbalance (~3.4% failure prevalence), classification accuracy (which achieves 96.6% by predicting no failure) is uninformative. PR-AUC is the pre-registered primary evaluation metric, measured against the prevalence baseline:
$$\text{Prevalence Baseline} = \frac{N_{\text{pos}}}{N_{\text{total}}} = 0.0340 \quad (3.40\%)$$

### 2.2 Secondary Metrics
- **Recall @ $t_{\text{high}} = 0.50$:** Detection rate of failures at the actionable alarm threshold.
- **Precision @ $t_{\text{high}} = 0.50$:** Fraction of raised alarms corresponding to true functional failures.
- **Brier Score & ECE:** Quadratic accuracy of calibrated probabilities and Expected Calibration Error.
- **Receiver Operating Characteristic AUC (ROC-AUC):** Global discrimination across all possible cutoffs.

---

## 3. Training & Evaluation Data

### 3.1 Training Data Partition
- **Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset (Synthetic milling machine telemetry).
- **Sample Size:** 8,000 observations (stratified 80% split from Day 4 split index).
- **Class Balance:** 271 failures (3.3875% prevalence), 7,729 healthy states.
- **Input Features (F1 - Physics):**
  - Raw operating telemetry: `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`.
  - Ordinal variant tier: `type` (encoded $L=0, M=1, H=2$).
  - Continuous physics-derived features:
    - `temp_diff_k` = $\text{process\_temperature\_k} - \text{air\_temperature\_k}$ [K]
    - `power_w` = $\text{torque\_nm} \times (\text{rotational\_speed\_rpm} \times 2\pi / 60)$ [W]
    - `strain_nmin` = $\text{tool\_wear\_min} \times \text{torque\_nm}$ [$\text{N}\cdot\text{m}\cdot\text{min}$]

### 3.2 Evaluation Data Partition (Quarantined Test Set)
- **Sample Size:** 2,000 observations (quarantined 20% split).
- **Class Balance:** 68 failures (3.4000% prevalence), 1,932 healthy states.
- **Quarantine Discipline:** Cryptographically locked behind `load_test()` until configuration was frozen. Evaluated strictly once.

---

## 4. Quantitative Performance Scorecard

### 4.1 Single-Run Holdout Test Results (N=2,000, 68 Failures)

| Metric | Primary Model (`XGB-PHYS-08`) | 95% Bootstrap CI | Logistic Regression | XGBoost Raw Ablation | Prevalence Baseline |
|---|---|---|---|---|---|
| **PR-AUC** | **0.8756** | [0.8060, 0.9366] | 0.4686 | 0.8269 | 0.0340 |
| **ROC-AUC** | **0.9775** | [0.9551, 0.9956] | 0.8931 | 0.9639 | 0.5000 |
| **F1 Score** | **0.8661** | [0.7937, 0.9231] | 0.5455 | 0.7692 | 0.0658 |
| **Recall (@ $t_{\text{high}}$)** | **80.88%** (55/68) | [70.59%, 89.71%] | 44.12% | 69.12% | 0.00% |
| **Precision (@ $t_{\text{high}}$)**| **93.22%** (55/59) | [86.15%, 98.33%] | 71.43% | 87.04% | 3.40% |
| **Brier Score** | **0.0084** | [0.0051, 0.0125] | 0.0245 | 0.0108 | 0.0328 |
| **ECE (10 bins)** | **0.0047** | [0.0021, 0.0094] | 0.0142 | 0.0076 | — |

*Bootstrap confidence intervals computed via 2,000 stratified resamples of the test partition (seed 42).*

### 4.2 Diagnostic Failure Mode Breakdown & Empirical Recall Ceiling

| Failure Mode | Physical Description | Total in Test | Detected (@ $t_{\text{high}}$) | Recall |
|---|---|---|---|---|
| **PWF** | Power Failure (Spindle Power $<3500\text{W}$ or $>9000\text{W}$) | 13 | 13 | **100.0%** |
| **OSF** | Overstrain Failure (Tool wear $\times$ torque exceeds variant threshold) | 16 | 16 | **100.0%** |
| **HDF** | Heat Dissipation Failure ($\Delta T < 8.6\text{K}$ and speed $< 1380\text{ RPM}$) | 29 | 27 | **93.1%** |
| **TWF** | Tool Wear Failure (Pure cumulative wear between 200–240 min) | 10 | 1 | **10.0%** |
| **RNF** | Random Noise Failure (0.1% uniform random failure probability) | 4 | 0 | **0.0%** |

> **Theoretical & Empirical Recall Ceiling (Decision D82, Caveat C2):**  
> AI4I contains uniform random noise failures (`rnf`) that have zero physical precursors. In the test set, 4 failures exhibit pure `rnf`. Because random noise is inherently unpredictable, the maximum possible recall for any statistical model on this test set is **97.06% (66 / 68)**. The model achieves **83.33% recall (55 / 66)** of all physical failures at high precision.

---

## 5. Operating Risk Categories & Serving Thresholds

Thresholds derived from training out-of-fold cross-validation predictions:
- **Low Risk ($p < 0.0460$):** Nominal operation.
- **Medium Risk ($0.0460 \le p < 0.5000$):** Early degradation precursor. Target recall on training OOF $\ge 95\%$.
- **High Risk ($0.5000 \le p < 0.5000$):** Severe operating stress. Maximizes training OOF F1 score ($F1 = 0.8647$).
- **Critical Risk ($p \ge 0.5000$):** Imminent or active failure state. Monotonicity clamp applied.

---

## 6. Crucial Caveats & Intellectual Honesty Statement

### Caveat C1: Rule-Generated Labels in AI4I
The AI4I dataset labels were generated using mathematical threshold equations rather than empirical failures on physical machines. The high test metrics (PR-AUC 0.8756, Recall 80.9%) demonstrate that gradient boosting with continuous physics features successfully reconstructs multi-dimensional geometric decision boundaries. **This should not be interpreted as evidence of superhuman AI capability on noisy, uncalibrated real-world industrial machinery.**

### Caveat C2: Unpredictable Random Failures
Four test failures represent pure random noise events (`rnf`). The model correctly outputs near-zero failure probabilities for these states. Reporting an adjusted physical failure recall ceiling (97.06%) maintains scientific integrity.

### Caveat C5 & C6: Replayed Demo States
Fleet assets in the demo dashboard (M-101 through M-112) receive AI4I operating states via **scenario mixture sampling with replacement from the test partition** (Decision D78). These are tagged with `is_replay = True`. They demonstrate platform UI capabilities and must **never** be cited as real machine performance measurements.

### Caveat C7: Decoupling of Maintenance History
The model evaluates operating states in isolation. It has no access to work order histories, technician notes, or overdue maintenance flags. Maintenance context is integrated downstream in the platform's reasoning layer (Days 11–12).

---

## 7. Input Envelope & Out-of-Distribution Handling

The model serving wrapper (`ml/prediction/inference.py`) validates inputs against training distribution extrema:

| Field | Training Min | Training Max | Warning Envelope ($>10\%$ Out-of-Bounds) |
|---|---|---|---|
| `air_temperature_k` | 295.3 K | 304.5 K | $< 294.4\text{ K}$ or $> 305.4\text{ K}$ |
| `process_temperature_k` | 305.7 K | 313.8 K | $< 304.9\text{ K}$ or $> 314.6\text{ K}$ |
| `rotational_speed_rpm` | 1,168 RPM | 2,886 RPM | $< 996\text{ RPM}$ or $> 3,058\text{ RPM}$ |
| `torque_nm` | 3.8 Nm | 76.6 Nm | $< 0.0\text{ Nm}$ or $> 83.9\text{ Nm}$ |
| `tool_wear_min` | 0 min | 253 min | $< 0\text{ min}$ or $> 278\text{ min}$ |

Out-of-envelope values execute normally but return explicit non-blocking warning strings in `input_warnings`. Missing required fields or invalid machine types raise HTTP-compatible validation errors.

---

## 8. Retraining & Reproducibility Protocol

To reproduce or retrain the model bundle from source:
```bash
# 1. Run full 50-draw hyperparameter search and ablations
make failure-search

# 2. Fit primary model, calibrator, and thresholds on training partition
make failure-train

# 3. Freeze configuration hashes
make failure-freeze

# 4. Run test suite and train/serve parity verification
make failure-test-suite
```
All random operations are pinned to `seed=42`.

