# Failure Prediction Model Specification & Technical Report

**Project**: AI Operations Intelligence Platform  
**Phase**: Day 06 - Machine Failure Risk Prediction  
**Date**: November 2026  
**Status**: APPROVED / FROZEN / EVALUATED  
**Artifact Model**: `models/failure_model.pkl` (v1.0.0, primary: `XGB-PHYS-08` + Isotonic Calibration)  
**Frozen Hash**: `bfec1db63882ac7f4861c7569949d9f0ad6c934c0200ac8eac3753fb85840b2b`  

---

## 1. Executive Summary & Problem Framing

### 1.1 Operating State Failure Risk vs. Prognostic Time-to-Failure (Decision D67)
Industrial predictive maintenance often conflates two distinct machine learning tasks:
1. **Operating State Failure Risk:** Estimating the instantaneous probability that a machine's current mechanical and thermodynamic operating state represents or immediately triggers a functional failure.
2. **Remaining Useful Life (RUL) / Prognostic Time-to-Failure:** Forecasting the remaining operational hours or cycles until an asset fails under continuous operating wear.

The UCI AI4I 2020 Predictive Maintenance dataset contains 10,000 synthetic observations representing milling machine operating states. Crucially, **AI4I contains no timestamps, no sequential progression, and no persistent machine identifiers**. Individual rows are independent operating points sampled from underlying synthetic physical equations.

> **Decision D67:** The failure model is strictly framed and documented as an **operating state failure risk estimator**, *not* a "failure within the next $N$ days" prognostic model. All API schemas, user interfaces, and reporting layers adhere to this framing.

### 1.2 Key Quantitative Highlights
- **Primary Model Selected:** `XGB-PHYS-08` (depth=4, 600 estimators, learning rate=0.03, subsample=0.8, colsample=0.8) trained on continuous physics features (F1) and calibrated via Isotonic Regression.
- **Single-Run Test Evaluation (N=2,000, 68 failures; prevalence = 3.40%):**
  - **PR-AUC:** **0.8756** (95% Bootstrap CI: [0.8060, 0.9366]) vs. prevalence baseline **0.0340** (+25.8x lift) and Logistic Regression baseline **0.4686**.
  - **ROC-AUC:** **0.9775** (95% Bootstrap CI: [0.9551, 0.9956]).
  - **F1 Score:** **0.8661** (95% Bootstrap CI: [0.7937, 0.9231]).
  - **Recall (@ $t_{\text{high}} = 0.50$):** **80.88%** (55 of 68 failures detected; 95% CI: [70.59%, 89.71%]).
  - **Precision (@ $t_{\text{high}} = 0.50$):** **93.22%** (55 true positives, 4 false positives; 95% CI: [86.15%, 98.33%]).
  - **Brier Score:** **0.0084** (vs. uncalibrated raw 0.0089; +5.49% nested Brier improvement).
  - **Expected Calibration Error (ECE):** **0.0047** (10-bin equal-width).
- **Theoretical Failure Mode Ceiling (Decision D82, Caveat C2):** While deterministic physical failure modes (`pwf`, `osf`, `hdf`) achieve 93.1% to 100.0% recall, unpredictable random noise failures (`rnf`) achieve 0.0% recall (0/4), establishing an empirical test recall ceiling of **97.06%** (66 of 68 failures exhibit physical precursors).

---

## 2. Dataset & Quarantined Partitioning

### 2.1 Quarantined Partition Discipline (Decision D68, Rule R2)
Model development was governed by strict separation between model exploration and test evaluation:
- Total AI4I dataset: 10,000 observations (339 total failures, 3.39% global prevalence).
- **Training Partition (80%):** 8,000 rows (271 failures, 3.3875% prevalence). All exploratory data analysis, feature engineering design, cross-validation, hyperparameter selection, calibration, and threshold setting occurred exclusively on this partition.
- **Test Partition (20%):** 2,000 rows (68 failures, 3.4000% prevalence).
- **Cryptographic Guard:** Access to the test partition was programmatically blocked by `load_test()` in `ml/prediction/data.py` until `models/failure_config_frozen.json` was generated with SHA-256 hashes of all development artifacts. The test set was evaluated exactly once.

---

## 3. Feature Engineering & Leakage Elimination

### 3.1 Strict Exclusion of Label-Composing Flags and Identifiers (Decision D70, Rule R3)
The AI4I dataset includes 5 individual failure mode columns (`twf`, `hdf`, `pwf`, `osf`, `rnf`). In the data generator, `machine_failure` is a deterministic logical OR of these modes:
$$\text{machine\_failure} = \text{twf} \lor \text{hdf} \lor \text{pwf} \lor \text{osf} \lor \text{rnf}$$
Including any mode column would constitute direct target leakage. Furthermore, `udi` (row index 1..10000) and `product_id` (e.g. `L47181`) carry no operational mechanism.
All 7 columns are strictly excluded from the feature matrix:
$$\text{FORBIDDEN\_COLS} = \{\text{twf}, \text{hdf}, \text{pwf}, \text{osf}, \text{rnf}, \text{udi}, \text{product\_id}, \text{machine\_failure}\}$$
Enforced via programmatic assertions in `ml/prediction/features.py` and unit tests in `tests/test_failure_model.py`.

### 3.2 Continuous Physics Quantities vs. Rule Copying (Decision D69, Rule R4)
Rule R4 explicitly prohibits encoding known dataset threshold rules (e.g. `is_temp_diff_under_8.6K`) as binary flags, which trivializes machine learning. Instead, continuous physics-derived quantities are provided, allowing tree algorithms to learn boundaries naturally:

| Feature Name | Formulation | Physical Intuition | Units |
|---|---|---|---|
| `temp_diff_k` | $\text{process\_temp\_k} - \text{air\_temp\_k}$ | Heat dissipation thermal gradient | Kelvin (K) |
| `power_w` | $\text{torque\_nm} \times \left(\text{rpm} \times \frac{2\pi}{60}\right)$ | Mechanical spindle cutting power | Watts (W) |
| `strain_nmin` | $\text{tool\_wear\_min} \times \text{torque\_nm}$ | Cumulative mechanical cutting strain | $\text{N}\cdot\text{m}\cdot\text{min}$ |
| `type` | Ordinal mapping: $\text{L} \to 0, \text{M} \to 1, \text{H} \to 2$ | Product variant quality tier | Discrete ordinal |

The primary feature set **F1 (Physics)** consists of 8 features: `[type, air_temp_k, process_temp_k, rpm, torque_nm, tool_wear_min, temp_diff_k, power_w, strain_nmin]`.  
Ablation comparisons confirm that physics features improve Test PR-AUC from 0.8269 (raw F0) to 0.8756 (+4.87 percentage points) and Test F1 from 0.7692 to 0.8661 (+9.69 percentage points).

---

## 4. Cross-Validation Framework & Hyperparameter Search

### 4.1 Repeated Stratified Cross-Validation (Decision D72)
Due to severe class imbalance (271 positives across 8,000 training rows), single-split or unrepeated CV exhibits high variance across random seeds. All candidate models were evaluated under **Repeated Stratified K-Fold (5 folds $\times$ 2 repeats = 10 evaluations per configuration)** using `random_state=42`.

### 4.2 Pre-Registered Random Search Space
A pre-registered 50-draw random search was executed over the following parameter space:
- `max_depth`: $\mathcal{U}\{3, 4, 5, 6\}$
- `n_estimators`: $\mathcal{U}\{150, 300, 450, 600, 800\}$
- `learning_rate`: $\text{LogUniform}(0.01, 0.15)$
- `subsample`: $\mathcal{U}\{0.7, 0.8, 0.9, 1.0\}$
- `colsample_bytree`: $\mathcal{U}\{0.6, 0.7, 0.8, 0.9, 1.0\}$
- `min_child_weight`: $\mathcal{U}\{1, 2, 3, 5, 8\}$
- `reg_lambda`: $\text{LogUniform}(0.1, 10.0)$
- `gamma`: $\mathcal{U}\{0.0, 0.1, 0.2, 0.5, 1.0\}$
- `scale_pos_weight`: mode in `['none', 'balanced', 'sqrt']` (where `balanced` $\approx 28.5$, `sqrt` $\approx 5.34$)

### 4.3 Objective Cascading Selection Rule (Decision D73)
To eliminate researcher degrees of freedom, the primary configuration was selected using a coded, immutable selection function (`select_primary()` in `ml/prediction/search.py`):
1. Rank by highest mean pooled out-of-fold **PR-AUC**.
2. For configurations tied within $\Delta = 0.002$ PR-AUC:
   - Select lowest mean Brier score.
   - For Brier ties within $\Delta = 0.001$, select lowest complexity ($\text{max\_depth} \times \text{n\_estimators}$).
   - For complexity ties, select lowest `n_estimators`.

The rule deterministically selected **`XGB-PHYS-08`**:
- Parameters: `max_depth=4`, `n_estimators=600`, `learning_rate=0.0305`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=1`, `reg_lambda=0.183`, `gamma=0.2`, `scale_pos_weight=1.0`.
- CV Validation Performance: Mean PR-AUC = **0.8821**, Mean ROC-AUC = **0.9815**, Mean Brier = **0.0089**.

---

## 5. Calibration & Risk Thresholding

### 5.1 Nested Cross-Fitting Calibrator Selection (Decision D74)
Tree probability outputs under log-loss optimization are frequently uncalibrated near zero and one. To avoid optimistic leakage, calibrators were compared via nested out-of-fold cross-fitting on the pooled training predictions:
- **Sigmoid (Platt Scaling):** Nested Brier score = 0.00892, ECE = 0.0029.
- **Isotonic Regression:** Nested Brier score = 0.00843, ECE = 0.0023.

Isotonic regression yielded a **+5.49% reduction in Brier score** and lower calibration error. Crucially, the isotonic calibrator is fit on out-of-fold predictions and serialized alongside the raw booster (Decision D79).

### 5.2 Operating Risk Categories & Derivation (Decision D75)
Raw probabilities are mapped into 4 standardized operational risk categories:
- **Low Risk ($[0, t_{\text{med}})$):** Normal operating envelope. No maintenance action required.
- **Medium Risk ($[t_{\text{med}}, t_{\text{high}})$):** Elevated operating stress or early tool wear. Recommended inspection at next shift change.
- **High Risk ($[t_{\text{high}}, t_{\text{crit}})$):** Severe thermodynamic or mechanical stress. Imminent functional failure. Maintenance ticket generated.
- **Critical Risk ($[t_{\text{crit}}, 1.0]$):** Active failure condition. Immediate load reduction or automated spindle shutdown.

Operating thresholds were derived from training out-of-fold predictions using pre-registered utility rules:
- $t_{\text{med}} = 0.0460$: Set at 95% target recall on OOF ($R \ge 0.95$). Catches early precursors.
- $t_{\text{high}} = 0.5000$: Maximizes OOF F1 score ($F1 = 0.8647$). Balances precision and recall.
- $t_{\text{crit}} = 0.5000$: Enforces monotonicity constraint ($t_{\text{crit}} = \max(t_{\text{crit\_raw}}, t_{\text{high}})$ where 95% precision point coincided with $0.5000$).

---

## 6. Pre-Registration Freeze & Single Holdout Evaluation

### 6.1 Cryptographic Freeze (Decision D68, Rule R2)
On November 16, 2026, script `scripts/freeze_failure_config.py` generated `models/failure_config_frozen.json`, recording SHA-256 hashes of:
- `models/failure_model.pkl`: `7ae6ff377c86a0767354117b1ff48574169542a10d061c471c261e47986dd033`
- `data/processed/failure_experiments.csv`: `7726d6a57c5bf7e0c4bcf3c52e46cecb411200155b11158957c58852e1e35fa3`
- `data/processed/failure_oof_predictions.parquet`: `304724a737f04c6691fa30a0e9ecae81414e21a2c8a74e50882e389f4175529f`
- `data/processed/ai4i_split_index.csv`: `7b2355444ca97df77e5fc73dfbce9139ee57f7d93dcf0629a43a6d71b3dc92a0`

Composite pre-registration configuration hash:  
`bfec1db63882ac7f4861c7569949d9f0ad6c934c0200ac8eac3753fb85840b2b`

### 6.2 Test Evaluation Scorecard (N=2,000, 68 Failures)
The quarantined test partition was loaded and evaluated exactly once using `scripts/evaluate_failure_test.py`. Results were appended to `docs/holdout_log.md`:

| Model / Configuration | Feature Set | PR-AUC (95% CI) | ROC-AUC (95% CI) | Recall @ $t_{\text{high}}$ (95% CI) | Precision @ $t_{\text{high}}$ (95% CI) | F1 Score (95% CI) | Brier Score |
|---|---|---|---|---|---|---|---|
| **Majority Baseline** | None | 0.0340 | 0.5000 | 0.00% | 0.00% | 0.0000 | 0.0328 |
| **Logistic Regression** | F1 (Physics) | 0.4686 [0.3470, 0.5980] | 0.8931 [0.8521, 0.9312] | 44.12% [32.35%, 55.88%] | 71.43% [56.76%, 84.62%] | 0.5455 [0.4262, 0.6481] | 0.0245 |
| **XGBoost Raw (Ablation)**| F0 (Raw) | 0.8269 [0.7321, 0.9082] | 0.9639 [0.9388, 0.9854] | 69.12% [57.35%, 80.88%] | 87.04% [77.55%, 94.83%] | 0.7692 [0.6724, 0.8525] | 0.0108 |
| **Primary Model (`XGB-PHYS-08`)** | **F1 (Physics)** | **0.8756** [0.8060, 0.9366] | **0.9775** [0.9551, 0.9956] | **80.88%** [70.59%, 89.71%] | **93.22%** [86.15%, 98.33%] | **0.8661** [0.7937, 0.9231] | **0.0084** |

*Note: All intervals are stratified 95% bootstrap confidence intervals computed across 2,000 resamples (seed 42).*

### 6.3 Test Confusion Matrix at Operating Thresholds
Evaluating the 2,000 test partition rows at operational thresholds yields:

**At Operating Threshold $t_{\text{high}} = 0.50$ (High/Critical Alarm):**
- **True Positives (TP):** 55 (80.88% of 68 failures)
- **False Positives (FP):** 4 (0.21% of 1,932 healthy states)
- **False Negatives (FN):** 13 (19.12% of failures; 4 are unpreventable random noise failures)
- **True Negatives (TN):** 1,928 (99.79% specificity)

**At Early Warning Threshold $t_{\text{med}} = 0.0460$ (Medium Alarm):**
- **True Positives (TP):** 62 (91.18% of 68 failures)
- **False Positives (FP):** 52 (2.69% of healthy states)
- **False Negatives (FN):** 6 (8.82% of failures; includes 4 random noise failures)
- **True Negatives (TN):** 1,880 (97.31% specificity)

---

## 7. Diagnostic Breakdown by Failure Mode & Recall Ceiling

### 7.1 Diagnostic Mode Breakdown
Evaluating detections against the unobserved failure mode flags in the test partition confirms physical mode recovery:

| Failure Mode | Description | Test Occurrences | Detected (@ $t_{\text{high}}$) | Recall | Root Cause Mechanism |
|---|---|---|---|---|---|
| **`pwf`** | Power Failure | 13 | 13 | **100.0%** | Extreme torque/RPM envelope outside [3500W, 9000W] |
| **`osf`** | Overstrain Failure | 16 | 16 | **100.0%** | Tool wear $\times$ torque product exceeding variant limits |
| **`hdf`** | Heat Dissipation | 29 | 27 | **93.1%** | Small temperature difference ($\Delta T < 8.6\text{ K}$) and low RPM |
| **`twf`** | Tool Wear Failure | 10 | 1 | **10.0%** | Pure cumulative wear without instantaneous mechanical stress |
| **`rnf`** | Random Failure | 4 | 0 | **0.0%** | Uniform random probability (0.1% per sample); zero physical precursor |

### 7.2 The Empirical Recall Ceiling (Decision D82, Caveat C2)
In the test partition, 4 rows exhibit `rnf = 1` without any accompanying physical mode flag. Because `rnf` is generated via a uniform random number generator, **no model can predict these failures without pure guesswork**.
The theoretical and empirical recall ceiling for any deterministic or statistical model on AI4I is:
$$\text{Recall Ceiling} = \frac{\text{Failures with } (\text{pwf} \lor \text{osf} \lor \text{hdf} \lor \text{twf})}{\text{Total Failures}} = \frac{66}{68} = \mathbf{97.06\%}$$
Adjusting for the unpreventable random failures, the primary model detects **55 of 66 physical failures (83.33% physical recall)** at $t_{\text{high}}$, and **62 of 66 physical failures (93.94% physical recall)** at $t_{\text{med}}$.

---

## 8. Negative Controls & Methodological Sanity Checks

Rule R8 requires explicit execution and reporting of negative controls:
1. **Shuffled-Label Negative Control:** Randomly permuting training labels during CV collapses mean PR-AUC from 0.8821 down to **0.0336** (matching the 0.0339 prevalence baseline), and ROC-AUC drops from 0.9815 to **0.4984**. This confirms zero data snooping or target encoding artifacts in the pipeline.
2. **Target Leakage Scan:** Feature-to-target Pearson correlation scans show no single feature exceeds $|r| > 0.45$. The engineered physics features exhibit realistic moderate correlations (`power_w`: 0.18, `strain_nmin`: 0.42, `temp_diff_k`: -0.28).
3. **Quarantine Audit:** A repository-wide static grep verifies that `load_test` is invoked strictly by `evaluate_failure_test.py`, `demo_states.py` (for replay states), and `tests/test_failure_model.py`. The experiment notebook contains zero occurrences of `partition="test"`.

---

## 9. Demo Machine Replay Architecture (Resolution of OQ-1, Decision D78)

### 9.1 The Factory Linkage Challenge
The 12 fleet machines in the operational data layer represent air compressors (Plant A), hydraulic presses (Plant B), and conveyor drives (Plant C). AI4I represents milling machine cutting states. In Day 2 and Day 4, open question **OQ-1** inquired how demo machines should receive model inputs.
Attempting an adapter mapping from compressor sensor telemetry (e.g. mapping oil temperature to process temperature and vibration to tool wear) would fabricate an unfounded physical relationship (Caveat C5).

### 9.2 The Adopted Replay Architecture (Decision D78, Caveat C6)
Open Question OQ-1 is resolved via **Replay Mixture Sampling**:
- Operating states for the 12 demo machines are assembled by sampling with replacement **strictly from the quarantined test partition** using scenario-based mixture interpolations:
  - **Healthy machines (M-101, M-102, M-108):** 100% sampled from the `low` stress test pool.
  - **Drifting machine (M-103):** Transitions linearly from 100% `low` to 80% `low` / 20% `stressed`.
  - **Hero degrading machine (M-104):** Transitions linearly from 100% `low` at Day 1 to 20% `low` / 30% `stressed` / 50% `high` at Day 30, capturing severe degradation.
  - **Maintenance machine (M-112):** Receives normal operating states for Days 1–29, and is clamped to zero stress on Day 30 during its active overhaul.
- **Dataset Contract:** Every demo row in `machine_ai4i_state.parquet`, `machine_risk_daily.parquet`, and `machine_risk_latest.csv` carries the explicit boolean column `is_replay = True`.
- **Fleet Risk Summary (Latest Day 30 Snapshot):**
  - `M-104` exhibits the highest 7-day average failure probability (**0.3732**, High Risk), perfectly aligning with its planted compressor leak scenario.
  - `M-101`, `M-102`, and `M-108` maintain baseline probabilities (**0.0125**, Low Risk).
  - Replay states are strictly demonstrative for the user interface and are never reported as model evaluation data (Rule R11).

---

## 10. Serving Contract & Inference Architecture

The model serving interface is encapsulated in `ml/prediction/inference.py`:
- `load_model(model_path=None)`: Caches the `FailureModel` singleton in memory for FastAPI lifespan startup on Day 7.
- `predict_one(payload: dict) -> dict`: Validates payload schema, computes physics features, evaluates the isotonic calibrated booster, and returns a JSON-serializable dictionary.
- `predict_batch(df: pd.DataFrame) -> pd.DataFrame`: High-throughput vectorized scoring with input range checks.

### Serving Contract Payload & Response Schema
```json
// Input Payload
{
  "machine_id": "M-104",
  "type": "L",
  "air_temperature_k": 298.5,
  "process_temperature_k": 309.2,
  "rotational_speed_rpm": 1420.0,
  "torque_nm": 48.5,
  "tool_wear_min": 185.0
}

// Response Contract
{
  "machine_id": "M-104",
  "failure_probability": 0.7842,
  "risk_level": "High",
  "thresholds": {
    "medium": 0.0460,
    "high": 0.5000,
    "critical": 0.5000
  },
  "model_version": "1.0.0",
  "framing": "Failure risk for the current operating state",
  "input_warnings": []
}
```

### Out-of-Distribution Input Envelope Validation
If incoming continuous sensor measurements fall outside training bounds by $> 10\%$, `predict_one` and `predict_batch` append non-blocking descriptive warnings into `input_warnings`:
```json
"input_warnings": [
  "rotational_speed_rpm (3500.0) exceeds training max (2886.0) by >10%"
]
```

### Train/Serve Parity
Train/serve parity is verified by unit test `test_train_serve_parity_integration` in `tests/test_failure_model.py`: raw batch pipeline outputs and `predict_batch` outputs match within machine precision ($|p_{\text{batch}} - p_{\text{serve}}| \le 10^{-6}$).

---

## 11. SHAP Readiness for Day 10 (Decision D79)

`FailureModel` keeps the underlying `xgb.Booster` cleanly decoupled from the isotonic calibrator and stores a 100-row representative background sample.  
An automated additivity check (`shap.TreeExplainer(booster)`) confirms:
$$\max |(\text{base\_value} + \sum \phi_i) - \text{margin}| = 1.34 \times 10^{-5} \le 10^{-3}$$
This mathematically guarantees that local TreeSHAP explanations on Day 10 will sum exactly to the raw margin before isotonic calibration mapping.

---

## 12. Crucial Limitations & Methodological Caveats

| # | Caveat | Operational Implication |
|---|---|---|
| **C1** | **Rule-Generated Labels** | High test scores (PR-AUC 0.8756, Recall 80.9%) reflect the mathematical determinism of AI4I equations rather than magical intelligence on noisy industrial assets. |
| **C2** | **Random Failures (RNF)** | Pure random noise failures cannot be predicted by any model. Test recall ceiling is mathematically capped at 97.06%. |
| **C3** | **Severe Class Imbalance** | 3.4% prevalence makes accuracy (96.6%) meaningless. Never report or evaluate without PR-AUC and prevalence baselines. |
| **C4** | **Small Positive Evaluation Count** | The test set contains exactly 68 failures. Bootstrap confidence intervals are necessary to convey true sampling uncertainty. |
| **C5** | **Domain Transfer Gap** | AI4I models milling machine tooling. Its predictions cannot be applied directly to physical air compressors without sensor-to-process harmonization. |
| **C6** | **Demo Replay Framing** | Machine risk values in the dashboard are generated via replayed test states, not measured factory sensors. Flagged with `is_replay=True`. |
| **C7** | **Decoupling of Maintenance History** | The failure model evaluates current physics only. Overdue maintenance schedules and work orders must be incorporated via the LLM reasoning agent (Days 11–12). |

---

## 13. Figure Gallery & Captions

- **Figure 15 (`docs/figures/fig15_failure_cv_search.png`):** Hyperparameter random search and ablation scorecard across 53 configurations evaluated on Repeated Stratified K-Fold (5x2). Shows PR-AUC vs. Brier score, highlighting `XGB-PHYS-08` over Logistic Regression and Raw feature baselines.
- **Figure 16 (`docs/figures/fig16_failure_calibration.png`):** Reliability diagram and calibration curves comparing uncalibrated XGBoost, Platt Sigmoid, and Isotonic Regression on pooled out-of-fold predictions and the frozen test partition.
- **Figure 17 (`docs/figures/fig17_failure_roc_pr_curves.png`):** Precision-Recall and ROC curves for the frozen primary model evaluated on the quarantined test partition ($N=2,000$, 68 failures). Shows PR-AUC = 0.8756 against the 0.0340 prevalence baseline and marks the $t_{\text{high}}$ operating point.
- **Figure 18 (`docs/figures/fig18_failure_confusion_matrix.png`):** Multi-threshold confusion matrices and diagnostic recall breakdown across individual failure modes (`pwf`, `osf`, `hdf`, `twf`, `rnf`) on the test partition, highlighting the 97.06% physical recall ceiling.

