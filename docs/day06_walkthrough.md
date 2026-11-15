# Day 6 Walkthrough: Failure Prediction Model (XGBoost)

**Project:** AI Operations Intelligence Platform  
**Branch:** `feature/day-06-failure-model`  
**Date:** November 2026  
**Audience:** Platform Engineering, Lead Data Scientist, Technical Evaluator  

---

## 1. EDA Handoff & Stop Condition Verifications

### 1.1 Decision Register Handoff (from `docs/eda_findings.md`, EQ40 to EQ47)
- **EQ40 (Target Prevalence & Imbalance Baseline):** In the 8,000-row training partition, failure prevalence is exactly 3.3875% (~3.39%; 271 failures vs. 7,729 healthy). Majority accuracy is 96.61%; the baseline PR-AUC (average precision) is exactly **0.0339**.
- **EQ41 (Failure Modes & Co-occurrence):** Heat dissipation failure (`hdf`, 86 rows) and power failure (`pwf`, 82 rows) are the most prevalent modes in training, followed by overstrain (`osf`, 78), tool wear (`twf`, 46), and random failures (`rnf`, 19). Exactly 22 training failure rows exhibit multiple simultaneous failure modes, and 7 failure rows have no mode flag.
- **EQ42 (Standardized Effect Sizes):** Torque ($d = +1.48$ [1.32, 1.64]) and tool wear ($d = +0.89$ [0.76, 1.02]) show strong positive shifts on failure; rotational speed exhibits a large negative shift ($d = -1.05$).
- **EQ43 (Failure Concentration):** Failure prevalence varies inversely with machine quality variant: 'L' (Low quality) has 3.92% failure rate vs. 2.14% on 'H' (High quality).
- **EQ44 (Physics-Derived Features):** Continuous physics features—specifically mechanical power ($P = \tau \cdot \omega$) and cumulative strain proxy ($\text{tool wear} \times \text{torque}$)—yield strong univariate separation ($ROC\text{-}AUC = 0.745$ and $0.692$ respectively).
- **EQ45 (Rule-Likeness Diagnostic):** A shallow depth-2 decision tree on `strain` and `power_w` alone isolates 85.2% precision, confirming that AI4I labels are largely deterministic functions of physical limits. Models that capture these thresholds will achieve near-perfect metrics; this demonstrates rule recovery, not mystical predictive power (Caveat C1).
- **EQ46 (Target Leakage Exclusion):** Mode indicator flags (`twf`, `hdf`, `pwf`, `osf`, `rnf`) are logically causal sub-components of `machine_failure` and must be strictly quarantined. Identifiers `udi` and `product_id` must be excluded.
- **EQ47 (Honest Reporting Standards):** Day 6 models must report PR-AUC against the 0.0339 baseline, Recall @ 80% Precision, and Brier calibration scores. Accuracy must never be reported without PR-AUC and recall.

### 1.2 Stop Conditions Verification
- `data/processed/ai4i_split_index.csv` verified: exactly 8,000 train rows (271 positives, 3.3875%) and 2,000 test rows (68 positives, 3.4000%). Stratification holds exactly.
- Environment check: Python 3.14, XGBoost 3.4.1, SHAP 0.52.0. Smoke test for XGBoost + SHAP TreeExplainer passed.
- No test partition data accessed during setup.

---

## 2. Architecture & Decisions (D67 to D82)
- **D67 (Problem Framing):** Output is **failure risk for the current operating state** (probability that an operating state of this kind corresponds to a failure in AI4I). No time horizon.
- **D68 (Data and Split):** Stratified 80/20 split (`ai4i_split_index.csv`). Train partition for cross-validation and calibration; test partition touched strictly once after freeze.
- **D69 (Features):** Raw features plus continuous physics quantities: temperature difference ($\Delta T$), mechanical power ($P$), and strain proxy ($\tau \times \text{wear}$). Ordinal encoding for `type` (L=0, M=1, H=2).
- **D70 (Excluded Columns):** Strict exclusion of `twf`, `hdf`, `pwf`, `osf`, `rnf`, `udi`, and `product_id`.
- **D71 (Cross-Validation):** Repeated Stratified K-Fold (5 folds $\times$ 2 repeats) on the training partition. Metric: mean out-of-fold PR-AUC.
- **D72 (Search Space):** Random search over 50 XGBoost configurations from pre-registered parameter distributions; 10 configurations reserved for ablations and refits (cap 60).
- **D73 (Class Imbalance):** `scale_pos_weight` evaluated with 1, $\sqrt{N/P}$, and $N/P$.
- **D74 (Calibration):** External probability calibration on out-of-fold predictions. Sigmoid (Platt) by default; isotonic only if Brier improves $\ge 5\%$ without worsening ECE.
- **D75 (Risk Categories):** Categorical bands `Low`, `Medium`, `High`, `Critical` derived from OOF operating points (`t_med` at recall $\ge 0.95$, `t_high` at max F1, `t_crit` at precision $\ge 0.90$).
- **D76 (Baselines and Ablations):** Benchmarked against majority class, logistic regression, and XGBoost on raw features only (F0).
- **D77 (Test Protocol):** Freeze configuration with SHA-256 hashes (`models/failure_config_frozen.json`), single test evaluation logged to `docs/holdout_log.md`.
- **D78 (Resolution of OQ-1: Demo Machine Inputs):** Replay approach: each demo machine receives daily AI4I-style operating states sampled from **test-partition** rows according to scenario mixtures. Sensor adapter mapping is rejected.
- **D79 (Model Structure for Explainability):** `FailureModel` bundle encapsulates uncalibrated booster (log-odds output for TreeExplainer), external calibrator, thresholds, feature metadata, and a 200-row training background sample.
- **D80 (Serving Contract):** Standardized input/output contract for `POST /predict/failure` on Day 7.
- **D81 (Artifacts):** `models/failure_model.pkl` (joblib bundle) and `models/failure_model_metadata.json`.
- **D82 (Maintenance Context):** Maintenance metrics are combined in reasoning layers (Days 10–12), not as direct model inputs.

---

*(Sections 3–10 will be populated as tasks T2–T13 are executed)*
