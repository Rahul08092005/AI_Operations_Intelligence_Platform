# Model Card: Fleet Equipment Anomaly Detector (`IF-06`)

## 1. Model Details
- **Model Name:** Isolation Forest Fleet Anomaly Detector (`IF-06`)
- **Model Version:** 1.0.0 (Day 5 Frozen Release)
- **Model Type:** Unsupervised Ensemble Tree Outlier Detector with Per-Machine Robust Normalization & Piecewise-Linear Score Calibration
- **Architecture:** `sklearn.ensemble.IsolationForest`
  - Number of estimators: 400
  - Subsample size (`max_samples`): 4,096
  - Feature subsample (`max_features`): 1.0 (all 5 core features)
  - Post-processing smoothing: Trailing rolling mean (window 36 = 3 hours)
  - Calibration anchors: $q_{\text{warn}} = 0.99$, $q_{\text{crit}} = 0.9999$
- **Primary Feature Set:** `plan_core` (`temperature`, `vibration`, `pressure`, `rpm`, `energy_consumption`)
- **Comparator Model:** `IF-25` (`real_only`, 11 physical sensor channels)
- **Artifact SHA-256:** `84def01407d0e4d750e5347388a6ea41f6acee72c42a89602a3ca83d7d378ce7`
- **Release Date:** November 2026
- **License:** Proprietary / Internal Enterprise Use

---

## 2. Intended Use & Application Scope
- **Intended Tasks:**
  - Automated continuous monitoring of high-frequency (5-minute) industrial equipment telemetry.
  - Generating early advisory warning alerts (72-hour horizon) before mechanical breakdown.
  - Ranking anomalous intervals by severity (`Warning` at $\ge 0.50$, `Critical` at $\ge 0.80$).
  - Attributing primary physical root-cause deviations via per-machine robust $z$-scores.
- **Intended Users:** Reliability engineers, maintenance dispatch supervisors, and plant operators.
- **Target Assets:** 12 industrial manufacturing units (compressors, packaging lines, milling centers, hydraulic presses).

### Out-of-Scope & Prohibited Applications
- **Safety-Critical Automated Shutdown:** This model must **never** be connected directly to emergency stops or automated trip circuits.
- **Unsupervised Closed-Loop Action:** Alerts must be triaged by a certified technician; models do not replace physical inspection.
- **Offline Machine Evaluation:** Scoring machinery during unpowered or maintenance intervals is invalid.

---

## 3. Training Data & Data Preprocessing
- **Training Population:** 267,897 healthy operating intervals across all 12 machines from the chronological Exploration Partition (`2026-03-21 00:00 UTC` to `2026-08-17 00:00 UTC`).
- **State Gating (D54):** Telemetry is filtered to active operational periods (`running == True`). Offline and unmonitored intervals receive status `Offline` / `NoData`.
- **Label-Based Contamination Exclusions (D56):** 100% of intervals falling within known failure episodes, precursor degradation, and maintenance downtime plus a $\pm 24$-hour buffer were excluded from training. Nominal running and benign vibration spikes (`M-109`) were preserved.
- **Per-Machine Normalization (D53):** All features are preconditioned with robust median and MAD scaling derived strictly from each asset's training rows.

---

## 4. Quantitative Evaluation Summary

### Cross-Validation Performance (3-Fold Expanding Window)
- **Validation Failure Recall:** 75.0% (3 out of 4 failure windows detected)
- **False Alarm Rate:** 0.88 Warning FA / 30 days (Passing budget $\le 1.0$)
- **Critical False Alarms:** 0 (Passing budget: 0)

### Frozen Single-Run Holdout Performance (64 Days Out-of-Time)
- **M-104 Authentic Failure Detection:** 1 / 1 detected (Early warning with 71.9h lead time)
- **Synthetic Fault Episode Recall:** 80.0% (4 / 5 detected)
- **Holdout False Alarm Rate:** 0.586 Warning FA / machine / 30 days
- **Critical False Alarms:** 0
- **Holdout Average Precision:** 0.655 [95% CI: 0.621, 0.688]

### Real-Only Comparator Benchmark (`IF-25`)
- When evaluated strictly on native physical channels without synthetic overlays, detection recall drops to 0.0% (0 / 1 on M-104). This confirms that high detection sensitivity on `M-104` is driven by synthetic overlays (Caveat C3).

---

## 5. Known Failure Modes & Operational Limitations
1. **Low Sensitivity on Native Unmodified Telemetry:** As proven by `IF-25`, authentic MetroPT failure signatures are subtle in native pressure/current channels.
2. **False Alarms from Severe Load Variations:** Rapid multi-stage pneumatic load changes can temporarily mimic anomalies. The 3-hour trailing smoothing filter suppresses these transients.
3. **Repeated Fleet Patterns:** Synthetic equipment overlays share underlying generation distributions; fleet statistics should not be assumed to represent 12 fully independent industrial sites.

---

## 6. Retraining & Maintenance Protocol
To retrain and re-freeze model artifacts following new telemetry ingestion:
```bash
# 1. Run 3-fold cross-validation grid search
python scripts/run_anomaly_cv.py

# 2. Train final models on exploration partition
python scripts/train_anomaly.py

# 3. Cryptographically freeze configuration
python scripts/freeze_anomaly_config.py

# 4. Run tests
pytest tests/test_anomaly.py -v
```
