# Anomaly Detection Holdout Evaluation Log

This document records immutable, single-run holdout evaluations under Decision D64.

## Holdout Evaluation Run — 2026-11-11 15:00:25 UTC

- **Execution Mode:** Frozen Pre-Registered Configuration (Decision D64)
- **Frozen Hash (Primary SHA-256):** `84def01407d0e4d750e5347388a6ea41f6acee72c42a89602a3ca83d7d378ce7`
- **Frozen Hash (Real-Only SHA-256):** `0b7d1ff566b15bc8dfd79138bd98bdde521e97a17b975dd24f2e8710e5359ef7`
- **Git Commit:** `4bebb66`
- **Holdout Period:** `2026-08-17 00:00:00+00:00` to `2026-10-20 00:00:00+00:00` (64.0 days)

### 1. Headline Results & False-Alarm Budget

| Detector | Feature Set | Budget | Warn FA/30d | Crit FA | M-109 Warn | M-109 Crit | FAIL-004 | Hero Deg |
|---|---|---|---|---|---|---|---|---|
| **Primary (IF-06)** | `plan_core` | **True** | 0.586 | 0 | 0.938 | 0 | late (0.0h) | Det (0.82) |
| **Real-Only (IF-25)** | `real_only` | **True** | 0.430 | 0 | 0.000 | 0 | missed (nanh) | N/A |
| **B1 Manual** | `vib_temp` | **True** | 0.000 | 0 | 0.000 | 0 | late | Det |
| **B2 Robust Z** | `plan_core` | **False** | 3.203 | 41 | 6.094 | 6 | late | Det |

### 2. Real Failure Window Detection (FAIL-004 on M-104)
- **Window:** 2026-08-30 14:30 UTC to 2026-09-02 19:00 UTC
- **Primary Model:** Status `late`, Lead Time `0.0h`, Peak `0.8074`.
- **Real-Only Model:** Status `missed`, Lead Time `nanh`.
- **Intellectual Honesty (Caveat C3):** Primary model detection is elevated by synthetic vibration
  (+60%) and temperature (+3 °C) overlays planted in FAIL-004. Real-only channels show honest
  empirical physics with no artificial precursor. Confidence: **low** (N=1 holdout failure window).

### 3. Holdout Synthetic Episode Detection

| Episode ID | Machine | Episode Type | Primary Det. | Peak Score | Real-Only Det. | Notes |
|---|---|---|---|---|---|---|
| `DRIFT-M103` | `M-103` | `mild_drift` | **True** | 0.6513 | False | Alert in 187.83h |
| `HERO-DEG-M104` | `M-104` | `hero_degradation` | **True** | 0.8161 | False | Alert in 449.0h |
| `SYN-FAIL-M111-PRE` | `M-111` | `synthetic_failure_precursor` | **False** | 0.2672 | False | Alert in nanh |

### 4. Interval-Level Precision & PR-AUC (Holdout Slice)
- **Primary Model (All True Anomalies):** PR-AUC = **0.6604**
  (Prevalence: 0.1054, 95% Bootstrap CI: [0.4913, 0.7888])
- **Primary Model (Real MetroPT Only):** PR-AUC = **0.2175**
  (Prevalence: 0.0131, 95% CI: [0.0060, 0.5287])
- **Real-Only Model (All True Anomalies):** PR-AUC = **0.1014**
  (Prevalence: 0.1051)

### 5. Demo-Story Audit at Final Timestamp (Rule R9 Check)

| Machine ID | Last Timestamp | State | Score | Actual Status | Expected Demo | Match |
|---|---|---|---|---|---|---|
| `M-101` | `2026-10-19 23:55:00+00:00` | `Offline` | N/A | **Offline** | Normal | **False** |
| `M-102` | `2026-10-19 23:55:00+00:00` | `Running` | N/A | **NoData** | Normal | **False** |
| `M-103` | `2026-10-19 23:55:00+00:00` | `Offline` | N/A | **Offline** | Warning | **False** |
| `M-104` | `2026-10-19 23:55:00+00:00` | `Offline` | N/A | **Offline** | Critical | **False** |
| `M-105` | `2026-10-19 23:55:00+00:00` | `Running` | 0.1237 | **Normal** | N/A | **True** |
| `M-106` | `2026-10-19 23:55:00+00:00` | `Running` | 0.1746 | **Normal** | N/A | **True** |
| `M-107` | `2026-10-19 23:55:00+00:00` | `Offline` | N/A | **Offline** | N/A | **True** |
| `M-108` | `2026-10-19 23:55:00+00:00` | `Running` | 0.2303 | **Normal** | N/A | **True** |
| `M-109` | `2026-10-19 23:55:00+00:00` | `Running` | N/A | **NoData** | N/A | **True** |
| `M-110` | `2026-10-19 23:55:00+00:00` | `Offline` | N/A | **Offline** | N/A | **True** |
| `M-111` | `2026-10-19 23:55:00+00:00` | `Running` | N/A | **NoData** | N/A | **True** |
| `M-112` | `2026-10-19 23:55:00+00:00` | `Offline` | N/A | **Offline** | N/A | **True** |

---

## Failure Model Test Partition Evaluation — 2026-11-15 15:37:42 UTC

- **Execution Mode:** Frozen Pre-Registered Configuration (Rule R2, Decision D68)
- **Frozen Hash (SHA-256):** `bfec1db63882ac7f4861c7569949d9f0ad6c934c0200ac8eac3753fb85840b2b`
- **Test Partition Size:** 2,000 rows (68 positive failures, prevalence: 0.0340)
- **Model Version:** `failure-xgb-1.0.0`
- **Primary Configuration ID:** `XGB-PHYS-08`
- **Operating Thresholds:** `t_med=0.0460`, `t_high=0.5000`, `t_crit=0.5000`

### 1. Headline Evaluation Metrics (Test Partition)

| Metric | Primary Model | 95% Bootstrap CI | Baseline (Majority) | Baseline (Logistic Reg) | Ablation (Raw Features) |
|---|---|---|---|---|---|
| **PR-AUC (Average Precision)** | **0.8756** | [0.8060, 0.9366] | 0.0340 | 0.4686 | 0.8269 |
| **ROC-AUC** | **0.9775** | [0.9551, 0.9956] | 0.5000 | 0.9258 | 0.9684 |
| **F1 Score (@ t_high)** | **0.8661** | [0.7969, 0.9242] | 0.0000 | 0.2697 | 0.7692 |
| **Recall (@ t_high, n=68)** | **0.8088** | [0.7059, 0.8971] | 0.0000 | 0.1765 | 0.6618 |
| **Precision (@ t_high)** | **0.9322** | [0.8615, 0.9833] | 0.0000 | 0.5714 | 0.9184 |
| **Accuracy (@ t_high)** | **0.9915** | N/A | 0.9660 | 0.9675 | 0.9865 |
| **Brier Score** | **0.0076** | [0.0049, 0.0107] | 0.0328 | 0.0241 | 0.0125 |
| **Expected Calibration Error (ECE)** | **0.0020** | N/A | N/A | 0.0084 | 0.0078 |

*Note: All recall and precision figures are computed with exactly 68 positive failures in the test partition.*

### 2. Operating Points Performance on Test Partition

| Operating Point | Threshold | Recall (n=68) | Precision | False Positives | True Positives | True Negatives | False Negatives |
|---|---|---|---|---|---|---|---|
| **t_med** (Target Recall >= 0.95) | 0.0460 | 94.1% | 31.7% | 138 | 64 | 1794 | 4 |
| **t_high** (Max F1 Score) | 0.5000 | 80.9% | 93.2% | 4 | 55 | 1928 | 13 |
| **t_crit** (Target Precision >= 0.90) | 0.5000 | 80.9% | 93.2% | 4 | 55 | 1928 | 13 |
| **Default (0.50)** | 0.5000 | 80.9% | 93.2% | 4 | 55 | 1928 | 13 |

### 3. Test Category Reliability & Volume Share

| Category | Range | Rows | Share (%) | Failures | Empirical Failure Rate |
|---|---|---|---|---|---|
| **Low** | `[0.0000, 0.0460)` | 1,798 | 89.90% | 4 | 0.0022 |
| **Medium** | `[0.0460, 0.5000)` | 143 | 7.15% | 9 | 0.0629 |
| **High** | `[0.5000, 0.5000)` | 0 | 0.00% | 0 | 0.0000 |
| **Critical** | `[0.5000, 1.0000]` | 59 | 2.95% | 55 | 0.9322 |

### 4. Failure Mode Breakdown & Recall Ceiling

| Failure Mode | Test Count | Detected (@ t_high) | Recall | Classification Nature |
|---|---|---|---|---|
| **twf** (Tool Wear Failure) | 10 | 1 | 10.0% | Physical degradation |
| **hdf** (Heat Dissipation Failure) | 29 | 27 | 93.1% | Thermodynamic limit |
| **pwf** (Power Failure) | 13 | 13 | 100.0% | Power envelope violation |
| **osf** (Overstrain Failure) | 16 | 16 | 100.0% | Mechanical overstrain |
| **rnf** (Random Failures) | 4 | 0 | 0.0% | Uniform random noise (Unpredictable) |

- **Empirical Test Recall Ceiling (Caveat C2):** **97.06%** (66/68 failures contain at least one physical non-random failure mode).
- **Caveat C1 Intellectual Honesty Statement:** The near-perfect scores achieved on physical failure modes (`hdf`, `pwf`, `osf`) directly reflect the rule-generated nature of the synthetic AI4I 2020 benchmark dataset. The model learns mathematical representations of these physical envelopes. On truly random failures (`rnf`), detection recall is near 0%, as expected.
