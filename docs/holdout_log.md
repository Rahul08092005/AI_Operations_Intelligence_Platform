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
