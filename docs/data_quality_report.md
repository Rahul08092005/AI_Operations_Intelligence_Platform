# Data Quality & Pipeline Audit Report

**Generated:** 2026-10-21T15:07:56.626809+00:00  
**Pipeline Mode:** FULL FLEET (12 machines, 214 days)  
**Pipeline Status:** PASSED (Zero Errors)  

---

## 1. Table Row Counts (Raw vs. Processed)

| Table Name | Raw Rows | Final Rows | Change / Reason |
| :--- | :--- | :--- | :--- |
| `anomaly_eval_labels` | 0 | 736,128 | +736,128 (explicit gaps / reindexing) |
| `business_daily` | 213 | 213 | 0 (preserved) |
| `business_features_daily` | 0 | 374 | +374 (explicit gaps / reindexing) |
| `business_transactions` | 536,636 | 536,636 | 0 (preserved) |
| `business_transactions_rejected` | 0 | 0 | 0 (preserved) |
| `machines` | 12 | 12 | 0 (preserved) |
| `maintenance` | 43 | 43 | 0 (preserved) |
| `operations_dataset` | 0 | 2,556 | +2,556 (explicit gaps / reindexing) |
| `product_demand_daily` | 0 | 18,700 | +18,700 (explicit gaps / reindexing) |
| `production` | 2,556 | 2,556 | 0 (preserved) |
| `sensor_features` | 0 | 736,128 | +736,128 (explicit gaps / reindexing) |
| `sensors` | 736,128 | 736,128 | 0 (preserved) |

---

## 2. Sensor Cleaning & Missing Value Imputation

- **Raw Rows Ingested:** 736,128
- **Exact Duplicates Dropped:** 0
- **Conflicting Duplicates Resolved:** 0
- **Explicit Gap Rows Created:** 0
- **Tier 1 Hard Range Violations (Set to NaN):** 0
- **Short Gaps Imputed (<= 15 min):** 1 rows
- **Warmup Rows Flagged (First 24h):** 3,456

### Longest Gap per Machine (in 5-min intervals)

| Machine ID | Longest Gap Intervals | Equivalent Hours |
| :--- | :--- | :--- |
| `M-101` | 0 | 0.0 h |
| `M-102` | 0 | 0.0 h |
| `M-103` | 0 | 0.0 h |
| `M-104` | 0 | 0.0 h |
| `M-105` | 0 | 0.0 h |
| `M-106` | 0 | 0.0 h |
| `M-107` | 0 | 0.0 h |
| `M-108` | 0 | 0.0 h |
| `M-109` | 0 | 0.0 h |
| `M-110` | 0 | 0.0 h |
| `M-111` | 0 | 0.0 h |
| `M-112` | 0 | 0.0 h |

---

## 3. Tier 2 Outlier Flagging & Anomaly Preservation

- **Total Rows Flagged by Tier 2:** 240,418
- **Preservation Suite Status:** PASSED
- **Real Failure Window Faults:** True (0 destroyed)
- **Peak Temperature Preservation:** True
- **Peak Vibration Preservation:** True
- **Benign Spikes Preserved:** True

### Tier 2 Outlier Flag Rates per Machine

| Machine ID | Flag Rate (%) |
| :--- | :--- |
| `M-101` | 31.68% |
| `M-102` | 31.98% |
| `M-103` | 31.98% |
| `M-104` | 33.20% |
| `M-105` | 32.12% |
| `M-106` | 31.41% |
| `M-107` | 32.93% |
| `M-108` | 32.92% |
| `M-109` | 33.84% |
| `M-110` | 32.04% |
| `M-111` | 34.88% |
| `M-112` | 32.95% |

---

## 4. Retail Transactions Cleaning & Rejections

- **Raw Transactions Ingested:** 536,636
- **Clean Kept Transactions:** 536,636
- **Rejected Transactions:** 0
- **Cancellations / Returns:** 8,706 / 8,706
- **Guest Customers Flagged:** 131,626
- **Non-Product Lines Flagged:** 207
- **Extreme Quantities Flagged (>= 10,000):** 4

### Rejection Reasons Breakdown

| Reject Reason | Count |
| :--- | :--- |

---

## 5. Validation Suite Comparison (Before vs. After)

| Metric | Before Pipeline | After Pipeline |
| :--- | :--- | :--- |
| **Overall Status** | PASSED | PASSED |
| **Total Checks** | 25 | 25 |
| **Checks Passed** | 25 | 25 |
| **Errors Remaining** | 0 | **0** |
| **Warnings (Audits)** | 0 | 0 |

---

## 6. Feature NaN Rates (Sensor Features & Operations Dataset)

### `sensor_features.parquet` (Top non-zero NaN rates)

| Feature Name | NaN Rate (%) | Note / Reason |
| :--- | :--- | :--- |
| `days_since_failure` | 90.10% | Expected window warmup / missing signal |
| `days_since_inspection` | 20.97% | Expected window warmup / missing signal |
| `operating_hours_since_inspection` | 20.97% | Expected window warmup / missing signal |
| `days_since_maintenance` | 20.27% | Expected window warmup / missing signal |
| `temperature_change_24h` | 2.43% | Expected window warmup / missing signal |
| `pressure_change_24h` | 2.43% | Expected window warmup / missing signal |
| `temperature_slope_24h` | 2.42% | Expected window warmup / missing signal |
| `energy_change_24h` | 2.35% | Expected window warmup / missing signal |
| `temperature_roll_mean_24h` | 1.25% | Expected window warmup / missing signal |
| `temperature_roll_std_24h` | 1.25% | Expected window warmup / missing signal |
| `pressure_roll_mean_24h` | 1.25% | Expected window warmup / missing signal |
| `pressure_roll_std_24h` | 1.25% | Expected window warmup / missing signal |
| `temperature_roll_max_24h` | 1.25% | Expected window warmup / missing signal |
| `energy_consumption_roll_mean_24h` | 1.21% | Expected window warmup / missing signal |
| `energy_consumption_roll_std_24h` | 1.21% | Expected window warmup / missing signal |

### `operations_dataset.parquet` (Daily Machine Level)

| Column Name | NaN Rate (%) | Note / Reason |
| :--- | :--- | :--- |
| `days_since_failure` | 90.10% | Expected lag warmup |
| `days_since_inspection` | 20.97% | Expected lag warmup |
| `operating_hours_since_inspection` | 20.97% | Expected lag warmup |
| `days_since_maintenance` | 20.27% | Expected lag warmup |
| `production_lag_7` | 3.29% | Expected lag warmup |
| `energy_per_unit` | 1.84% | Expected lag warmup |
| `temperature_mean` | 1.02% | Expected lag warmup |
| `temperature_max` | 1.02% | Expected lag warmup |
| `temperature_std` | 1.02% | Expected lag warmup |
| `pressure_mean` | 1.02% | Expected lag warmup |
| `pressure_max` | 1.02% | Expected lag warmup |
| `pressure_std` | 1.02% | Expected lag warmup |
| `temperature_roll_mean_24h_end` | 1.02% | Expected lag warmup |
| `motor_current_mean` | 0.98% | Expected lag warmup |
| `production_lag_1` | 0.47% | Expected lag warmup |

---

## 7. Temporal Leakage Audit & Verification

| Leakage Check | Policy | Result |
| :--- | :--- | :--- |
| `forbidden_metadata_in_operations` | L4 (Metadata Isolation) | **PASSED (No Leakage)** |
| `labels_in_operations_dataset` | L4 (Label Isolation) | **PASSED (No Leakage)** |
| `labels_in_sensor_features` | L4 (Label Isolation) | **PASSED (No Leakage)** |
| `maintenance_strictly_past` | L2 (Strictly Past Maintenance) | **PASSED (No Leakage)** |
| `operations_dataset_unique_keys` | Grain Integrity (Machine x Day) | **PASSED (No Leakage)** |

---

## 8. Database Loading Verification

*Database load was skipped (`--skip-db` or PostgreSQL not reachable).*
