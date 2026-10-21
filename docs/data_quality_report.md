# Data Quality & Pipeline Audit Report

**Generated:** 2026-10-21T14:58:33.662280+00:00  
**Pipeline Mode:** SAMPLE (2 machines, 45 days)  
**Pipeline Status:** PASSED (Zero Errors)  

---

## 1. Table Row Counts (Raw vs. Processed)

| Table Name | Raw Rows | Final Rows | Change / Reason |
| :--- | :--- | :--- | :--- |
| `anomaly_eval_labels` | 0 | 25,920 | +25,920 (explicit gaps / reindexing) |
| `business_daily` | 45 | 45 | 0 (preserved) |
| `business_features_daily` | 0 | 48 | +48 (explicit gaps / reindexing) |
| `business_transactions` | 128,933 | 128,933 | 0 (preserved) |
| `business_transactions_rejected` | 0 | 0 | 0 (preserved) |
| `machines` | 2 | 2 | 0 (preserved) |
| `maintenance` | 7 | 7 | 0 (preserved) |
| `operations_dataset` | 0 | 90 | +90 (explicit gaps / reindexing) |
| `product_demand_daily` | 0 | 2,400 | +2,400 (explicit gaps / reindexing) |
| `production` | 90 | 90 | 0 (preserved) |
| `sensor_features` | 0 | 25,920 | +25,920 (explicit gaps / reindexing) |
| `sensors` | 25,920 | 25,920 | 0 (preserved) |

---

## 2. Sensor Cleaning & Missing Value Imputation

- **Raw Rows Ingested:** 25,920
- **Exact Duplicates Dropped:** 0
- **Conflicting Duplicates Resolved:** 0
- **Explicit Gap Rows Created:** 0
- **Tier 1 Hard Range Violations (Set to NaN):** 0
- **Short Gaps Imputed (<= 15 min):** 0 rows
- **Warmup Rows Flagged (First 24h):** 576

### Longest Gap per Machine (in 5-min intervals)

| Machine ID | Longest Gap Intervals | Equivalent Hours |
| :--- | :--- | :--- |
| `M-103` | 0 | 0.0 h |
| `M-104` | 0 | 0.0 h |

---

## 3. Tier 2 Outlier Flagging & Anomaly Preservation

- **Total Rows Flagged by Tier 2:** 8,476
- **Preservation Suite Status:** PASSED
- **Real Failure Window Faults:** True (0 destroyed)
- **Peak Temperature Preservation:** True
- **Peak Vibration Preservation:** True
- **Benign Spikes Preserved:** True

### Tier 2 Outlier Flag Rates per Machine

| Machine ID | Flag Rate (%) |
| :--- | :--- |
| `M-103` | 32.32% |
| `M-104` | 33.08% |

---

## 4. Retail Transactions Cleaning & Rejections

- **Raw Transactions Ingested:** 128,933
- **Clean Kept Transactions:** 128,933
- **Rejected Transactions:** 0
- **Cancellations / Returns:** 1,572 / 1,572
- **Guest Customers Flagged:** 30,072
- **Non-Product Lines Flagged:** 35
- **Extreme Quantities Flagged (>= 10,000):** 2

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
| `days_since_failure` | 100.00% | Expected window warmup / missing signal |
| `temperature_change_24h` | 4.44% | Expected window warmup / missing signal |
| `pressure_change_24h` | 4.44% | Expected window warmup / missing signal |
| `energy_change_24h` | 4.44% | Expected window warmup / missing signal |
| `temperature_slope_24h` | 4.43% | Expected window warmup / missing signal |
| `vibration_change_24h` | 2.22% | Expected window warmup / missing signal |
| `temperature_roll_mean_24h` | 2.21% | Expected window warmup / missing signal |
| `temperature_roll_std_24h` | 2.21% | Expected window warmup / missing signal |
| `pressure_roll_mean_24h` | 2.21% | Expected window warmup / missing signal |
| `pressure_roll_std_24h` | 2.21% | Expected window warmup / missing signal |
| `energy_consumption_roll_mean_24h` | 2.21% | Expected window warmup / missing signal |
| `energy_consumption_roll_std_24h` | 2.21% | Expected window warmup / missing signal |
| `motor_current_roll_mean_24h` | 2.21% | Expected window warmup / missing signal |
| `motor_current_roll_std_24h` | 2.21% | Expected window warmup / missing signal |
| `temperature_roll_max_24h` | 2.21% | Expected window warmup / missing signal |

### `operations_dataset.parquet` (Daily Machine Level)

| Column Name | NaN Rate (%) | Note / Reason |
| :--- | :--- | :--- |
| `days_since_failure` | 100.00% | Expected lag warmup |
| `production_lag_7` | 15.56% | Expected lag warmup |
| `orders_lag_1` | 4.44% | Expected lag warmup |
| `retail_daily_sales` | 2.22% | Expected lag warmup |
| `retail_revenue` | 2.22% | Expected lag warmup |
| `retail_daily_orders` | 2.22% | Expected lag warmup |
| `retail_average_order_value` | 2.22% | Expected lag warmup |
| `retail_weekly_sales` | 2.22% | Expected lag warmup |
| `retail_monthly_sales` | 2.22% | Expected lag warmup |
| `retail_sales_growth` | 2.22% | Expected lag warmup |
| `production_lag_1` | 2.22% | Expected lag warmup |
| `production_roll_mean_7d` | 2.22% | Expected lag warmup |
| `downtime_roll_mean_7d` | 2.22% | Expected lag warmup |
| `temperature_mean` | 1.11% | Expected lag warmup |
| `temperature_max` | 1.11% | Expected lag warmup |

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
