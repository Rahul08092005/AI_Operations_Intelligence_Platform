# Feature Catalog

**Project:** AI Operations Intelligence Platform  
**Pipeline Phase:** Day 03 (Data Engineering & Feature Engineering)  
**Target Datasets:** `sensor_features.parquet` and `operations_dataset.parquet`  

## How to Read This Catalog

Every engineered feature strictly complies with temporal leakage prevention rules:
- **L1 (Trailing Only):** Features use strictly current and past data ($t' \le t$).
- **L2 (Strictly Past Events):** Maintenance history dated $D$ is visible from $D + 1$.
- **L3 (No Global Statistics):** All statistics are from local trailing windows.
- **L4 (Label Isolation):** Failure labels reside only in `anomaly_eval_labels.parquet`.
- **L5 (Forward Imputation):** Short gaps (<= 15m) filled and flagged via `is_imputed`.

## Feature Inventory

| Feature Name | Group | Definition | Unit | Window | Leakage Rule | Doc Name |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `timestamp` | keys | 5-min UTC sensor timestamp | ISO 8601 | None | L1 | `timestamp` |
| `machine_id` | keys | Machine identifier | category | None | L1 | `machine_id` |
| `temperature` | base_signals | Cleaned temperature | deg_C | current | L1 | `temperature` |
| `vibration` | base_signals | Cleaned RMS vibration | mm/s | current | L1 | `vibration` |
| `pressure` | base_signals | Cleaned discharge pressure | bar | current | L1 | `pressure` |
| `rpm` | base_signals | Cleaned shaft rotational speed | rpm | current | L1 | `rpm` |
| `energy_consumption` | base_signals | 5-min energy consumption | kWh | current | L1 | `energy_consumption` |
| `operating_hours` | base_signals | Monotonic operating hours | hours | current | L1 | `operating_hours` |
| `motor_current` | base_signals | Cleaned motor current | A | current | L1 | `motor_current` |
| `comp_state` | base_signals | Valve load state (0=load, 1=offload) | binary | current | L1 | `comp_state` |
| `running` | base_signals | Running flag (current > 1.55A) | boolean | current | L1 | `running` |
| `is_imputed` | quality_flags | Row contains imputed values | boolean | None | L5 | `is_imputed` |
| `is_gap` | quality_flags | Row created by 5-min grid alignment | boolean | None | L1 | `is_gap` |
| `is_sensor_fault` | quality_flags | Row had physical limit violation | boolean | None | L1 | `is_sensor_fault` |
| `is_outlier_flag` | quality_flags | Tier 2 outlier flag (|Z| > 6.0) | boolean | 24h trailing | L1 | `is_outlier_flag` |
| `outlier_signals` | quality_flags | Signals triggering outlier flag | string | 24h trailing | L1 | `outlier_signals` |
| `is_warmup` | quality_flags | First 24h of machine series | boolean | None | L1 | `is_warmup` |
| `temperature_roll_mean_1h` | rolling_stats | 1h trailing mean temperature | deg_C | 1h (12) | L1 | `rolling_temperature_mean_1h` |
| `temperature_roll_std_1h` | rolling_stats | 1h trailing std temperature | deg_C | 1h (12) | L1 | `rolling_temperature_std_1h` |
| `temperature_roll_mean_6h` | rolling_stats | 6h trailing mean temperature | deg_C | 6h (72) | L1 | `rolling_temperature_mean_6h` |
| `temperature_roll_std_6h` | rolling_stats | 6h trailing std temperature | deg_C | 6h (72) | L1 | `rolling_temperature_std_6h` |
| `temperature_roll_mean_24h` | rolling_stats | 24h trailing mean temperature | deg_C | 24h (288) | L1 | `rolling_temperature_mean_24h` |
| `temperature_roll_std_24h` | rolling_stats | 24h trailing std temperature | deg_C | 24h (288) | L1 | `rolling_temperature_std_24h` |
| `temperature_roll_max_1h` | rolling_stats | 1h trailing max temperature | deg_C | 1h (12) | L1 | `temperature_max_1h` |
| `temperature_roll_max_24h` | rolling_stats | 24h trailing max temperature | deg_C | 24h (288) | L1 | `temperature_max_24h` |
| `vibration_roll_mean_1h` | rolling_stats | 1h trailing mean vibration | mm/s | 1h (12) | L1 | `rolling_vibration_mean_1h` |
| `vibration_roll_std_1h` | rolling_stats | 1h trailing std vibration | mm/s | 1h (12) | L1 | `rolling_vibration_std_1h` |
| `vibration_roll_mean_6h` | rolling_stats | 6h trailing mean vibration | mm/s | 6h (72) | L1 | `rolling_vibration_mean_6h` |
| `vibration_roll_std_6h` | rolling_stats | 6h trailing std vibration | mm/s | 6h (72) | L1 | `rolling_vibration_std_6h` |
| `vibration_roll_mean_24h` | rolling_stats | 24h trailing mean vibration | mm/s | 24h (288) | L1 | `rolling_vibration_mean_24h` |
| `vibration_roll_std_24h` | rolling_stats | 24h trailing std vibration | mm/s | 24h (288) | L1 | `rolling_vibration_std_24h` |
| `vibration_roll_max_1h` | rolling_stats | 1h trailing max vibration | mm/s | 1h (12) | L1 | `vibration_max_1h` |
| `vibration_roll_max_24h` | rolling_stats | 24h trailing max vibration | mm/s | 24h (288) | L1 | `vibration_max_24h` |
| `temperature_change` | trends | Temperature delta vs 1h earlier | deg_C | 1h (12) | L1 | `temperature_change` |
| `vibration_change` | trends | Vibration delta vs 1h earlier | mm/s | 1h (12) | L1 | `vibration_change` |
| `pressure_change` | trends | Pressure delta vs 1h earlier | bar | 1h (12) | L1 | `pressure_change` |
| `energy_change` | trends | Energy delta vs 1h earlier | kWh | 1h (12) | L1 | `energy_change` |
| `temperature_change_24h` | trends | Temperature delta vs 24h earlier | deg_C | 24h (288) | L1 | `temperature_change_24h` |
| `vibration_change_24h` | trends | Vibration delta vs 24h earlier | mm/s | 24h (288) | L1 | `vibration_change_24h` |
| `pressure_change_24h` | trends | Pressure delta vs 24h earlier | bar | 24h (288) | L1 | `pressure_change_24h` |
| `energy_change_24h` | trends | Energy delta vs 24h earlier | kWh | 24h (288) | L1 | `energy_change_24h` |
| `vibration_slope_24h` | trends | LS vibration slope/h (24h) | mm/s per h | 24h (288) | L1 | `vibration_slope_24h` |
| `temperature_slope_24h` | trends | LS temperature slope/h (24h) | deg_C per h | 24h (288) | L1 | `temperature_slope_24h` |
| `running_ratio_1h` | context | Fraction 1h running | fraction | 1h (12) | L1 | `running_ratio_1h` |
| `running_ratio_24h` | context | Fraction 24h running | fraction | 24h (288) | L1 | `running_ratio_24h` |
| `load_ratio_1h` | context | Fraction 1h under load | fraction | 1h (12) | L1 | `load_ratio_1h` |
| `vibration_above_warning` | context | Vibration > 4.5 mm/s flag | boolean | current | L1 | `vibration_above_warning` |
| `vibration_above_critical` | context | Vibration > 7.0 mm/s flag | boolean | current | L1 | `vibration_above_critical` |
| `temperature_above_spec` | context | Temperature > 85 deg_C flag | boolean | current | L1 | `temperature_above_spec` |
| `vibration_above_warning_ratio_24h` | context | Trailing 24h share in warning | fraction | 24h (288) | L1 | `vibration_above_warning_ratio_24h` |
| `days_since_maintenance` | maintenance | Days since last maintenance | days | past date | L2 | `days_since_maintenance` |
| `days_since_inspection` | maintenance | Days since last inspection | days | past date | L2 | `days_since_inspection` |
| `days_since_preventive` | maintenance | Days since last PM | days | past date | L2 | `days_since_preventive` |
| `operating_hours_since_inspection` | maintenance | Operating hours since inspection | hours | past date | L2 | `operating_hours_since_inspection` |
| `operating_hours_since_preventive` | maintenance | Operating hours since PM | hours | past date | L2 | `operating_hours_since_preventive` |
| `hours_since_estimated` | maintenance | 1 if hours since insp/PM estimated | binary | past date | L2 | `hours_since_estimated` |
| `inspection_overdue` | maintenance | Operating hours since insp >= 500h | boolean | past date | L2 | `inspection_overdue` |
| `preventive_overdue` | maintenance | Operating hours since PM >= 2000h | boolean | past date | L2 | `preventive_overdue` |
| `maintenance_count` | maintenance | Count of past maintenance events | count | past date | L2 | `maintenance_count` |
| `maintenance_count_90d` | maintenance | Events count trailing 90d | count | 90d trailing | L2 | `maintenance_count_90d` |
| `maintenance_cost_avg` | maintenance | Expanding mean cost of past events | GBP | past date | L2 | `maintenance_cost_avg` |
| `maintenance_cost_90d` | maintenance | Sum of costs in trailing 90d | GBP | 90d trailing | L2 | `maintenance_cost_90d` |
| `failure_count` | maintenance | Count of past Corrective failures | count | past date | L2 | `failure_count` |
| `days_since_failure` | maintenance | Days since last failure | days | past date | L2 | `days_since_failure` |
| `previous_failure` | maintenance | 1 if failure in trailing 90d | binary | 90d trailing | L2 | `previous_failure` |
