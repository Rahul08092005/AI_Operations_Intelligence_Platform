# Data Dictionary: Source Datasets and Unified Internal Schema

**Project:** AI Operations Intelligence Platform  
**Generated Date:** 2026-10-20  
**Author:** AI Operations Intelligence Platform (Automated Profiler)

---

## 1. MetroPT-3 (Air Compressor Dataset)

- **Source File:** `data/raw/metropt/MetroPT3(AirCompressor).csv`
- **Documentation:** `data/raw/metropt/Data Description_Metro.pdf`
- **Total Rows:** 1,516,948
- **Time Span:** 2020-02-01 00:00:00 to 2020-09-01 03:59:50
- **Modal Sampling Interval:** ~10 seconds (0.1 Hz)
- **Data Gaps (> 5 min):** 275 gaps (Maximum gap: 172,918.0 seconds)
- **Motor Current Threshold (`I_OFF`):** **1.55 A** (separating inactive/off states from operational/running states)
- **Operating Fraction:** 45.3% running time
- **`COMP` Signal Definition (from documentation):** Active (1.0) when air intake valve is closed (compressor off or offloaded/unloaded). Inactive (0.0) when compressor is loading and compressing air into reservoirs.

### MetroPT Columns & Statistics

| Column | Type | Unit | Min | Max | Mean | Missing | Description / Notes |
|---|---|---|---|---|---|---|---|
| `timestamp` | datetime | ISO 8601 | 2020-02-01 00:00:00 | 2020-09-01 03:59:50 | - | 0 | Recorded sensor timestamp (10s intervals) |
| `TP2` | float32 | bar | -0.032 | 10.676 | 1.3678 | 0 | Pressure on compressor (bar) |
| `TP3` | float32 | bar | 0.73 | 10.302 | 8.9846 | 0 | Pressure generated at pneumatic panel (bar) |
| `H1` | float32 | bar | -0.036 | 10.288 | 7.5682 | 0 | Cyclonic separator pressure drop (bar) |
| `DV_pressure` | float32 | bar | -0.032 | 9.844 | 0.056 | 0 | Air dryers discharge pressure (bar, 0 = under load) |
| `Reservoirs` | float32 | bar | 0.712 | 10.3 | 8.9852 | 0 | Reservoir downstream pressure (bar) |
| `Oil_temperature` | float32 | °C | 15.4 | 89.05 | 62.6442 | 0 | Compressor oil temperature (°C) |
| `Motor_current` | float32 | A | 0.02 | 9.295 | 2.0502 | 0 | Phase current (A; ~0A off, ~4A offload, ~7A load) |
| `COMP` | float32 | 0/1 | 0.0 | 1.0 | 0.837 | 0 | Air intake valve (1 = off/offload, 0 = load) |
| `DV_eletric` | float32 | 0/1 | 0.0 | 1.0 | 0.1606 | 0 | Electrical outlet valve signal (0/1) |
| `Towers` | float32 | 0/1 | 0.0 | 1.0 | 0.9198 | 0 | Dryer tower in operation (0 = T1, 1 = T2) |
| `MPG` | float32 | 0/1 | 0.0 | 1.0 | 0.8327 | 0 | Intake valve start signal (< 8.2 bar) |
| `LPS` | float32 | 0/1 | 0.0 | 1.0 | 0.0034 | 0 | Low pressure signal (< 7.0 bar) |
| `Pressure_switch` | float32 | 0/1 | 0.0 | 1.0 | 0.9914 | 0 | Air drying tower discharge detection (0/1) |
| `Oil_level` | float32 | 0/1 | 0.0 | 1.0 | 0.9042 | 0 | Low oil level indicator (0/1) |
| `Caudal_impulses` | float32 | count | 0.0 | 1.0 | 0.9371 | 0 | Air flow impulse counter (count) |

### Known Quirks
- The raw CSV starts with an unnamed index column (`Unnamed: 0`).
- Several sampling intervals deviate from 10s (ranging from 1s to ~20s, plus a few offline maintenance gaps).
- `COMP` is inverse logic to intuition: 1 denotes no air intake (offloaded/idle), 0 denotes active air intake (under load).

---

## 2. AI4I 2020 Predictive Maintenance Dataset

- **Source File:** `data/raw/ai4i/ai4i2020.csv`
- **Total Rows:** 10,000
- **Total Machine Failures:** 339 (3.39%)
- **Multi-failure Rows:** 24 rows with >1 failure flag
- **Variant Distribution:** L: 6,000, M: 2,997, H: 1,003

### Failure Mode Breakdown
- **Tool Wear Failure (`TWF`):** 46
- **Heat Dissipation Failure (`HDF`):** 115
- **Power Failure (`PWF`):** 95
- **Overstrain Failure (`OSF`):** 98
- **Random Failure (`RNF`):** 19

### Column Summary
| Column | Type | Min | Max | Mean | Missing | Notes |
|---|---|---|---|---|---|---|
| `UDI` | int64 | 1.0 | 10000.0 | 5000.5 | 0 | |
| `Product ID` | str | - | - | - | 0 | |
| `Type` | str | - | - | - | 0 | |
| `Air temperature [K]` | float64 | 295.3 | 304.5 | 300.0 | 0 | |
| `Process temperature [K]` | float64 | 305.7 | 313.8 | 310.01 | 0 | |
| `Rotational speed [rpm]` | int64 | 1168.0 | 2886.0 | 1538.78 | 0 | |
| `Torque [Nm]` | float64 | 3.8 | 76.6 | 39.99 | 0 | |
| `Tool wear [min]` | int64 | 0.0 | 253.0 | 107.95 | 0 | |
| `Machine failure` | int64 | 0.0 | 1.0 | 0.03 | 0 | |
| `TWF` | int64 | 0.0 | 1.0 | 0.0 | 0 | |
| `HDF` | int64 | 0.0 | 1.0 | 0.01 | 0 | |
| `PWF` | int64 | 0.0 | 1.0 | 0.01 | 0 | |
| `OSF` | int64 | 0.0 | 1.0 | 0.01 | 0 | |
| `RNF` | int64 | 0.0 | 1.0 | 0.0 | 0 | |
| `failure_mode_sum` | int64 | 0.0 | 3.0 | 0.04 | 0 | |

### Known Quirks & Leakage Traps
- `Machine failure` is a deterministic combination of failure modes plus random failures. Modes `TWF`, `HDF`, `PWF`, `OSF`, `RNF` **must not** be used as predictive features.
- AI4I has no temporal dimension or machine persistence; each row is an independent synthetic machine state.

---

## 3. UCI Online Retail Dataset

- **Source File:** `data/raw/retail/Online Retail.xlsx` (cached as `online_retail.parquet`)
- **Total Rows:** 541,909
- **Date Range:** 2010-12-01 08:26:00 to 2011-12-09 12:50:00
- **Distinct Invoices:** 25,900
- **Cancellations:** 9,288 (InvoiceNo starting with 'C')
- **Negative Quantity Records:** 10,624
- **Missing Customer IDs:** 135,080
- **Zero or Negative Unit Price:** 2,517
- **Top 5 Countries by Revenue (GBP):**
  - **United Kingdom**: £9,025,222.08
  - **Netherlands**: £285,446.34
  - **EIRE**: £283,453.96
  - **Germany**: £228,867.14
  - **France**: £209,715.11

### Column Summary
| Column | Type | Min | Max | Mean | Missing | Notes |
|---|---|---|---|---|---|---|
| `InvoiceNo` | str | - | - | - | 0 | |
| `StockCode` | str | - | - | - | 0 | |
| `Description` | str | - | - | - | 1454 | |
| `Quantity` | int64 | -80995.0 | 80995.0 | 9.55 | 0 | |
| `InvoiceDate` | datetime64[us] | - | - | - | 0 | |
| `UnitPrice` | float64 | -11062.06 | 38970.0 | 4.61 | 0 | |
| `CustomerID` | str | - | - | - | 135080 | |
| `Country` | str | - | - | - | 0 | |

### Known Quirks
- Cancellations carry negative quantities and often mirror earlier valid invoices.
- Several stock codes are non-product administrative records (e.g. `POST`, `D`).
- Day 2 preserves all cancellations, negative quantities, and missing customer IDs as raw data; cleaning is deferred to Day 3.

---

## 4. M5 Forecasting Dataset (Optional)

- **Source Status:** Optional. If present in `data/raw/m5/`, consists of `calendar.csv`, `sales_train_validation.csv`, and `sell_prices.csv`.
- **Planned Subset:** Top 30 items in store `CA_1` across 1,913 days (57,390 rows).
- **Notes:** Unshifted demand benchmark for retail demand modeling.

---

## 5. Unified Internal Schema (Contract)

The target schema generated on Day 2 and stored under `data/raw/unified/`:

### 5.1 `machines.csv`
- `machine_id` (str, PK): Unique machine code (`M-101` to `M-112`)
- `machine_type` (str): `Air Compressor`, `Hydraulic Press`, `Conveyor Drive`
- `location` (str): Plant and production line
- `installation_date` (date): Date of installation
- `status` (str): Operational state snapshot at timeline end (`Running`, `Warning`, `Critical`, `Maintenance`, `Idle`)
- `data_origin` (str, extension): `metropt_real_signals` or `metropt_derived`
- `scenario` (str, extension): Simulation scenario (e.g. `hero_degrading`, `healthy`)
- `ai4i_type` (str, extension): Quality tier variant (`L`, `M`, `H`)

### 5.2 `sensors_raw.csv`
- `timestamp` (datetime ISO 8601 UTC): 5-minute sampling interval
- `machine_id` (str, FK): Identifier of the machine
- `temperature` (float, °C): Operating oil / body temperature
- `vibration` (float, mm/s RMS): Vibration level (synthetic formula with AR(1))
- `pressure` (float, bar): System pneumatic / hydraulic pressure
- `rpm` (float, rev/min): Rotational operating speed
- `energy_consumption` (float, kWh): 5-minute electrical energy consumption
- `operating_hours` (float, hours): Cumulative non-decreasing operating hours
- `motor_current` (float, A, extension): Motor phase current
- `comp_state` (int 0/1, extension): Compressor load/unload state

### 5.3 `maintenance_raw.csv`
- `maintenance_id` (str, PK): Sequential ID (`MNT-000001`...)
- `machine_id` (str, FK): Machine serviced
- `maintenance_date` (date): Date of intervention
- `maintenance_type` (str): `Inspection`, `Preventive`, `Corrective`, `Predictive`
- `failure_type` (str): Cause of corrective maintenance (`Air Leak` or `None`)
- `maintenance_cost` (float, GBP): Associated intervention cost

### 5.4 `production_raw.csv`
- `timestamp` (date 00:00 UTC): Production date
- `machine_id` (str, FK): Machine identifier
- `production_quantity` (int): Units produced
- `target_quantity` (int): Target planned units
- `downtime_minutes` (int): Total downtime (0 to 1,440 min)

### 5.5 `business_daily_raw.csv`
- `date` (date): Date
- `machine_id` (str, FK): Machine identifier
- `production` (int): Actual daily units produced
- `orders` (int): Demand orders allocated
- `inventory` (int): Non-negative finished goods inventory
- `downtime` (int): Downtime in minutes
- `maintenance_cost` (float): Direct maintenance expenditure in GBP

### 5.6 `business_transactions_raw.csv`
- `transaction_id` (str, PK): Unique transaction reference (`{InvoiceNo}-{LineNo}`)
- `timestamp` (datetime ISO 8601 UTC): Time-shifted transaction timestamp
- `product_id` (str): Product identifier (`StockCode`)
- `quantity` (int): Transacted quantity
- `unit_price` (float): Unit price in GBP
- `revenue` (float): Total transaction value (`Quantity * UnitPrice`)
- `customer_id` (str, nullable): Customer account ID
- `invoice_no` (str, extension): Original invoice number
- `description` (str, extension): Product description
- `country` (str, extension): Destination country

