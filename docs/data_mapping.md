# Data Mapping Specification & Transformation Architecture

**Project**: AI Operations Intelligence Platform  
**Phase**: Day 02 - Data Sources & Schema Integration  
**Date**: October 2026  
**Status**: APPROVED / IMPLEMENTED  

---

## 1. Executive Summary

This document specifies the exact mapping, transformations, derivations, and time-shift rules converting external raw datasets into the 8 standardized tables of the internal operational intelligence schema.

The system integrates three primary datasets:
1. **MetroPT-3**: Real high-frequency (1 Hz downsampled to 5 min) telemetry from an urban train air compressor with real air-leak failure windows.
2. **AI4I 2020**: Tabular predictive maintenance dataset with 10,000 synthetic observations across 5 physical failure modes.
3. **Online Retail**: 541,909 transactional records from a UK-based e-commerce gift retailer, driving the operational demand and revenue model.

---

## 2. Timeline Alignment & Shift Mechanics (D20)

To achieve historical realism while maintaining live operations relevance in October 2026, all datasets are shifted forward by **an exact integer number of whole weeks ($7 \times W$ days)**. This preserves:
- Diurnal (day-of-week) work cycles and weekend operational drop-offs.
- Inter-day and intra-day autocorrelation structure.
- Seasonality and shift patterns.

| Dataset | Raw Time Range | Whole Weeks Shift ($W$) | Shifted Time Range | Alignment Anchor |
| :--- | :--- | :--- | :--- | :--- |
| **MetroPT-3** | 2020-02-01 to 2020-09-01 | **320 weeks** (+2,240 days) | 2026-03-21 to 2026-10-20 | Last full day lands on `2026-10-19` |
| **Online Retail** | 2010-12-01 to 2011-12-09 | **776 weeks** (+5,432 days) | 2025-10-15 to 2026-10-23 | Completely envelopes 213-day operations window |

---

## 3. Source-to-Target Mappings

### 3.1 MetroPT-3 to `sensors_raw`

| Target Column | Target Type | Source Field | Transformation / Logic |
| :--- | :--- | :--- | :--- |
| `id` | `BIGSERIAL` / `INT PK` | Auto-generated | Row-level autoincrement primary key. |
| `timestamp` | `TIMESTAMPTZ` | `timestamp` | Source timestamp $+ 2240\text{ days}$ (UTC). |
| `machine_id` | `VARCHAR(10) FK` | Fleet Assignment | `M-101` through `M-112`. |
| `temperature` | `NUMERIC(6,2)` | `Oil_temperature` | Shifted, scaled by machine profile, hero degradation overlay. |
| `vibration` | `NUMERIC(6,3)` | Synthetic Physical Derivation | $1.6 + 1.2 L_t + 0.02(T_t - T_{ref}) + \text{AR}(1)\text{ noise}$. |
| `pressure` | `NUMERIC(6,3)` | `TP3` | Compressor discharge pressure in bar. |
| `motor_current` | `NUMERIC(6,2)` | `Motor_current` | Current in Amperes. Offloaded/Idle cutoff $I_{off} = 1.55\text{ A}$. |
| `energy_consumption`| `NUMERIC(8,4)` | Derived Physical | $\sqrt{3} \times 0.400\text{ kV} \times I_t \times 0.85 \times \frac{5}{60}\text{ kWh}$. |
| `rpm` | `NUMERIC(7,2)` | Synthetic Physical Derivation | $RPM_{nom} \times (1.0 - 0.02 L_t) + \mathcal{N}(0, \sigma)$. |
| `comp_state` | `NUMERIC(3,1)` | `COMP` | Electrical state (1.0 = off/offloaded, 0.0 = running under load). |
| `operating_hours` | `NUMERIC(10,2)` | Cumulative Derivation | Initial machine hours $+ \sum (I_t > 1.55\text{ A}) \times \frac{5}{60}\text{ h}$. |
| `created_at` | `TIMESTAMPTZ` | System Default | Current ingestion timestamp (`NOW()`). |

---

### 3.2 AI4I 2020 to `ai4i_state` & Fleet Failure Taxonomy

| Target Column | Target Type | Source Field (AI4I) | Transformation / Logic |
| :--- | :--- | :--- | :--- |
| `state_id` | `VARCHAR(36) PK` | `UDI` + `Product ID` | Deterministic UUIDv5 string. |
| `udi` | `INTEGER UNIQUE` | `UDI` | Preserved original integer record index (1..10000). |
| `product_id` | `VARCHAR(10)` | `Product ID` | Preserved product identifier (e.g. `M14860`). |
| `machine_type` | `VARCHAR(20)` | Constant | `"Milling Machine"`. |
| `machine_variant`| `VARCHAR(1)` | `Type` | Product variant tier (`"L"`, `"M"`, `"H"`). |
| `air_temperature_k`| `NUMERIC(6,2)` | `Air temperature [K]` | Ambient temperature in Kelvin (~300 K). |
| `process_temperature_k`| `NUMERIC(6,2)` | `Process temperature [K]`| Operating cycle temperature in Kelvin (~310 K). |
| `rotational_speed_rpm`| `INTEGER` | `Rotational speed [rpm]`| Spindle rotational velocity (RPM). |
| `torque_nm` | `NUMERIC(6,2)` | `Torque [Nm]` | Applied spindle torque in Newton-meters. |
| `tool_wear_min`| `INTEGER` | `Tool wear [min]` | Accumulated tool cutting time in minutes. |
| `machine_failure`| `BOOLEAN` | `Machine failure` | Binary flag (0 -> `False`, 1 -> `True`). |
| `twf` | `BOOLEAN` | `TWF` | Tool Wear Failure indicator. |
| `hdf` | `BOOLEAN` | `HDF` | Heat Dissipation Failure indicator. |
| `pwf` | `BOOLEAN` | `PWF` | Power Failure indicator. |
| `osf` | `BOOLEAN` | `OSF` | Overstrain Failure indicator. |
| `rnf` | `BOOLEAN` | `RNF` | Random Failure indicator. |

---

### 3.3 Online Retail to `business_transactions` & `business_daily`

| Target Column | Target Type | Source Field (Retail) | Transformation / Logic |
| :--- | :--- | :--- | :--- |
| `transaction_id` | `VARCHAR(36) PK` | Invoice + Line | Deterministic UUIDv5 identifier. |
| `invoice_no` | `VARCHAR(20)` | `InvoiceNo` | Preserved invoice number. 'C' prefix denotes cancellation. |
| `stock_code` | `VARCHAR(20)` | `StockCode` | Product SKU. Admin codes (`POST`, `D`, `M`, etc.) filtered out. |
| `description` | `TEXT` | `Description` | Cleaned product description string. |
| `quantity` | `INTEGER` | `Quantity` | Preserved item count. Negative values represent returns. |
| `unit_price` | `NUMERIC(10,2)` | `UnitPrice` | Unit price in GBP. Filtered `UnitPrice > 0`. |
| `customer_id` | `VARCHAR(20)` | `CustomerID` | Preserved nullable customer ID (empty string if anonymous). |
| `country` | `VARCHAR(50)` | `Country` | Customer delivery jurisdiction (default: `"United Kingdom"`). |
| `transaction_date`| `TIMESTAMPTZ` | `InvoiceDate` | Shifted $+ 5432\text{ days}$ (776 whole weeks) into 2026. |
| `is_cancellation` | `BOOLEAN` | `InvoiceNo` | `True` if `InvoiceNo.startswith('C')`, else `False`. |
| `line_total` | `NUMERIC(10,2)` | Calculated | $\text{quantity} \times \text{unit\_price}$. |

---

## 4. Derived Operational Entities

### 4.1 `production_daily`
- **Grain**: One record per machine-day ($12 \text{ machines} \times 213 \text{ days} = 2,556 \text{ rows}$).
- **Target Capacities**:
  - Air Compressor: 600 units/day
  - Hydraulic Press: 1,200 units/day
  - Conveyor Drive: 2,400 units/day
- **Operating Hours**: Directly computed from `sensors_raw.csv` by summing 5-minute intervals where `motor_current > 1.55 A`.
- **Hero Degradation Scrap Elevation**: On machine `M-104`, scrap rate elevates from a healthy baseline of 2.0% up to 10.5% over the final 14 days of the timeline, providing the business impact link for diagnostic agents.
- **OEE Formulation**:
  $$\text{Availability} = \frac{\text{operating\_hours}}{24.0}$$
  $$\text{Performance} = \min\left(1.0, \frac{\text{units\_produced}}{\text{units\_target} \times \text{Availability}}\right)$$
  $$\text{Quality} = \frac{\text{units\_produced} - \text{scrap\_units}}{\text{units\_produced}}$$
  $$\text{OEE} = \text{Availability} \times \text{Performance} \times \text{Quality}$$

### 4.2 `business_daily`
- **Grain**: Plant-wide daily summary ($213 \text{ days} = 213 \text{ rows}$).
- **Energy Cost**: $\sum \text{kWh} \times £0.22/\text{kWh}$.
- **Maintenance Cost**: Summed daily performed maintenance costs from `maintenance_raw.csv`.
- **Retail Demand Grounding**:
  $$\text{Demand Index}_t = \frac{\text{Retail Units}_t}{\text{Mean Retail Units}}$$
  $$\text{Daily Demanded Units}_t = \text{Capacity} \times \text{Demand Index}_t$$
- **Stock-Flow Balance**:
  $$\text{Inventory}_t = \max\left(0, \text{Inventory}_{t-1} + \text{Good Units}_t - \text{Shipped}_t\right)$$
  $$\text{Backorder}_t = \max\left(0, \text{Daily Demanded Units}_t - (\text{Inventory}_{t-1} + \text{Good Units}_t)\right)$$

---

## 5. Fleet Scenario Roster (12 Machines)

| Machine ID | Type | Plant / Line | Scenario Class | Key Operational Phenotype |
| :--- | :--- | :--- | :--- | :--- |
| **M-101** | Air Compressor | Plant A, Line 1 | Healthy | Baseline reference, vibration < 4.0 mm/s |
| **M-102** | Air Compressor | Plant A, Line 1 | Healthy | Baseline reference, vibration < 4.0 mm/s |
| **M-103** | Air Compressor | Plant A, Line 2 | Mild Drift | Gradual acoustic and current drift over final 21 days |
| **M-104** | Air Compressor | Plant A, Line 2 | Hero Degrading | Real MetroPT signals; inspection on 2026-09-20; vib warning at -13d |
| **M-105** | Hydraulic Press | Plant B, Line 1 | Past Failure Old | Pressure drop failure event at Day 93 (2026-06-21), resolved |
| **M-106** | Hydraulic Press | Plant B, Line 1 | Recently Maintained | Full 2000h service on 2026-10-05, new seals and oil |
| **M-107** | Hydraulic Press | Plant B, Line 2 | Overdue Maint | Last PM at 33,350h, current hours 35,786h (> 400h overdue) |
| **M-108** | Hydraulic Press | Plant B, Line 2 | Healthy | Baseline reference press, vibration < 4.0 mm/s |
| **M-109** | Conveyor Drive | Plant C, Line 1 | Benign Spikes | Intermittent short current/vibration spikes (no degradation) |
| **M-110** | Conveyor Drive | Plant C, Line 1 | Past Failure Mid | Overstrain bearing failure at Day 138 (2026-08-05), repaired |
| **M-111** | Conveyor Drive | Plant C, Line 2 | Past Failure Recent | Heat dissipation VFD failure at Day 155 (2026-08-22), repaired |
| **M-112** | Conveyor Drive | Plant C, Line 2 | In Maintenance | Ongoing overhaul started 2026-10-18 08:00 (active outage) |
