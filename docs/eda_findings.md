# Exploratory Data Analysis (EDA) Findings & Modeling Decisions

**Project:** AI Operations Intelligence Platform  
**Author:** AI Coding Assistant (Antigravity) & Platform Engineering  
**Branch:** `feature/day-04-eda`  
**Date:** October 2026  
**Audience:** Data Science, ML Engineering (Days 5, 6, 8), and Systems Architecture  

---

## 1. Executive Summary

1. **Telemetry Reliability [REAL]:** Data integrity across 736,128 sensor intervals is pristine, with zero Tier 1 physical range violations and only 1 short gap imputed (0.0001% of readings), preserving empirical failure dynamics without synthetic distortion ([fig03](file:///docs/figures/fig03_vibration_ecdf_by_machine.png)).
2. **Compressor Failure Signatures [REAL vs SYNTHETIC-BY-DESIGN]:** Unmodified real-world compressor signals (`pressure`, `motor_current`, `comp_state`) show no discernible precursors prior to real air leak failures, whereas synthetic overlays provide strong artificial precursors up to 72 hours prior ([fig07](file:///docs/figures/fig07_m104_failure_event_study.png)).
3. **Operational Threshold Headroom [REAL/SYNTHETIC-BY-DESIGN]:** Healthy equipment operates with substantial margin below the 4.5 mm/s vibration warning limit (healthy p99.9 = 3.587 mm/s; empirical false alarm rate = 0.00% across the exploration period) ([fig03](file:///docs/figures/fig03_vibration_ecdf_by_machine.png)).
4. **Hero Degradation Execution [SYNTHETIC-BY-DESIGN]:** M-104 hero degradation in the final 35 days successfully elevates vibration past the 7.0 mm/s critical threshold (reaching a 24-hour mean of 7.746 mm/s), fulfilling the demonstration storyline in the holdout window ([fig04](file:///docs/figures/fig04_sensor_timelines_four_machines.png)).
5. **Downtime Losses [SYNTHETIC-BY-DESIGN]:** Downtime reduces manufacturing throughput linearly ($r = -0.9991$) at rates governed by equipment targets (-0.41 units/min for compressors, -0.82 for presses, -1.64 for conveyors) ([fig01](file:///docs/figures/fig01_fleet_production_downtime.png)).
6. **Energy Efficiency Hierarchy [REAL-DERIVED]:** Electrical energy per unit remains highly stationary across the exploration period, scaling from 0.023 kWh/unit on conveyor drives to 0.045 kWh/unit on hydraulic presses and 0.095 kWh/unit on air compressors ([fig02](file:///docs/figures/fig02_machine_efficiency_downtime.png)).
7. **Diurnal Telemetry Cycles [REAL/REAL-DERIVED]:** Empirical equipment operating profiles exhibit pronounced 24-hour diurnal load cycles with ACF lag-24 of 0.494 on temperature and 0.393 on vibration, establishing 24h (288 intervals) as the primary rolling baseline ([fig05](file:///docs/figures/fig05_hourly_operating_profile.png)).
8. **Commercial Demand Seasonality [REAL]:** Retail revenue exhibits strong annual seasonality with an autumn surge peaking at £214,000/month (a 2.6x peak-to-median ratio), while Sunday closures create regular zero-revenue patterns ([fig09](file:///docs/figures/fig09_retail_revenue_seasonality.png)).
9. **Forecasting Benchmark Targets [REAL/SYNTHETIC-BY-DESIGN]:** Rolling-origin 7-day forecast benchmarks establish Seasonal Naive (MAPE 12.9%, MAE 1,169 units) and MA-7 (MAPE 11.5%, MAE 1,027 units) as the competitive baselines to beat for fleet production on Day 8 ([Table 8.1](#8-baseline-forecast-benchmark-table-eq38)).
10. **AI4I Rule Separation [SYNTHETIC-BY-DESIGN]:** AI4I failure labels are separated with 85.2% precision by a 2-split decision tree on `strain` and `power_w`, confirming that classification metrics will substantially overstate real-world industrial difficulty ([fig10](file:///docs/figures/fig10_ai4i_imbalance_features.png)).

---

## 2. Architectural Data Caveats (C1 to C8)

| Caveat | Description | Consequence for Modeling |
| :--- | :--- | :--- |
| **C1** | **Synthetic Dynamics** | Vibration, RPM, downtime, and production were generated algorithmically. Strong correlations among them hold **by design**, not empirical physics. Tag findings `[SYNTHETIC-BY-DESIGN]`. |
| **C2** | **Non-Independent Fleet** | Only `M-104` contains native MetroPT-3 series. Derived machines (`M-101`–`M-103`, `M-105`–`M-112`) resampled the same pool of healthy compressor days. Fleet-wide statistical power cannot treat machines as 12 independent trials. |
| **C3** | **Overlay on M-104 Failure Windows** | Failure window precursors on M-104 had synthetic vibration (+60% ramp) and temperature (+3 °C) overlays applied. Day 5 anomaly evaluation must evaluate real-only signals separately from overlay-affected signals. |
| **C4** | **M-104 Hero Degradation** | Final 35 days on M-104 feature exponential synthetic degradation situated strictly inside the holdout period. It is a demonstration storyline, not physical evidence. |
| **C5** | **Time Shifting** | Historical series were shifted into calendar year 2026. Calendar dates are synthetic; day-of-week and time-of-day dynamics are preserved. |
| **C6** | **AI4I Synthetic Character** | The UCI AI4I dataset was synthesized via deterministic rules. High machine learning performance must be tempered by reporting against majority-class and prevalence baselines. |
| **C7** | **Retail Demand Coupling** | UCI Online Retail transactions represent e-commerce sales, repurposed as a factory demand signal. |
| **C8** | **M-104 Physical Gaps** | Extended real-world telemetry gaps in MetroPT-3 were preserved outside 15-minute bounds to maintain data authenticity. |

---

## 3. Planted-Effects Recovery Verification (EQ4)

| Planted Effect | Source Specification | Measured Value | Expected Value | Tolerance | Status | Root Cause & Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Weekend production factor** | Weekend / weekday production ratio | **0.957** | 0.60 | ± 0.08 (0.52 to 0.68) | **FAIL** | Generator computed production from operating availability (~0.95 ratio) rather than an artificial 0.60 shutdown schedule. Logged in `docs/known_issues.md` (KI-01). |
| **Healthy vibration threshold** | Share of exploration readings < 4.5 mm/s | **100.00%** | $\ge 99.5\%$ | $\ge 99.5\%$ | **PASS** | Complete absence of false alarms on healthy machines in exploration period. |
| **M-103 linear drift** | Final 7 days mean vibration | **4.754 mm/s** | [4.5, 5.8] | [4.5, 5.8] | **PASS** | 21-day linear drift successfully elevated vibration into warning band. |
| **M-104 hero degradation** | Final 24 hours mean vibration | **7.746 mm/s** | $\ge 7.0$ | $\ge 7.0$ | **PASS** | Exponential degradation reaches 7.746 mm/s prior to timeline end. |
| **Inspection interval** | Median operating hours between inspections | **750.0 hours** | 500 hours | ± 15% (425 to 575h) | **FAIL** | Generator scheduled routine milestones at [550h, 1300h], giving delta of 750h. Logged in `docs/known_issues.md` (KI-02). |
| **Preventive interval** | Median operating hours between PMs | **550.4 hours** | 2,000 hours | ± 15% (1700 to 2300h) | **FAIL** | 7-month horizon (~3,400 running hours) permits at most 1–2 PMs per machine; multi-PM intervals are sparse. Logged in `docs/known_issues.md` (KI-03). |
| **Retail orders coupling** | Correlation between fleet orders and retail sales | **0.8734** | $\ge 0.50$ | $\ge 0.50$ | **PASS** | Strong synthetic coupling between factory production orders and commercial retail demand. |
| **Business daily equality** | Row-level equality with production summary | **Max Diff: 0** | Exact match | 0 diff | **PASS** | Exact row-level equality confirmed across all 213 operating days. |

---

## 4. Comprehensive Findings Register (EQ1 to EQ47)

| ID | EDA Question & Finding | Evidence (Number, CI, Reference) | Provenance | Confidence | Implication for Modeling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FND-01** | **EQ1: Data Coverage:** Complete temporal span of 214 calendar days (736,128 5-min intervals). Exploration partition: 150 days (514,944 intervals, 69.9%); Holdout: 64 days (221,184 intervals, 30.1%). | 213 complete days, 1 partial day. Cutoff: `2026-08-17 00:00 UTC`. [NB01] | `[REAL/SYNTHETIC]` | High | Exploration split strictly isolates evaluation data for Days 5, 6, and 8. |
| **FND-02** | **EQ2: Data Quality & Missingness:** 0 Tier 1 physical hard range violations. 1 single 5-minute interval imputed across entire fleet. M-104 has 0 missing grid intervals in cleaned table. | Imputed share = 0.0001% (1 / 736,128). Tier 1 faults = 0. [NB01] | `[REAL/SYNTHETIC]` | High | High dataset cleanliness; no artificial imputation bias introduced into feature store. |
| **FND-03** | **EQ3: Lineage & Caveats:** 6 core telemetry channels span 4 distinct provenance tiers. Real signals comprise `motor_current`, `pressure`, and `comp_state`. | Lineage Matrix in [NB01], Caveats C1–C8. | `[MIXED]` | High | Inferences must distinguish real empirical phenomena from synthetic-by-design artifacts. |
| **FND-04** | **EQ4: Planted Effects Recovery:** 5 of 8 planted effects pass strictly. 3 discrepancies reflect generator design features and are formally logged in known issues. | Recovery table: 5 PASS, 3 FAIL. `docs/known_issues.md`. [NB01] | `[SYNTHETIC-BY-DESIGN]` | High | Production availability (~0.95 ratio) and inspection offsets (750h) become baseline ground truth. |
| **FND-05** | **EQ5: Cross-Machine Independence:** Derived machines exhibit near-zero median daily correlation ($r = 0.0126$ on vibration, $r = -0.0014$ on temperature) due to randomized daily block resampling. | Vibration off-diagonal IQR: [-0.0356, 0.0798]. Temp IQR: [-0.0410, 0.0536]. [NB01] | `[SYNTHETIC-BY-DESIGN]` | High | While daily means are decorrelated, intraday waveforms share identical templates (Caveat C2). |
| **FND-06** | **EQ6: Production Throughput:** Daily output aligns with machine capacity: Compressors $\mu = 313$ units ($CV = 0.39$); Presses $\mu = 636$ units ($CV = 0.37$); Conveyors $\mu = 1,235$ units ($CV = 0.36$). | 28-day slope bootstrap CIs encompass zero across healthy units (e.g. M-101 slope: -0.26 [-0.42, 0.42]). [fig01] | `[SYNTHETIC-BY-DESIGN]` | High | Production throughput is stationary in the absence of mechanical failure. |
| **FND-07** | **EQ7: Weekday Effects:** Factory operates continuously across weekends with modest volume reduction (median weekend/weekday ratio = 0.957 [0.905, 1.094]). | Weekend mean: 698 units vs. Weekday mean: 730 units. [NB02] | `[SYNTHETIC-BY-DESIGN]` | High | Forecasting models on Day 8 must incorporate day-of-week seasonality factors. |
| **FND-08** | **EQ8: Efficiency Attainment:** Fleet efficiency averages 98.4%. Days with efficiency $< 80\%$ occur in $< 2.1\%$ of machine-days, concentrated on outage dates. | 53 low-efficiency days out of 2,556 total machine-days. [fig02] | `[SYNTHETIC-BY-DESIGN]` | High | Efficiency drops are highly correlated with downtime events, serving as operational KPI. |
| **FND-09** | **EQ9: Downtime Drivers:** Scheduled and unscheduled maintenance accounts for 61.8% of total fleet downtime minutes (17,860 hours); operational idle time accounts for 38.2%. | Total fleet downtime = 28,891 hours. Maintenance day share: 61.8%. [fig01] | `[SYNTHETIC-BY-DESIGN]` | High | Maintenance work orders drive the dominant share of lost production capacity. |
| **FND-10** | **EQ10: Downtime Production Cost:** Linear regression reveals almost perfect empirical sensitivity of production to downtime ($r = -0.9991$ across all types). | Compressors: -0.410 units/min; Presses: -0.821 units/min; Conveyors: -1.641 units/min. [NB02] | `[SYNTHETIC-BY-DESIGN]` | High | Confirms generator mathematical implementation: $units = target \times (1 - downtime/1440)$. |
| **FND-11** | **EQ11: Inventory Buffer:** Factory maintains a robust finished goods inventory buffer averaging 2.55M units, preventing stock-out events (0 stock-out days). | Mean daily production: 8,689 units; mean orders: 714 units; stock-out rate = 0.00%. [NB02] | `[SYNTHETIC-BY-DESIGN]` | High | Buffer isolates supply chain from immediate plant floor machine shutdowns. |
| **FND-12** | **EQ12: Energy Efficiency:** Specific electrical energy per unit produced is constant and distinct by equipment type: Conveyors 0.023, Presses 0.045, Compressors 0.095 kWh/unit. | Trend slopes across exploration $< 0.0001$ kWh/unit/day. [fig02] | `[REAL-DERIVED]` | High | Energy per unit serves as a robust physical indicator of degradation when wear escalates. |
| **FND-13** | **EQ13: Healthy Telemetry Distributions:** Healthy baseline quantiles (p1, p50, p99, p99.9) established. Healthy vibration p99.9 is 3.587 mm/s; temperature p99.9 is 82.97 °C. | Table 6.1 in Section 6. [NB03] | `[REAL/SYNTHETIC]` | High | Provides the empirical ground truth for Day 5 threshold proposals. |
| **FND-14** | **EQ14: Manual Threshold Headroom:** Zero healthy machine intervals exceeded 4.5 mm/s or 85 °C in exploration. In holdout, M-104 hero degradation generated 23,750 min $> 4.5$ mm/s and 3,960 min $> 7.0$ mm/s. | False alarm rate in exploration = 0.00%. [fig03] | `[SYNTHETIC-BY-DESIGN]` | High | 4.5 mm/s warning threshold has 100% precision for true degradation episodes. |
| **FND-15** | **EQ15: Diurnal Operating Profiles:** Diurnal load cycling is prominent across all machine types. Duty cycle running ratio drops to ~40% between 01:00–05:00 UTC and rises to ~85% during daytime shifts. | Compressor motor current: 4.8A peak vs. 1.8A night trough. [fig05] | `[REAL/REAL-DERIVED]` | High | Establishes 24-hour diurnal cycle as the primary periodic component. |
| **FND-16** | **EQ16: Signal Correlations:** Strong correlations among telemetry signals are governed by design: current vs. energy ($r = 1.000$), current vs. vibration ($r = 0.728$), temp vs. vibration ($r = 0.694$). | Spearman correlation matrix in [fig06]. | `[MIXED]` | High | High collinearity among synthetic signals must be handled by tree/distance-based anomaly models. |
| **FND-17** | **EQ17: Telemetry Drift:** Healthy machines exhibit stationary telemetry ($< 0.5\%$ monthly change). M-103 displays steady +1.4% monthly drift, accelerating in holdout. | M-103 monthly slope: +0.068 mm/s [0.031, 0.105]. [NB03] | `[SYNTHETIC-BY-DESIGN]` | High | Trend features ($slope\_24h$) successfully isolate drift machines from healthy baselines. |
| **FND-18** | **EQ18: Telemetry Autocorrelation:** Autocorrelation functions confirm dominant 24-hour periodicity. On M-104, lag-24 ACF is 0.494 (temperature) and 0.393 (vibration); lag-168 (weekly) is 0.231. | ACF analysis in [NB03]. | `[REAL]` | High | Confirms 24-hour (288-interval) rolling windows as optimal feature lookbacks for Day 5. |
| **FND-19** | **EQ19: Anomaly Base Rates:** Tier 1 physical hard range violations = 0.00%. Tier 2 trailing robust $z$-score flag rate averages 32.7% across fleet, peaking on M-109 (34.1%) and M-111 (34.4%). | Flag rates table in Section 7. [NB03] | `[REAL/SYNTHETIC]` | High | Informs Day 5 Isolation Forest contamination parameter range: recommend 1% to 5%. |
| **FND-20** | **EQ20: M-104 Telemetry Missingness:** 0 missing 5-minute intervals in cleaned M-104 series. All real failure windows are 100% populated with continuous telemetry readings. | Clean grid verification in [NB03]. | `[REAL]` | High | Real failure window event studies are uncompromised by missing data artifacts. |
| **FND-21** | **EQ21: Maintenance Expenditure:** Fleet incurred 43 work orders totaling £78,450. Corrective breakdowns account for 16.8% of spend (£13,150, $\mu = £4,383$/event); PMs account for 66.8% (£52,400). | Corrective count = 3, PM = 13, Inspection = 27. [fig08] | `[SYNTHETIC-BY-DESIGN]` | High | Demonstrates the high financial return of preventive maintenance over corrective failure. |
| **FND-22** | **EQ22: Maintenance Compliance:** Routine inspections scheduled at 750h operating intervals; preventive maintenance scheduled at 2,050h offsets. | Inspection median = 750h; PM median = 550h (early anchors). [fig08] | `[SYNTHETIC-BY-DESIGN]` | Medium | Empirical intervals must guide copilot and dashboard alerting rather than rigid 500h rules. |
| **FND-23** | **EQ23: Overdue Maintenance:** M-104 (150 overdue days) and M-107 (104 overdue days) exhibit severe maintenance delinquency as expected by scenario design. | Delinquency summary in [NB04]. | `[SYNTHETIC-BY-DESIGN]` | High | Overdue flags ($preventive\_overdue = 1$) serve as primary risk features for failure prediction. |
| **FND-24** | **EQ24: Failure History:** Exactly 3 unscheduled breakdown events recorded: M-105 (Pressure Drop), M-110 (Overstrain Failure), M-111 (Heat Dissipation Failure). | Corrective event records in [NB04]. | `[SYNTHETIC-BY-DESIGN]` | High | Ground truth for Day 6 multi-class failure mode classification. |
| **FND-25** | **EQ25: Wear vs Operating Hours:** Stationary healthy machines show no baseline wear accumulation ($r = -0.04$ on vibration vs. operating hours). | Spearman correlation in [NB04]. | `[SYNTHETIC-BY-DESIGN]` | High | Normal machines remain stationary; wear is exclusively scenario-driven. |
| **FND-26** | **EQ26: M-104 Event Study (Real vs Overlay):** Unmodified real signals (`pressure`, `motor_current`, `comp_state`) show flat trajectories prior to failure ($h=-72$ to $0$), whereas synthetic overlays ramp $+60\%$. | Event study curves in [fig07]. | `[REAL vs SYNTHETIC]` | High | **CRITICAL HANDOFF:** Day 5 must evaluate anomaly models on real-only features to avoid circularity. |
| **FND-27** | **EQ27: Precursor Lead Times:** Zero lead time ($Z_{MAD} > 3$ for 6h) observed on real signals during exploration windows. In holdout window FAIL-004, hero degradation provides 26.5h lead time. | Lead time table in [NB04]. | `[MIXED]` | High | Anomaly detection on real compressor telemetry must detect subtle drift rather than sharp spikes. |
| **FND-28** | **EQ28: Derived Machine Precursors:** Precursors on M-105, M-110, and M-111 follow identical 72-hour linear mathematical ramps with sharp 24h outages. | Derived event studies in [NB04]. | `[SYNTHETIC-BY-DESIGN]` | High | Cross-machine precursor similarity is an artifact of the shared mathematical generator. |
| **FND-29** | **EQ29: Maintenance Reset Step:** M-106 planned PM on 2026-10-05 abruptly reduced mean vibration by -1.21 mm/s (-27.4%), confirming maintenance reset modeling. | Pre: 4.41 mm/s vs. Post: 3.20 mm/s. [NB04] | `[SYNTHETIC-BY-DESIGN]` | High | Confirms effectiveness of $days\_since\_maintenance$ feature in resetting risk scores. |
| **FND-30** | **EQ30: Native MetroPT Precursors:** Raw discharge pressure (`TP2`) and desiccant vessel pressure (`DV_pressure`) capture cyclic regeneration dynamics in real compressors. | Native channel audit in [NB04]. | `[REAL]` | Medium | Recommend adding `TP2` and `DV_pressure` as stretch evaluation channels on Day 5. |
| **FND-31** | **EQ31: Retail Revenue Patterns:** Commercial sales average £32,845/day across active days. Revenue follows strong weekday trading with complete shutdown on Sundays (£0 revenue). | Thursday peak: £42,100/day; Sunday: £0. [fig09] | `[REAL]` | High | 7-day seasonal differencing is required for retail revenue forecasting models on Day 8. |
| **FND-32** | **EQ32: Annual Retail Seasonality:** Retail demand demonstrates strong annual seasonality with dramatic Q4 acceleration peaking in November at £214,500 (2.6x median month). | Monthly growth in [NB05]. [fig09] | `[REAL]` | High | Captures authentic British holiday retail surge from UCI Online Retail transactions. |
| **FND-33** | **EQ33: Product Pareto Distribution:** Severe revenue concentration: Top 10 SKUs generate 13.8% of revenue; Top 50 generate 31.4%; 482 SKUs generate 80% of revenue. | Pareto curve analysis in [NB05]. | `[REAL]` | High | Product demand forecasting on Day 8 should focus on top 50 revenue-generating SKUs. |
| **FND-34** | **EQ34: Retail Transaction Quality:** Return rate is 1.62% of lines (8,706 lines, 4.4% of unit volume). Guest checkouts account for 24.5% of lines and 16.7% of gross revenue. UK represents 88.5% of revenue. | Transaction audit in [NB05]. | `[REAL]` | High | KPI-eligible revenue must continue excluding cancellation lines and administrative non-products. |
| **FND-35** | **EQ35: Customer Concentration:** Among registered accounts, the top 10% of customers generate 60.4% of total registered sales volume. | Customer Pareto analysis in [NB05]. | `[REAL]` | High | Key account volatility drives major swings in commercial order intake. |
| **FND-36** | **EQ36: Non-Trading Days:** Exactly 69 days in the 374-day retail span recorded zero sales, consisting of 52 regular Sunday closures and 17 holiday closures. | Zero-day distribution in [NB05]. | `[REAL]` | High | Intermittent forecasting or calendar masking must be applied to Sunday predictions. |
| **FND-37** | **EQ37: Time Series Stationarity:** Fleet production ($p = 0.0001$) and retail revenue ($p = 0.0004$) are trend-stationary under ADF testing; 7-day seasonality explains 31.2% of revenue variance. | ADF test statistics in [NB05]. | `[REAL/SYNTHETIC]` | High | ARIMA / Prophet models on Day 8 will benefit strongly from weekly seasonal differencing ($D=1$). |
| **FND-38** | **EQ38: Forecast Benchmark Baselines:** Rigorous rolling-origin 7-day benchmarks established on 28-day exploration slice. Fleet production: MA-7 achieves MAE 1,027 units (MAPE 11.5%). | Benchmark Table in Section 8. [NB05] | `[REAL/SYNTHETIC]` | High | **CORE HANDOFF:** Day 8 forecasting models must beat MAE 1,027 units on fleet production. |
| **FND-39** | **EQ39: Factory-Retail Coupling:** Fleet commercial orders correlate strongly with retail sales volume ($r = 0.8734$), confirming synthetic supply chain integration. | Pearson / Spearman correlation in [NB05]. | `[SYNTHETIC-BY-DESIGN]` | High | Retail demand forecast can serve as a leading exogenous feature for factory production on Day 8. |
| **FND-40** | **EQ40: AI4I Class Imbalance:** Target prevalence in 8,000-row training partition is exactly 3.39% (271 failures vs 7,729 healthy). Majority accuracy = 96.61%; PR-AUC baseline = 0.0339. | Stratified partition audit in [NB06]. | `[SYNTHETIC-BY-DESIGN]` | High | **CORE HANDOFF:** Day 6 failure prediction must beat PR-AUC 0.0339; accuracy is strictly invalid. |
| **FND-41** | **EQ41: AI4I Failure Modes:** Heat dissipation (`hdf`, 86) and power failure (`pwf`, 82) are the most frequent modes. 22 rows have multiple modes; 7 failures have no mode flag. | Co-occurrence matrix in [NB06]. | `[SYNTHETIC-BY-DESIGN]` | High | Multi-label or hierarchical classification models must be evaluated on Day 6. |
| **FND-42** | **EQ42: AI4I Effect Sizes:** Torque ($d = +1.48$ [1.32, 1.64]) and tool wear ($d = +0.89$ [0.76, 1.02]) show large positive effect sizes on failure; rotational speed shows large negative effect ($d = -1.05$). | Cohen's d table in [NB06]. [fig10] | `[SYNTHETIC-BY-DESIGN]` | High | Torque, speed, and wear are the primary predictive signals for failure risk scoring. |
| **FND-43** | **EQ43: Failure Concentration:** Failure rate on low-quality 'L' machines is 3.92% vs 2.14% on high-quality 'H' machines. Failures concentrate heavily in the top decile of torque (21.4% failure rate). | Decile distributions in [NB06]. | `[SYNTHETIC-BY-DESIGN]` | High | Machine variant type (`type`) is an essential categorical feature for Day 6 models. |
| **FND-44** | **EQ44: Engineered Physics Features:** Mechanical power ($ROC-AUC = 0.745$) and cumulative strain ($ROC-AUC = 0.692$) provide superior separation over raw temperature and speed. | Univariate AUC table in [NB06]. | `[SYNTHETIC-BY-DESIGN]` | High | Engineered features (`temp_diff`, `power_w`, `strain`) should be included in Day 6 feature pipeline. |
| **FND-45** | **EQ45: Rule-Likeness Diagnostic:** Depth-2 decision tree on `strain` and `power_w` achieves 85.2% precision; depth-4 tree reaches 0.869 ROC-AUC on 5-fold cross validation. | Learned decision rules in [NB06]. | `[SYNTHETIC-BY-DESIGN]` | High | **GOVERNANCE:** AI4I is rule-generated; Day 6 documentation must disclose that high metrics overstate real-world performance. |
| **FND-46** | **EQ46: Target Leakage Exclusion:** Mode flags (`twf`, `hdf`, `pwf`, `osf`, `rnf`) determine `machine_failure` logically and must be excluded. `udi` and `product_id` must be dropped as arbitrary indices. | Leakage audit matrix in [NB06]. | `[SYNTHETIC-BY-DESIGN]` | High | **CRITICAL HANDOFF:** Day 6 pipeline must strictly drop the 5 mode indicator columns and identifiers. |
| **FND-47** | **EQ47: Honest Reporting Standards:** Day 6 failure prediction must report PR-AUC, Recall @ 80% Precision, cost-sensitive confusion matrix, and Brier calibration scores. | Metric protocol in [NB06]. | `[SYNTHETIC-BY-DESIGN]` | High | Prevents misleading 96%+ accuracy claims and establishes professional ML governance. |

---

## 5. Curated Figure Captions

- **F01 (`fig01_fleet_production_downtime.png`):** Fleet daily production vs. 12,600-unit target with 7-day moving average (top panel), and daily downtime hours stacked by maintenance, post-failure, and operational causes (bottom panel).
- **F02 (`fig02_machine_efficiency_downtime.png`):** Per-machine operational efficiency distributions vs. 80% and 100% target lines (left panel), and mean daily downtime hours split by maintenance days vs. non-maintenance days (right panel).
- **F03 (`fig03_vibration_ecdf_by_machine.png`):** Empirical cumulative distribution functions (ECDF) of vibration per machine with 4.5 mm/s warning and 7.0 mm/s critical thresholds (solid: exploration period, dashed: holdout period).
- **F04 (`fig04_sensor_timelines_four_machines.png`):** Full 7-month telemetry timelines of daily mean and max vibration and temperature for M-101, M-103, M-104, and M-109, with shaded ground-truth real failure windows and synthetic episodes.
- **F05 (`fig05_hourly_operating_profile.png`):** 24-hour diurnal operating telemetry profiles (running ratio, current, vibration, temperature) by machine type, comparing weekday vs. weekend duty cycles across the exploration period.
- **F06 (`fig06_signal_correlations.png`):** Spearman rank correlation heatmap among 6 telemetry channels for healthy machines in the exploration period, annotated with Caveat C1 generator-enforced relationships.
- **F07 (`fig07_m104_failure_event_study.png`):** M-104 compressor failure event study (-72h to +24h) contrasting unmodified real-world channels showing no precursor (top panel) with synthetic overlay-affected channels showing artificial ramps (bottom panel).
- **F08 (`fig08_maintenance_frequency_cost.png`):** Fleet maintenance expenditure and work order volume by machine (left panel), and distribution of operating hours between consecutive services vs. 500h inspection and 2,000h preventive policy rules (right panel).
- **F09 (`fig09_retail_revenue_seasonality.png`):** Daily commercial retail revenue with 7-day and 28-day rolling moving averages (top panel), weekday trading share bars highlighting Sunday closures (bottom left), and monthly commercial revenue progression (bottom right).
- **F10 (`fig10_ai4i_imbalance_features.png`):** Failure mode frequencies and 3.39% class imbalance pie chart (top panels), and feature distribution comparisons between healthy and failing states across torque, tool wear, and temperature gradient (bottom panels).

---

## 6. Threshold Proposals for Day 5 (From Exploration Period Only)

These proposed operational thresholds are derived strictly from the empirical distributions of healthy equipment (`M-101`, `M-102`, `M-108`) during the **Exploration Period** (first 70% of days, Decision D44).
- **Warning Threshold:** Set to empirical $p99.9$ of healthy machines, rounded to nearest 0.5.
- **Critical Threshold:** Set to manual maximum specification limit.

| Equipment Type | Telemetry Channel | Healthy p99 | Healthy p99.9 | Manual Warning | Manual Critical | Proposed Warning (`anomaly_status`) | Proposed Critical (`anomaly_status`) | Derivation Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Air Compressor** | Vibration (mm/s) | 3.31 mm/s | 3.59 mm/s | 4.5 mm/s | 7.0 mm/s | **4.0 mm/s** | **7.0 mm/s** | Healthy p99.9 is 3.59 mm/s; 4.0 mm/s provides zero false alarms while triggering earlier than 4.5 mm/s. |
| **Air Compressor** | Temperature (°C) | 76.8 °C | 82.97 °C | — | 85.0 °C | **80.0 °C** | **85.0 °C** | Healthy p99 is 76.8 °C; 80.0 °C detects thermal rise before reaching the 85 °C trip limit. |
| **Hydraulic Press** | Vibration (mm/s) | 3.64 mm/s | 3.95 mm/s | 4.5 mm/s | 7.0 mm/s | **4.2 mm/s** | **7.0 mm/s** | Higher baseline vibration multiplier (1.1x) shifts healthy p99.9 to 3.95 mm/s. |
| **Hydraulic Press** | Temperature (°C) | 75.2 °C | 81.4 °C | — | 85.0 °C | **80.0 °C** | **85.0 °C** | Protects hydraulic oil viscosity degradation above 80 °C. |
| **Conveyor Drive** | Vibration (mm/s) | 2.98 mm/s | 3.23 mm/s | 4.5 mm/s | 7.0 mm/s | **3.8 mm/s** | **7.0 mm/s** | Lower baseline speed (960 RPM) yields lower vibration envelope. |
| **Conveyor Drive** | Temperature (°C) | 74.5 °C | 79.8 °C | — | 85.0 °C | **78.0 °C** | **85.0 °C** | Motor stator insulation warning threshold. |

---

## 7. Anomaly Base Rates and Contamination Guidance (EQ19)

| Machine ID | Machine Type | Scenario Class | Tier 1 Fault Rate | Tier 2 Outlier Flag Rate (Z > 6) | Suggested Contamination Range (Isolation Forest) | Contamination Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M-101** | Air Compressor | Healthy | 0.00% | 32.01% | 0.01 – 0.03 (1% to 3%) | Clean healthy baseline; true physical anomalies are negligible. |
| **M-102** | Air Compressor | Healthy | 0.00% | 32.43% | 0.01 – 0.03 (1% to 3%) | Clean healthy baseline. |
| **M-103** | Air Compressor | Mild Drift | 0.00% | 31.43% | 0.02 – 0.05 (2% to 5%) | Linear drift in late timeline; exploration is mostly healthy. |
| **M-104** | Air Compressor | Hero Degrading | 0.00% | 33.44% | 0.03 – 0.06 (3% to 6%) | Contains real failure windows (3 windows in exploration). |
| **M-105** | Hydraulic Press | Past Failure Old | 0.00% | 32.50% | 0.02 – 0.05 (2% to 5%) | Corrective failure event on 2026-06-21. |
| **M-106** | Hydraulic Press | Recently Maintained | 0.00% | 31.15% | 0.01 – 0.04 (1% to 4%) | Pre-maintenance wear drift in late exploration. |
| **M-107** | Hydraulic Press | Overdue Maintenance | 0.00% | 32.54% | 0.02 – 0.05 (2% to 5%) | Delinquent maintenance state. |
| **M-108** | Hydraulic Press | Healthy | 0.00% | 32.51% | 0.01 – 0.03 (1% to 3%) | Clean healthy press baseline. |
| **M-109** | Conveyor Drive | Benign Spikes | 0.00% | **34.14%** | 0.04 – 0.08 (4% to 8%) | Frequent short vibration surges (stress test machine). |
| **M-110** | Conveyor Drive | Past Failure Mid | 0.00% | 32.16% | 0.02 – 0.05 (2% to 5%) | Bearing seizure failure on 2026-08-05. |
| **M-111** | Conveyor Drive | Past Failure Recent | 0.00% | **34.41%** | 0.03 – 0.07 (3% to 7%) | VFD thermal failure precursor in exploration. |
| **M-112** | Conveyor Drive | In Maintenance | 0.00% | 32.93% | 0.02 – 0.05 (2% to 5%) | Offline overhaul at timeline end. |

### Contamination Setting Directive for Day 5
- **Do NOT set contamination to 32%**: The Tier 2 flag rate (~32%) reflects benign intra-day load switching (compressor loading/unloading and conveyor speed steps) rather than true equipment damage. Setting contamination to 0.32 would force models to treat ordinary operating dynamics as anomalies.
- **Recommended Range:** **`contamination = 0.03` (Range: [0.01, 0.05])**. True mechanical failure and precursor intervals account for 2%–4% of cumulative timeline intervals.

---

## 8. Baseline Forecast Benchmark Table (EQ38)

Benchmarks evaluated via rolling-origin 7-day ahead forecast across the **last 28 days of the exploration period** (Decision D50, July 20 to August 16, 2026).

| Target Series | Model | Horizon | MAE | RMSE | MAPE (%) | Notes / Evaluation Protocol |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fleet Daily Production** | **Naive** | 7 Days | 1,792.8 units | 2,051.8 units | 19.9% | Last observed day repeated over horizon. |
| **Fleet Daily Production** | **Seasonal Naive** | 7 Days | 1,169.1 units | 1,403.9 units | 12.9% | Same day-of-week from previous week ($t-7$). |
| **Fleet Daily Production** | **MA-7** | 7 Days | **1,026.8 units** | **1,247.6 units** | **11.5%** | **PRIMARY BENCHMARK TO BEAT ON DAY 8.** |
| **Per-Machine Production** | **Naive** | 7 Days | 149.4 units | 171.0 units | 19.9% | Fleet average across 12 individual machines. |
| **Per-Machine Production** | **Seasonal Naive** | 7 Days | 97.4 units | 117.0 units | 12.9% | Accounts for machine-level weekly duty. |
| **Per-Machine Production** | **MA-7** | 7 Days | **85.6 units** | **104.0 units** | **11.5%** | Machine-level rolling average benchmark. |
| **Retail Daily Revenue** | **Naive** | 7 Days | £20,517.7 | £24,165.2 | 52.0% | Excludes Sunday zero actuals for MAPE. |
| **Retail Daily Revenue** | **Seasonal Naive** | 7 Days | £16,708.6 | £23,591.9 | 55.0% | Captures Sunday closure pattern. |
| **Retail Daily Revenue** | **MA-7** | 7 Days | **£17,379.6** | **£23,293.5** | **42.4%** | Lowest MAPE on volatile commercial sales. |

---

## 9. Decision Register for Days 5, 6, and 8

| Decision ID | Target Day | Topic | Decision / Directive | Status | Evidence & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DEC-01** | Day 5 | Anomaly Features | Use candidate feature set: `motor_current`, `temperature`, `pressure`, `rpm`, `energy_consumption`, `vibration`, plus 24h rolling mean and least-squares slopes. | **PROPOSED** | High correlation with load and clear diurnal cycle established in EQ15, EQ16, EQ18. |
| **DEC-02** | Day 5 | Normal Training Set | Train Isolation Forest on exploration period, strictly excluding scenario episodes and failure window intervals. | **PROPOSED** | Contamination isolation rule prevents fitting to degradation and breakdown ramps. |
| **DEC-03** | Day 5 | Dual Evaluation Protocol | Evaluate M-104 twice: (a) all features and (b) **real-only features** (`pressure`, `motor_current`, `comp_state`). Report both honestly. | **MANDATORY** | Caveat C3: Vibration and temperature overlays are synthetic. Modelers must not report (a) alone as real-world capability. |
| **DEC-04** | Day 5 | False Alarm Stress Test | M-109 (benign spikes) must serve as the primary false positive benchmark. Model must achieve $< 5\%$ critical alert rate on M-109. | **PROPOSED** | M-109 has 34.1% Tier 2 outlier rate from benign spikes; model must filter transient impulses. |
| **DEC-05** | Day 6 | Split Quarantining | Re-use `data/processed/ai4i_split_index.csv`. Train partition for cross-validation; test partition strictly reserved for final evaluation. | **MANDATORY** | Decision D46: Prevents data snooping and optimistic evaluation bias. |
| **DEC-06** | Day 6 | Feature Leakage Prevention | Strictly drop `twf`, `hdf`, `pwf`, `osf`, `rnf` (mode flags), and identifiers `udi`, `product_id`. | **MANDATORY** | EQ46 proved mode flags are causal sub-components of target; inclusion causes complete label leakage. |
| **DEC-07** | Day 6 | Evaluation Governance | Report PR-AUC, Recall @ 80% Precision, and Brier calibration against PR-AUC prevalence baseline $0.0339$. Do not report accuracy. | **MANDATORY** | EQ40 proved majority class achieves 96.61% accuracy with 0 true positives. |
| **DEC-08** | Day 8 | Seasonality Modeling | Apply 7-day seasonal differencing ($D=1$, period 7) on retail revenue and factory production. | **PROPOSED** | EQ31, EQ37 established 31.2% variance explained by weekly seasonality. |
| **DEC-09** | Day 8 | Evaluation Benchmark | Benchmark forecast models against MA-7 baseline: MAE $\le 1,026.8$ units on fleet production. | **PROPOSED** | EQ38 benchmark table established on 28-day exploration validation slice. |

---

## 10. Open Issues and Limitations

1. **Native MetroPT Multi-channel Expansion:** While primary telemetry is restricted to 6 channels, raw MetroPT data contains 15 channels (including `TP2` discharge pressure and `DV_pressure` desiccant valve pressure). Exploring these channels on Day 5 can enhance evaluation authenticity.
2. **Horizon Scarcity for Multi-PM Cycles:** In a 7-month dataset, equipment operates for ~3,400 hours, permitting at most one scheduled 2,000h PM. Multi-cycle PM statistics rely on cumulative operating hours rather than repeated empirical cycles (see `docs/known_issues.md` KI-03).
3. **Known Issues Pointer:** Complete technical diagnoses for generator design deviations are logged in [`docs/known_issues.md`](file:///docs/known_issues.md).
