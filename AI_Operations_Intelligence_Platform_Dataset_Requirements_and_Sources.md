# AI Operations Intelligence Platform --- Dataset Requirements & Sources

## Purpose

This document defines the datasets required for the **AI Operations
Intelligence Platform**.

The recommended architecture is:

``` text
Sensor / IoT Data
        +
Business Data
        +
Maintenance / Failure Data
        +
Operational Documents
        ↓
Data Engineering
        ↓
ML / Deep Learning
        ↓
Forecasting + Anomaly Detection + Prediction
        ↓
SHAP + LLM + RAG
        ↓
Business Insights
        ↓
Web Dashboard
```

------------------------------------------------------------------------

# 1. Dataset Strategy

  -----------------------------------------------------------------------
  Category                Purpose                 Recommended Source
  ----------------------- ----------------------- -----------------------
  Sensor / IoT Data       Machine health and      MetroPT-3 / AI4I 2020
                          operational monitoring  

  Failure / Maintenance   Predictive maintenance  AI4I 2020 / MetroPT-3 /
  Data                    and risk prediction     NASA C-MAPSS

  Time-Series Data        Forecasting and trend   M5 / MetroPT-3 / Gas
                          analysis                Turbine

  Business Data           Orders, sales,          UCI Online Retail / M5
                          inventory and revenue   

  Documents               RAG, maintenance        Dataset documentation +
                          explanations and        synthetic documents
                          operational knowledge   
  -----------------------------------------------------------------------

## Recommended Core Datasets

1.  **MetroPT-3** --- primary sensor + anomaly detection
2.  **AI4I 2020 Predictive Maintenance** --- machine failure
    classification
3.  **UCI Online Retail** --- business / transaction data
4.  **M5 Forecasting** --- demand forecasting
5.  **NASA C-MAPSS** --- advanced degradation / RUL experiments

Optional:

6.  SECOM
7.  Gas Turbine CO and NOx
8.  MIMII

------------------------------------------------------------------------

# 2. MetroPT-3 Dataset

### Role

**Primary sensor and anomaly-detection dataset**

MetroPT-3 contains operational measurements from a metro train's
compressor Air Production Unit. It includes pressure, temperature, motor
current and other operational measurements.

The UCI repository lists approximately **1.5 million instances and 15
features** and provides a data-description PDF.

### Useful fields

``` text
timestamp
TP2
TP3
H1
DV_pressure
Reservoirs
Oil_temperature
Motor_current
COMP
```

### Use in the project

``` text
Sensor Data
     ↓
Cleaning
     ↓
Feature Engineering
     ↓
Isolation Forest
     ↓
Anomaly Score
     ↓
Operational Risk
```

### Official source

**UCI Machine Learning Repository**

https://archive.ics.uci.edu/dataset/791/metropt3%2Bdataset

### DOI

https://doi.org/10.24432/C5VW3R

### Files

``` text
MetroPT3(AirCompressor).csv
Data Description_Metro.pdf
```

------------------------------------------------------------------------

# 3. AI4I 2020 Predictive Maintenance Dataset

### Role

**Primary machine-failure prediction dataset**

AI4I 2020 is a synthetic dataset designed to reflect industrial
predictive-maintenance data.

It contains **10,000 observations** and machine-operation variables such
as:

``` text
Air temperature
Process temperature
Rotational speed
Torque
Tool wear
Machine failure
```

Failure indicators include:

``` text
TWF
HDF
PWF
OSF
RNF
```

### Use in the project

``` text
Machine Features
       ↓
Feature Engineering
       ↓
XGBoost / LightGBM
       ↓
Failure Probability
       ↓
Risk Category
```

### Official source

https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset

### DOI

https://doi.org/10.24432/C5HS5C

### License

CC BY 4.0.

------------------------------------------------------------------------

# 4. UCI Online Retail Dataset

### Role

**Business / transaction dataset**

The UCI Online Retail dataset contains approximately **541,909
transactions** from a UK-based online retailer.

### Important fields

``` text
InvoiceNo
StockCode
Description
Quantity
InvoiceDate
UnitPrice
CustomerID
Country
```

### Use in the project

Transform transactions into:

``` text
Daily Sales
Revenue
Order Volume
Product Demand
Customer Activity
Product Trends
```

Pipeline:

``` text
Transactions
     ↓
Daily Aggregation
     ↓
Business KPIs
     ↓
Dashboard
```

### Official source

https://archive.ics.uci.edu/dataset/352/online%2Bretail

### Alternative: Online Retail II

https://archive.ics.uci.edu/dataset/502/online%2Bretail

------------------------------------------------------------------------

# 5. M5 Forecasting Dataset

### Role

**Demand forecasting dataset**

M5 is useful for forecasting product sales and demonstrating the
forecasting layer of the platform.

### Important files

``` text
calendar.csv
sales_train_validation.csv
sales_train_evaluation.csv
sell_prices.csv
sample_submission.csv
```

### Use in the project

``` text
Historical Sales
       ↓
Feature Engineering
       ↓
Forecasting Model
       ↓
Future Demand
       ↓
Business Recommendation
```

Possible models:

``` text
Moving Average
XGBoost
LightGBM
LSTM
```

### Official source

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

### Recommendation

Do not process the full dataset for the MVP. Select a limited number of
stores/products.

------------------------------------------------------------------------

# 6. NASA C-MAPSS Jet Engine Dataset

### Role

**Advanced degradation and Remaining Useful Life dataset**

NASA C-MAPSS contains multivariate time-series data from simulated
aircraft engines. Engines develop faults during operation, making the
dataset useful for degradation and RUL experiments.

### Use in the project

``` text
Engine Sensor Data
       ↓
Time-Series Processing
       ↓
Degradation Modeling
       ↓
RUL Prediction
       ↓
Maintenance Recommendation
```

### Data includes

``` text
Unit Number
Cycle
Operational Settings
Sensor Measurements
```

### Official source

https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

### Recommendation

Keep C-MAPSS as an **advanced extension**, not a core dependency for the
3-week MVP.

------------------------------------------------------------------------

# 7. SECOM Semiconductor Manufacturing Dataset

### Role

**Manufacturing process monitoring**

SECOM contains data from a semiconductor manufacturing process with
hundreds of monitored variables.

### Useful for

-   Feature selection
-   Manufacturing process monitoring
-   Classification
-   Sensor-quality analysis
-   Root-cause investigation

### Official source

https://archive.ics.uci.edu/dataset/179/secom

------------------------------------------------------------------------

# 8. Gas Turbine CO and NOx Dataset

### Role

**Industrial sensor and time-series dataset**

The dataset contains approximately **36,733 observations** and 11 sensor
measurements aggregated hourly from a gas turbine.

### Important variables

``` text
Ambient Temperature
Ambient Pressure
Ambient Humidity
Air Filter Difference Pressure
Gas Turbine Exhaust Pressure
Turbine Inlet Temperature
Turbine After Temperature
Compressor Discharge Pressure
Turbine Energy Yield
CO
NOx
```

### Useful for

-   Sensor analytics
-   Regression
-   Industrial forecasting
-   Operational monitoring
-   Environmental analysis

### Official source

https://archive.ics.uci.edu/dataset/551/gas%2Bturbine%2Bco%2Band%2Bnox%2Bemission%2Bdata%2Bset

### DOI

https://doi.org/10.24432/C5WC95

### License

CC BY 4.0.

------------------------------------------------------------------------

# 9. MIMII Industrial Machine Sound Dataset

### Role

**Optional deep-learning anomaly detection**

MIMII contains normal and anomalous sounds from:

``` text
Valves
Pumps
Fans
Slide Rails
```

### Use in the project

``` text
Machine Audio
       ↓
Spectrogram / Audio Features
       ↓
Deep Learning
       ↓
Anomaly Detection
       ↓
Machine Health Score
```

### Official source

https://zenodo.org/records/3384388

### Recommendation

Do not include the full MIMII dataset in the initial MVP. It is very
large. Use a small subset only if adding an audio-anomaly extension.

Check the current Zenodo terms before redistribution or commercial use.

------------------------------------------------------------------------

# 10. Document Dataset / RAG Corpus

The platform's document layer should contain:

``` text
Machine Manuals
Maintenance SOPs
Inspection Reports
Equipment Specifications
Maintenance History
Safety Procedures
Troubleshooting Guides
```

For this portfolio project, a small controlled document corpus is better
than searching for a huge generic document dataset.

## Recommended structure

``` text
documents/
│
├── manuals/
│   ├── machine_m104_manual.pdf
│   ├── machine_m105_manual.pdf
│   └── compressor_manual.pdf
│
├── maintenance/
│   ├── maintenance_sop.pdf
│   ├── preventive_maintenance.pdf
│   └── maintenance_history.pdf
│
├── inspection/
│   ├── inspection_report_m104.pdf
│   └── inspection_report_m105.pdf
│
└── specifications/
    ├── machine_specifications.pdf
    └── operating_limits.pdf
```

------------------------------------------------------------------------

# 11. Public Documents for the Initial RAG Prototype

The MetroPT-3 source includes a **Data Description PDF** that can be
used as an initial document for testing document ingestion and
retrieval.

Source:

https://archive.ics.uci.edu/dataset/791/metropt3%2Bdataset

The NASA Prognostics Center of Excellence repository also provides
dataset documentation and data resources:

https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

------------------------------------------------------------------------

# 12. Synthetic Company Documents

For the final portfolio application, create fictional but technically
realistic documents and clearly label them as **synthetic/demo
documents**.

## Machine Manual

``` text
Machine ID: M-104

Operating Temperature:
20°C – 85°C

Normal Vibration:
0 – 4.5 mm/s

Warning Vibration:
4.5 – 7.0 mm/s

Critical Vibration:
> 7.0 mm/s

Recommended Inspection:
Every 500 operating hours
```

## Maintenance SOP

``` text
Procedure:
1. Stop machine
2. Isolate power
3. Inspect bearings
4. Check temperature
5. Check vibration
6. Record findings
7. Approve machine restart
```

## Inspection Report

``` text
Machine: M-104

Inspection Date: 2026-09-20

Temperature: Elevated
Vibration: Above normal
Bearing condition: Warning
Maintenance status: Overdue

Recommendation:
Perform preventive maintenance.
```

These documents provide enough content for RAG questions such as:

> What is the recommended maintenance interval for Machine M-104?

------------------------------------------------------------------------

# 13. Dataset-to-Architecture Mapping

  -----------------------------------------------------------------------
  Project Component       Dataset                 Main Purpose
  ----------------------- ----------------------- -----------------------
  Sensor Data             MetroPT-3               Operational sensor
                                                  monitoring

  Machine Failure         AI4I 2020               Failure prediction

  Business Data           UCI Online Retail       Sales / transaction
                                                  KPIs

  Demand Forecasting      M5                      Future demand
                                                  prediction

  Degradation / RUL       NASA C-MAPSS            Advanced predictive
                                                  maintenance

  Manufacturing Analytics SECOM                   High-dimensional
                                                  process monitoring

  Industrial Sensors      Gas Turbine             Industrial time-series
                                                  analysis

  Audio Anomaly           MIMII                   Optional deep-learning
                                                  extension

  Documents / RAG         Metro PDF + synthetic   AI reasoning /
                          docs                    retrieval
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 14. Recommended Dataset Combination for the 3-Week MVP

Do **not** use every dataset.

Use these four as the core:

### 1. MetroPT-3

For:

``` text
Sensors
Anomaly Detection
Time-Series Analysis
```

### 2. AI4I 2020

For:

``` text
Failure Prediction
Classification
SHAP
```

### 3. UCI Online Retail

For:

``` text
Business Data
Revenue
Orders
Product Demand
```

### 4. M5

For:

``` text
Demand Forecasting
```

Then create a small synthetic document corpus for:

``` text
RAG
Maintenance Knowledge
AI Reasoning
```

------------------------------------------------------------------------

# 15. Unified Internal Data Model

The public datasets have different schemas. Normalize them into an
internal project schema.

## Machine Table

``` text
machine_id
machine_type
location
installation_date
status
```

## Sensor Readings

``` text
timestamp
machine_id
temperature
vibration
pressure
rpm
energy_consumption
operating_hours
```

## Production

``` text
timestamp
machine_id
production_quantity
target_quantity
downtime_minutes
```

## Maintenance

``` text
maintenance_id
machine_id
maintenance_date
maintenance_type
failure_type
maintenance_cost
```

## Business Transactions

``` text
transaction_id
timestamp
product_id
quantity
unit_price
revenue
customer_id
```

## Predictions

``` text
prediction_id
machine_id
timestamp
failure_probability
risk_level
model_version
```

## Anomalies

``` text
anomaly_id
machine_id
timestamp
anomaly_score
anomaly_type
severity
```

## AI Insights

``` text
insight_id
timestamp
machine_id
insight_type
explanation
recommendation
severity
```

------------------------------------------------------------------------

# 16. Data Integration Strategy

The public datasets come from different domains. Do not present them as
if they were originally collected from one company.

Instead, create a **unified demonstration environment**.

``` text
                 PUBLIC DATASETS
                       │
       ┌───────────────┼───────────────┐
       │               │               │
    MetroPT-3       AI4I           Online Retail
       │               │               │
       └───────────────┼───────────────┘
                       ↓
              DATA STANDARDIZATION
                       │
                       ↓
             INTERNAL DATA MODEL
                       │
                       ↓
              POSTGRESQL DATABASE
                       │
             ┌─────────┼─────────┐
             ↓         ↓         ↓
        ML Models   Forecasts  Analytics
```

For the portfolio application, create a synthetic company/machine ID
layer that maps the public datasets into the common demo schema.
Document this transformation in the README.

------------------------------------------------------------------------

# 17. Feature Engineering

## Sensor Features

``` text
rolling_temperature_mean
rolling_temperature_std
rolling_vibration_mean
rolling_vibration_std
temperature_change
vibration_change
pressure_change
energy_change
operating_hours
```

## Maintenance Features

``` text
days_since_maintenance
maintenance_count
maintenance_cost_avg
failure_count
previous_failure
```

## Business Features

``` text
daily_sales
weekly_sales
monthly_sales
revenue
average_order_value
product_demand
sales_growth
```

------------------------------------------------------------------------

# 18. Model-to-Dataset Mapping

  Model                Dataset              Output
  -------------------- -------------------- -------------------------
  Isolation Forest     MetroPT-3            Anomaly Score
  XGBoost Classifier   AI4I                 Failure Probability
  XGBoost Regressor    M5 / Business Data   Demand Forecast
  LSTM                 MetroPT-3 / M5       Time-Series Forecast
  SHAP                 AI4I / XGBoost       Feature Explanation
  RAG                  PDF Documents        Document-based Answer
  LLM                  Model + SHAP + RAG   Operational Explanation

------------------------------------------------------------------------

# 19. Minimum Dataset Requirements

If time becomes limited, the project can still be completed with:

``` text
1. MetroPT-3
2. AI4I 2020
3. UCI Online Retail
4. 5–10 synthetic operational PDFs
```

M5 and NASA C-MAPSS can then be added as advanced extensions.

------------------------------------------------------------------------

# 20. Dataset Download Checklist

## Required

-   [ ] MetroPT-3 CSV
-   [ ] AI4I 2020 CSV
-   [ ] UCI Online Retail
-   [ ] Synthetic maintenance documents
-   [ ] Synthetic machine manuals
-   [ ] Synthetic inspection reports

## Recommended

-   [ ] M5 Forecasting subset
-   [ ] NASA C-MAPSS subset

## Optional

-   [ ] SECOM
-   [ ] Gas Turbine
-   [ ] MIMII

------------------------------------------------------------------------

# 21. Data Licensing & Attribution

Before publishing or redistributing datasets, check the individual
dataset's current license and terms.

Known terms for the recommended sources include:

-   UCI AI4I 2020 --- CC BY 4.0
-   UCI Online Retail --- CC BY 4.0
-   UCI MetroPT-3 --- CC BY 4.0
-   UCI Gas Turbine --- CC BY 4.0
-   Kaggle M5 --- subject to competition/data terms
-   NASA datasets --- check the specific dataset's current access and
    usage terms
-   MIMII --- check the current Zenodo license and usage conditions

For GitHub, keep large raw datasets out of the repository where
appropriate. Provide official source links and a data-preparation script
instead.

------------------------------------------------------------------------

# 22. Recommended Repository Data Structure

``` text
data/
│
├── raw/
│   ├── metropt/
│   ├── ai4i/
│   ├── retail/
│   └── m5/
│
├── processed/
│   ├── sensors.parquet
│   ├── machines.parquet
│   ├── maintenance.parquet
│   ├── production.parquet
│   └── business_transactions.parquet
│
├── external/
│   └── cmapss/
│
└── README.md
```

Documents:

``` text
documents/
│
├── manuals/
├── maintenance/
├── inspection/
├── specifications/
└── README.md
```

------------------------------------------------------------------------

# 23. Official Source Links --- Quick Reference

  ---------------------------------------------------------------------------------------------------------------------------------------------------
  Dataset / Source                    Official Link
  ----------------------------------- ---------------------------------------------------------------------------------------------------------------
  UCI AI4I 2020 Predictive            https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset
  Maintenance                         

  UCI MetroPT-3                       https://archive.ics.uci.edu/dataset/791/metropt3%2Bdataset

  UCI Online Retail                   https://archive.ics.uci.edu/dataset/352/online%2Bretail

  UCI Online Retail II                https://archive.ics.uci.edu/dataset/502/online%2Bretail

  Kaggle M5 Forecasting               https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

  NASA C-MAPSS                        https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

  NASA PCoE Repository                https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

  UCI SECOM                           https://archive.ics.uci.edu/dataset/179/secom

  UCI Gas Turbine                     https://archive.ics.uci.edu/dataset/551/gas%2Bturbine%2Bco%2Band%2Bnox%2Bemission%2Bdata%2Bset

  MIMII on Zenodo                     https://zenodo.org/records/3384388

  PHM Society Data Repository         https://data.phmsociety.org/phm-society-data-repository/
  ---------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 24. Final Recommended Dataset Architecture

``` text
                         DATA SOURCES
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
   SENSOR DATA           BUSINESS DATA          DOCUMENTS
       │                      │                      │
   MetroPT-3             Online Retail          PDF Docs
   AI4I 2020              M5 Forecasting        Manuals
   NASA C-MAPSS           Production Data       SOPs
       │                      │                  Reports
       └──────────────────────┼──────────────────────┘
                              ▼
                       DATA ENGINEERING
                              │
                    Cleaning + Integration
                              │
                       Feature Engineering
                              ▼
                         ML ENGINE
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         Forecasting      Anomaly         Prediction
                          Detection
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                        AI REASONING
                              │
                       SHAP + LLM + RAG
                              ▼
                     BUSINESS INSIGHTS
                              │
                    Alerts + Recommendations
                              ▼
                       WEB APPLICATION
```

------------------------------------------------------------------------

# 25. Final Recommendation

For the **3-week development + 1-week testing** schedule, use:

``` text
CORE

MetroPT-3
    ↓
Sensor Analytics + Anomaly Detection

AI4I 2020
    ↓
Failure Prediction + SHAP

UCI Online Retail
    ↓
Business Analytics

M5
    ↓
Demand Forecasting

Synthetic Operational PDFs
    ↓
RAG + AI Reasoning
```

Keep:

``` text
NASA C-MAPSS
SECOM
Gas Turbine
MIMII
```

as optional extensions.

The target should be **4 core public datasets + a controlled document
corpus + one complete end-to-end pipeline**, rather than a collection of
disconnected ML experiments.
