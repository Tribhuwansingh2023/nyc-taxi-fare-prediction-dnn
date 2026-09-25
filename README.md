# 🚖 New York City Taxi Fare Prediction Using Deep Feedforward Neural Networks

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An end-to-end deep learning engineering pipeline designed to predict NYC Yellow Taxi fares from raw telemetry data. This project implements a **33-feature geodesic, spatial, and cyclical temporal transformation pipeline**, rigorously audits anomalies on a **1,000,000-record dataset**, trains a **PyTorch Deep Feedforward Neural Network with Huber Loss ($\delta=1.0$)**, and deploys an interactive **Streamlit web application** backed by an automated 4-scenario verification test harness.

---

## 📌 Table of Contents
- [Academic Information](#-academic-information)
- [Project Team Registry](#-project-team-registry)
- [Problem Overview & Engineering Pipeline](#-problem-overview--engineering-pipeline)
- [Repository Structure](#-repository-structure)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Quickstart & Execution Guide](#-quickstart--execution-guide)
  - [1. Interactive Streamlit Web Application](#1-launch-the-streamlit-web-app)
  - [2. Automated Deployment Verification Suite](#2-run-automated-test-suite)
  - [3. Full Academic Report Generator (Word & PDF)](#3-build-academic-lab-records)
  - [4. Recompile the Jupyter Notebook](#4-compile-the-end-to-end-notebook)
- [Key Features & Feature Engineering (33 Features)](#-feature-engineering-pipeline-33-features)
- [Deep Neural Network Architecture](#-deep-neural-network-architecture)
- [Experimental Benchmarks & Results](#-experimental-benchmarks--model-comparison)
- [Visualizations Gallery](#-visualizations-gallery)
- [License](#-license)

---

## 🎓 Academic Information

- **Course:** Machine Learning Projects with Python (`CSE 4192`)
- **Department:** Department of Computer Science & Engineering | Centre for Artificial Intelligence & Machine Learning
- **Institution:** Siksha 'O' Anusandhan (Deemed to be University), ITER, Bhubaneswar
- **Academic Session:** 2026– 2027 | **Admission Batch:** 2023 – 2027
- **Course Faculty:** Dr. Gyana Ranjan Patra

---

## 👥 Project Team Registry

| Sl. No. | Student Name | Registration No. | Core Technical Responsibilities | Contribution |
| :---: | :--- | :---: | :--- | :---: |
| 1 | **Tribhuwan Singh** | `2341019538` | Deep Feedforward Neural Network Architecture, Huber Loss Formulation & Streamlit Deployment | **25%** |
| 2 | **Surajit Sahoo** | `2341019165` | Exploratory Data Analysis, Geolocation Spatial Mapping & Ridership Heatmaps | **25%** |
| 3 | **Anwesha Srichandan** | `2341019594` | Data Quality Auditing, Outlier Cleansing & 33-Feature Geodesic Pipeline | **25%** |
| 4 | **Priti Rani Maity** | `2341013065` | Hyperparameter Search Optimization, Baseline Benchmarking & Academic Report Authoring | **25%** |

---

## 🧠 Problem Overview & Engineering Pipeline

Predicting taxi fares in New York City is inherently non-linear and subject to severe geospatial clustering, rush-hour surcharges, airport flat-rate regimes, and extreme telemetry noise (GPS bounce, negative fares, coordinates outside NYC bounding boxes). 

This project provides an end-to-end production-grade solution:
1. **Data Ingestion & Bounding Box Filtering:** Strict coordinate bounds (`[-74.5, -72.8]` lon, `[40.5, 41.8]` lat), positive passenger counts (`1 ≤ n ≤ 6`), and fare filtering (`$2.50 ≤ fare ≤ $500.00`).
2. **Geodesic & Temporal Feature Engineering:** Calculates Haversine, Manhattan ($L_1$), and Bearing features along with direct Euclidean distances to six major regional epicenters (JFK, LGA, EWR, Times Square, Grand Central, Midtown Manhattan), and encodes cyclical timestamps ($\sin/\cos$ for hour, day of week, and month).
3. **Leak-Free Transformation:** Uses a fitted `StandardScaler` calibrated exclusively on the training split ($70\%$), preventing data contamination into validation ($15\%$) and test ($15\%$) sets.
4. **Deep Neural Network Optimization:** A 4-layer PyTorch feedforward architecture with Batch Normalization, Dropout ($0.20, 0.10$), ReLU activations, and Huber Loss ($\delta=1.0$) trained with Adam and `ReduceLROnPlateau`.
5. **Interactive UI & Test Harness:** Streamlit application providing real-time fare inference, interactive map visualization, and a 4-scenario deployment verification test harness.

---

## 📁 Repository Structure

### GitHub Repository Layout (`origin/main`)
```text
nyc_taxi_fare_dnn_assignment/
├── app/
│   ├── app.py                             # Interactive Streamlit Web Application (Fare Card, Map, Audit)
│   └── test_deployment.py                 # Automated 4-Scenario Deployment Test Harness
│
├── notebooks/
│   └── NYC_Taxi_Fare_Prediction_DNN.ipynb # Complete 23-Section Mathematical & Python Notebook
│
├── results/
│   ├── baseline_comparison.csv            # OLS, Ridge, Random Forest, LightGBM, MLP Metrics
│   ├── dnn_performance_summary.json       # PyTorch DNN Checkpoint, Parameters & Metric Log
│   ├── eda_summary.json                   # 1,000,000-Row Statistical & Anomaly Audit Summary
│   ├── hyperparameter_tuning_results.csv  # 8-Trial Systematic Hyperparameter Search Log
│   └── model_comparison.csv               # Unified Multi-Model Benchmark Comparison Table
│
├── saved_models/
│   └── feature_metadata.json              # 33 Feature Column Definitions & Input Schema
│
├── src/
│   ├── baseline_models.py                 # Classical ML Training (OLS, Ridge, RF, LightGBM)
│   ├── data_loader.py                     # Raw Telemetry Parsing & Geodetic Verification
│   ├── dnn_model.py                       # PyTorch TaxiFareDNN Class, Huber Loss & Training Engine
│   ├── eda.py                             # 10 High-Resolution Exploratory Visualizations Generator
│   ├── feature_engineering.py             # 33 Geodesic, Spatial, and Temporal Transforms Pipeline
│   ├── hyperparameter_tuning.py           # Multi-Trial Neural Hyperparameter Search Harness
│   ├── model_comparison.py                # Comparative Multi-Metric Benchmark Evaluator
│   └── preprocessing.py                   # Bounding Box Filtering & Leak-Free Splitting
│
├── visualizations/                        # 15 Publication-Grade Exploratory & Loss Visualizations
│   ├── 01_fare_distribution.png
│   ├── 02_passenger_count_distribution.png
│   ├── 03_nyc_pickup_density.png
│   ├── 04_nyc_dropoff_density.png
│   ├── 05_pickup_vs_dropoff_geo.png
│   ├── 06_hourly_ridership_patterns.png
│   ├── 07_day_of_week_ridership.png
│   ├── 08_hourly_day_heatmap.png
│   ├── 09_distance_vs_fare.png
│   ├── 10_feature_correlation_heatmap.png
│   ├── 11_dnn_training_validation_loss.png
│   ├── 12_loss_functions_comparison.png
│   ├── 13_actual_vs_predicted_fare.png
│   ├── 14_residual_distribution.png
│   └── 15_model_comparison_bar.png
│
├── .gitignore                             # Git configuration ignoring heavy local assets & data
├── build_complete_lab_record.py           # Master Script Generating Word (.docx) & PDF Reports
├── LICENSE                                # MIT Open-Source License
├── make_notebook.py                       # Python Notebook Compiler Script
├── README.md                              # Comprehensive Project Documentation
└── requirements.txt                       # Project Python Dependencies
```

> [!NOTE]
> Heavy artifacts—including raw Kaggle datasets (`dataset/`, `data/`), binary model weights (`*.pt`, `*.pkl`), academic report documents (`*.docx`, `*.pdf`), and local UI screenshots—are ignored via `.gitignore` to keep the repository lightweight and standards-compliant.

---

## 🛠️ Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Tribhuwansingh2023/nyc-taxi-fare-prediction-dnn.git
cd nyc-taxi-fare-prediction-dnn
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Quickstart & Execution Guide

### 1. Launch the Streamlit Web App
Launch the interactive web application for real-time inference, coordinate lookup, and 3D geospatial route visualization:
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run Automated Test Suite
Execute the automated 4-scenario deployment verification suite to validate inference accuracy against real-world test cases:
```bash
python app/test_deployment.py
```
**Sample Output:**
```text
================================================================================
RUNNING AUTOMATED DEPLOYMENT AND INFERENCE VERIFICATION TESTS
================================================================================
[TestSetup] Scaler and PyTorch DNN successfully loaded into memory.

Evaluating Case 1: Standard Short Manhattan Trip (Times Square -> Grand Central)...
  Distance: 0.91 km | Predicted Fare: $8.26 (Expected: $4.00 - $15.00) -> PASSED [OK]

Evaluating Case 2: Long Airport Journey (JFK Terminal 4 -> Times Square)...
  Distance: 21.77 km | Predicted Fare: $59.45 (Expected: $35.00 - $75.00) -> PASSED [OK]

Evaluating Case 3: Borderline Ultra-Short Trip (200m hop)...
  Distance: 0.21 km | Predicted Fare: $5.91 (Expected: $2.50 - $12.00) -> PASSED [OK]

Evaluating Case 4: LaGuardia Airport to Lower Manhattan / Wall St...
  Distance: 13.74 km | Predicted Fare: $37.46 (Expected: $25.00 - $55.00) -> PASSED [OK]

================================================================================
DEPLOYMENT TEST SUMMARY: 4 / 4 Test Cases Passed.
================================================================================
```

### 3. Build Academic Lab Records
Compile the university laboratory record report documents (`.docx` Word and `.pdf` ReportLab):
```bash
python build_complete_lab_record.py
```

### 4. Compile the End-to-End Notebook
Regenerate `notebooks/NYC_Taxi_Fare_Prediction_DNN.ipynb` containing all 23 structured experiment cells:
```bash
python make_notebook.py
```

---

## 🔬 Feature Engineering Pipeline (33 Features)

| Feature Category | Features Count | Engineered Columns | Mathematical Rationale |
| :--- | :---: | :--- | :--- |
| **Spatial Coordinates** | 4 | `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude` | Raw coordinate telemetry within NYC bounding box. |
| **Geodesic Distances** | 3 | `haversine_distance`, `manhattan_distance`, `bearing` | Great-circle distance, Manhattan $L_1$ grid distance, and compass heading in degrees. |
| **Coordinate Displacements** | 2 | `delta_lon`, `delta_lat` | Absolute coordinate displacement vectors. |
| **Airport Proximity** | 6 | `pickup_JFK_dist`, `dropoff_JFK_dist`, `pickup_LGA_dist`, `dropoff_LGA_dist`, `pickup_EWR_dist`, `dropoff_EWR_dist` | Direct Euclidean proximity to major airports with flat-rate or toll fee structures. |
| **Landmark Proximity** | 6 | `pickup_TimesSquare_dist`, `dropoff_TimesSquare_dist`, `pickup_GrandCentral_dist`, `dropoff_GrandCentral_dist`, `pickup_Midtown_dist`, `dropoff_Midtown_dist` | Distance to high-density commercial centers and transit hubs. |
| **Capacity & Demand** | 2 | `passenger_count`, `dist_per_passenger` | Passenger loading and passenger-normalized journey length. |
| **Calendar & Time** | 4 | `hour`, `day`, `day_of_week`, `month`, `year` | Temporal breakdown capturing diurnal and seasonal trends. |
| **Rush Hour & Weekend** | 2 | `is_weekend`, `is_rush_hour` | Binary indicators for peak surcharge windows (weekdays 4 PM–8 PM) and weekends. |
| **Cyclical Encoding** | 6 | `sin_hour`, `cos_hour`, `sin_dow`, `cos_dow`, `sin_month`, `cos_month` | Continuous sine/cosine periodicity transformations for time continuity (23:59 $\leftrightarrow$ 00:00). |

---

## 🏗️ Deep Neural Network Architecture

```
                                 ┌───────────────────────────┐
                                 │    Input Layer (33 Dim)   │
                                 └─────────────┬─────────────┘
                                               │
                                 ┌─────────────▼─────────────┐
                                 │ Linear(33 -> 128)         │
                                 │ BatchNorm1d(128)          │
                                 │ ReLU() + Dropout(p=0.20)  │
                                 └─────────────┬─────────────┘
                                               │
                                 ┌─────────────▼─────────────┐
                                 │ Linear(128 -> 64)         │
                                 │ BatchNorm1d(64)           │
                                 │ ReLU() + Dropout(p=0.10)  │
                                 └─────────────┬─────────────┘
                                               │
                                 ┌─────────────▼─────────────┐
                                 │ Linear(64 -> 32)          │
                                 │ ReLU()                    │
                                 └─────────────┬─────────────┘
                                               │
                                 ┌─────────────▼─────────────┐
                                 │ Linear(32 -> 1)           │
                                 │ Output: Predicted Fare ($)│
                                 └───────────────────────────┘
```

### Mathematical Formulation
- **Loss Function:** Huber Loss with threshold parameter $\delta = 1.0$:
  $$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\ \delta \cdot \left(|y - \hat{y}| - \frac{1}{2}\delta\right) & \text{otherwise} \end{cases}$$
  *Why Huber Loss?* Combines the smooth gradient convergence of MSE for small residuals with the outlier robustness of MAE for extreme fare surges.
- **Optimizer:** Adam ($\eta = 0.001$, $\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 10^{-5}$).
- **Learning Rate Scheduler:** `ReduceLROnPlateau(mode='min', factor=0.5, patience=3)`.

---

## 📊 Experimental Benchmarks & Model Comparison

All models were evaluated using identical leak-free train/validation/test partitions ($70\% / 15\% / 15\%$):

| Model Architecture | Training Time (s) | Test MAE ($) | Test MSE ($²) | Test RMSE ($) | Test R² Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (OLS)** | 3.32 s | $1.99 | 13.47 | $3.67 | 0.8441 |
| **Ridge Regression (L2)** | 0.58 s | $1.99 | 13.47 | $3.67 | 0.8439 |
| **Random Forest Regressor** | 9434.22 s | $1.61 | 9.80 | $3.13 | 0.8866 |
| **LightGBM Regressor** | 19.32 s | **$1.50** | **8.94** | **$2.99** | **0.8963** |
| **MLPRegressor (Scikit-Learn)** | 705.97 s | $1.52 | 9.36 | $3.06 | 0.8917 |
| **Deep Neural Network (PyTorch)** | 120.00 s | **$1.57** | 10.89 | $3.30 | **0.8734** |

### Loss Function Ablation Study (PyTorch DNN)
| Objective Function | Validation MAE ($) | Validation RMSE ($) | Validation R² | Optimization Characteristic |
| :--- | :---: | :---: | :---: | :--- |
| **MSE Loss ($L_2$)** | $1.62 | $3.25 | 0.878 | Sensitive to large outlier fares; rapid initial gradient descent. |
| **MAE Loss ($L_1$)** | $1.55 | $3.28 | 0.876 | Highly robust to anomalies, but suffers from oscillating gradients near zero. |
| **Huber Loss ($\delta = 1.0$)** | **$1.57** | **$3.31** | **0.874** | **Best balanced trade-off:** smooth quadratic convergence with linear penalty for outliers. |

---

## 📈 Visualizations Gallery

The complete set of 15 high-resolution figures generated by the analysis pipeline:

| Figure Preview | Description |
| :---: | :--- |
| ![Fare Distribution](visualizations/01_fare_distribution.png) | **01. Fare Distribution:** Log-scaled frequency of NYC taxi fares illustrating peak ridership at base rates. |
| ![Pickup Density](visualizations/03_nyc_pickup_density.png) | **03. Pickup Density:** Heatmap showing heavy clustering across Midtown Manhattan and JFK/LGA airports. |
| ![Pickup vs Dropoff Geo](visualizations/05_pickup_vs_dropoff_geo.png) | **05. Spatial Telemetry:** Comparative 2D geospatial distribution of ride origins vs destinations. |
| ![Hourly Day Heatmap](visualizations/08_hourly_day_heatmap.png) | **08. Ridership Heatmap:** Cross-tabulation of hourly volume across days of the week. |
| ![DNN Loss Curves](visualizations/11_dnn_training_validation_loss.png) | **11. Training Dynamics:** Huber Loss progression across 25 epochs with adaptive learning rate reduction. |
| ![Actual vs Predicted](visualizations/13_actual_vs_predicted_fare.png) | **13. Calibration Scatter:** Actual vs. Predicted fare correlation on the untouched test partition. |
| ![Model Comparison](visualizations/15_model_comparison_bar.png) | **15. Benchmark Comparison:** Bar chart comparing MAE and RMSE metrics across all 6 evaluated models. |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this codebase with proper attribution.
