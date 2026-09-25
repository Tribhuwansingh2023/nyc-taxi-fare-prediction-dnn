# Lab Assignment 02: New York City Taxi Fare Prediction Using Deep Feedforward Neural Networks

**Course:** Machine Learning Projects with Python (`CSE 4192`)  
**Department:** Department of Computer Science & Engineering | Centre for Artificial Intelligence & Machine Learning  
**Institution:** Siksha 'O' Anusandhan (Deemed to be University), ITER, Bhubaneswar  
**Session:** 2025 – 2026 | **Admission Batch:** 2023 – 2027  
**Course Faculty:** Dr. Gyana Ranjan Patra  

---

## 👥 Project Team Registry

| Sl. No. | Student Name | Registration Number | Core Technical Responsibilities | Contribution |
| :---: | :--- | :---: | :--- | :---: |
| 1 | **Tribhuwan Singh** | `2341019538` | Deep Feedforward Neural Network Architecture, Huber Loss Formulation & Streamlit Deployment | **25%** |
| 2 | **Surajit Sahoo** | `2341019165` | Exploratory Data Analysis, Geolocation Spatial Mapping & Ridership Heatmaps | **25%** |
| 3 | **Anwesha Srichandan** | `2341019594` | Data Quality Auditing, Outlier Cleansing & 33-Feature Geodesic Pipeline | **25%** |
| 4 | **Priti Rani Maity** | `2341013065` | Hyperparameter Search Optimization, Baseline Benchmarking & Academic Report Authoring | **25%** |

---

## 📁 Repository & Deliverables Structure

```text
nyc_taxi_fare_dnn_assignment/
├── app/
│   ├── app.py                             # Interactive Streamlit Web Deployment Application
│   └── test_deployment.py                 # Automated 4-Scenario Deployment Verification Test Suite
├── data/
│   ├── raw/
│   │   └── train_sample_100k.csv          # 100,000 Sampled Real NYC Yellow Taxi Records
│   └── processed/
│       ├── train.csv                      # 70% Train Partition (68,166 Clean Records)
│       ├── val.csv                        # 15% Validation Partition (14,607 Clean Records)
│       └── test.csv                       # 15% Untouched Test Partition (14,608 Clean Records)
├── notebooks/
│   └── NYC_Taxi_Fare_Prediction_DNN.ipynb # End-to-End Mathematical & Python Jupyter Notebook
├── results/
│   ├── baseline_comparison.csv            # Performance Metrics for Initial Baselines
│   ├── dnn_performance_summary.json       # PyTorch Deep Neural Network Parameter & Metric Log
│   ├── eda_summary.json                   # Statistical & Anomaly Auditing Quantifications
│   ├── hyperparameter_tuning_results.csv  # 8-Trial Systematic Hyperparameter Search Log
│   └── model_comparison.csv               # Unified Multi-Model Benchmark Comparison Table
├── saved_models/
│   ├── baseline_models.pkl                # Trained LightGBM & Linear Regression Pipelines
│   ├── feature_metadata.json              # Schema & 33 Feature Column Definitions
│   ├── taxi_fare_dnn.pt                   # PyTorch Checkpoint of Optimized Deep Feedforward Neural Net
│   └── taxi_fare_scaler.pkl               # Fitted Leak-Free StandardScaler Transformation
├── src/
│   ├── baseline_models.py                 # Classical ML Training (OLS, Ridge, Random Forest, LightGBM)
│   ├── data_loader.py                     # Raw Telemetry Parsing & Geodetic Verification
│   ├── dnn_model.py                       # PyTorch TaxiFareDNN Class, Huber Loss & Training Engine
│   ├── eda.py                             # 10 High-Resolution Exploratory Visualizations Generator
│   ├── feature_engineering.py             # 33 Geodesic, Spatial, and Temporal Transforms Pipeline
│   ├── hyperparameter_tuning.py           # Multi-Trial Neural Hyperparameter Search Harness
│   ├── model_comparison.py                # Comparative Multi-Metric Benchmark Evaluator
│   └── preprocessing.py                   # Bounding Box Filtering & Leak-Free Splitting
├── visualizations/
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
│   ├── 15_model_comparison_bar.png
│   ├── ui_fare_prediction_screenshot.png  # Live Deployment Fare Card & 3D Map View
│   ├── ui_test_suite_screenshot.png       # Live In-App Verification Suite Passing All Tests
│   ├── ui_academic_registry_screenshot.png# University & Team Registry Tab
│   └── ui_performance_benchmarks_screenshot.png
├── build_complete_lab_record.py           # Master Script Generating Word & PDF Reports
├── make_notebook.py                       # Jupyter Notebook Compiler Script
├── Laboratory_Record_CSE4192.docx         # Official 23-Section Academic Lab Record (Word)
├── Laboratory_Record_CSE4192.pdf          # Official Publication-Grade Lab Record (ReportLab PDF)
├── soa_logo.png                           # University Official Crest
└── README.md                              # Comprehensive Project Documentation
```

---

## 🚀 Quickstart & Deployment Instructions

### 1. Launch the Streamlit Web Application
Ensure your active environment has the required packages installed (`streamlit`, `torch`, `scikit-learn`, `pandas`, `numpy`, `pydeck`, `joblib`):
```bash
cd nyc_taxi_fare_dnn_assignment
python -m streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run the Automated Deployment Verification Suite
To execute the test harness verifying model checkpoints and inference endpoints:
```bash
python app/test_deployment.py
```
Output:
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

### 3. Rebuild the Academic Laboratory Reports (Word & PDF)
To recompile the official university laboratory record documents:
```bash
python build_complete_lab_record.py
```

---

## 📊 Summary of Final Experimental Results

| Model Architecture | Training Time (s) | Test MAE ($) | Test MSE ($²) | Test RMSE ($) | Test R² Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (OLS)** | 3.32 s | $1.990 | 13.470 | $3.670 | 0.8441 |
| **Ridge Regression (L2)** | 0.58 s | $1.990 | 13.470 | $3.670 | 0.8439 |
| **Random Forest Regressor** | 9434.22 s | $1.610 | 9.800 | $3.130 | 0.8866 |
| **LightGBM Regressor** | 19.32 s | **$1.500** | **8.940** | **$2.990** | **0.8963** |
| **MLPRegressor (Scikit-Learn)** | 705.97 s | $1.520 | 9.360 | $3.060 | 0.8917 |
| **Deep Neural Network (PyTorch)** | 120.00 s | **$1.570** | 10.890 | $3.300 | **0.8734** |

**Optimal Neural Network Topology:**
$$\text{Input}(33) \longrightarrow \text{Dense}(128) + \text{BatchNorm} + \text{ReLU} + \text{Dropout}(0.20) \longrightarrow \text{Dense}(64) + \text{BatchNorm} + \text{ReLU} + \text{Dropout}(0.10) \longrightarrow \text{Dense}(32) + \text{ReLU} \longrightarrow \text{Linear}(1)$$
- **Loss Function:** Huber Loss with threshold parameter $\delta = 1.0$ (combines MSE convergence precision for small errors with MAE outlier resilience).
- **Optimizer:** Adam ($\eta = 0.001$, weight decay $= 10^{-5}$) with `ReduceLROnPlateau` adaptive learning rate scheduler.
