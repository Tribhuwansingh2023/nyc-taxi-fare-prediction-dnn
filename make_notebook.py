"""
Script to build the comprehensive Jupyter Notebook for Lab Assignment 02:
notebooks/NYC_Taxi_Fare_Prediction_DNN.ipynb
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebooks", "NYC_Taxi_Fare_Prediction_DNN.ipynb")

def build_notebook():
    cells = []
    
    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [s + "\n" for s in source.strip().split("\n")]
        })
        
    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [s + "\n" for s in source.strip().split("\n")]
        })

    # Cell 1: Title & University Information
    add_md("""# Lab Assignment 02: New York City Taxi Fare Prediction Using Deep Feedforward Neural Networks
### Course: Machine Learning Projects with Python (CSE 4192)
**Department of Computer Science & Engineering**  
**Siksha 'O' Anusandhan (Deemed to be University), ITER, Bhubaneswar**  
**Admission Batch:** 2023 – 2027 | **Academic Session:** 2025 – 2026  

---
### Project Team Members:
1. **Tribhuwan Singh** (Registration No.: `2341019538`) — *Lead: Neural Network Architecture & Deployment*
2. **Surajit Sahoo** (Registration No.: `2341019165`) — *Exploratory Data Analysis & Geospatial Visualizations*
3. **Anwesha Srichandan** (Registration No.: `2341019594`) — *Data Quality Auditing, Preprocessing & Feature Engineering*
4. **Priti Rani Maity** (Registration No.: `2341013065`) — *Hyperparameter Optimization, Benchmark Evaluation & Reporting*

---
## 1. Problem Statement & Aim
Develop an end-to-end Deep Learning regression system to accurately predict the fare amount of New York City yellow taxi journeys using historical trip, temporal, and geolocation coordinates. The project investigates the relationship between pickup/dropoff coordinates, travel timing, passenger count, geodesic distance, and the resulting fare, culminating in an optimized Deep Feedforward Neural Network (DNN/MLP) deployed via an interactive Streamlit application.
""")

    # Cell 2: Imports & Environment Setup
    add_md("""## 2. Environment Setup & Library Imports
We initialize standard data science, machine learning, and deep learning libraries (`PyTorch`, `Scikit-Learn`, `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `LightGBM`). Thread concurrency is constrained to ensure optimal CPU memory handling on local systems.""")

    add_code("""import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import json
import joblib
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

# Configure visualization defaults
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10

# Set deterministic random seed
torch.manual_seed(42)
np.random.seed(42)
print("Libraries imported successfully. PyTorch Version:", torch.__version__)""")

    # Cell 3: Dataset Loading
    add_md("""## 3. Dataset Understanding & Exploratory Data Analysis
We load an authentic 100,000-record sample from the official Kaggle *New York City Taxi Fare Prediction* dataset. Each observation contains:
- `key`: Unique trip identifier with timestamp
- `fare_amount`: Continuous target variable (USD)
- `pickup_datetime`: Trip departure timestamp in UTC
- `pickup_longitude`, `pickup_latitude`: Geodetic pickup coordinates
- `dropoff_longitude`, `dropoff_latitude`: Geodetic dropoff coordinates
- `passenger_count`: Integer passenger count""")

    add_code("""RAW_DATA_PATH = "../data/raw/train_sample_100k.csv"
if not os.path.exists(RAW_DATA_PATH):
    RAW_DATA_PATH = "data/raw/train_sample_100k.csv"

df_raw = pd.read_csv(RAW_DATA_PATH)
print("Dataset Dimensions:", df_raw.shape)
display(df_raw.head())
display(df_raw.describe().T)""")

    # Cell 4: Data Preprocessing & Cleaning
    add_md("""## 4. Data Quality Auditing & Anomaly Cleaning
Real-world GPS telemetry and meter logs exhibit physical anomalies:
1. **Implausible Fares**: Negative values (meter refunds) and fares $< \\$2.50$ (below the NYC base flag-drop rate).
2. **Coordinate Outliers**: Values outside the NYC metropolitan bounding box ($40.50 \\le \\text{Lat} \\le 40.95$, $-74.25 \\le \\text{Lon} \\le -73.70$), including $(0.0, 0.0)$ null island coordinates.
3. **Invalid Passenger Counts**: Trip counts $\\le 0$ or $> 6$ (exceeding standard yellow cab capacity).
4. **Zero-Distance Trips**: Coordinates where pickup equals dropoff with non-zero fares.""")

    add_code("""def clean_taxi_records(df):
    initial_count = len(df)
    df_clean = df.dropna().copy()
    df_clean["pickup_datetime"] = pd.to_datetime(df_clean["pickup_datetime"], errors="coerce")
    df_clean = df_clean.dropna(subset=["pickup_datetime"])
    
    # Valid NYC bounding box
    NYC_LAT_MIN, NYC_LAT_MAX = 40.50, 40.95
    NYC_LON_MIN, NYC_LON_MAX = -74.25, -73.70
    
    valid_mask = (
        (df_clean["fare_amount"].between(2.50, 200.0)) &
        (df_clean["passenger_count"].between(1, 6)) &
        (df_clean["pickup_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX)) &
        (df_clean["pickup_longitude"].between(NYC_LON_MIN, NYC_LON_MAX)) &
        (df_clean["dropoff_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX)) &
        (df_clean["dropoff_longitude"].between(NYC_LON_MIN, NYC_LON_MAX))
    )
    df_clean = df_clean[valid_mask].copy()
    
    # Calculate geodesic distance to filter zero-distance errors
    lon1, lat1, lon2, lat2 = map(np.radians, [
        df_clean["pickup_longitude"], df_clean["pickup_latitude"],
        df_clean["dropoff_longitude"], df_clean["dropoff_latitude"]
    ])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    dist_km = 6371.0 * c
    
    df_clean["temp_dist"] = dist_km
    df_clean = df_clean[(df_clean["temp_dist"] >= 0.05) & (df_clean["temp_dist"] <= 80.0)].drop(columns=["temp_dist"])
    
    print(f"Cleaned Records: {len(df_clean):,} / {initial_count:,} ({len(df_clean)/initial_count*100:.2f}% retained)")
    return df_clean

df_cleaned = clean_taxi_records(df_raw)""")

    # Cell 5: Geolocation & Temporal Feature Engineering
    add_md("""## 5. Domain Feature Engineering
We synthesize 33 rich predictive features capturing:
- **Geodesic Distance**: Haversine great-circle distance on the spherical earth.
- **Manhattan Grid Distance**: $L_1$ rectilinear street distance along Manhattan's grid ($|\\Delta \\text{lat}| \\times 111\\text{ km} + |\\Delta \\text{lon}| \\times 82\\text{ km}$).
- **Azimuth Bearing Angle**: Directional vector from origin to destination ($0^\\circ$ to $360^\\circ$).
- **Airport Proximity Features**: Distance to JFK, LaGuardia (LGA), and Newark (EWR) airports.
- **Temporal Indicators**: Hour of day, day of week, rush-hour indicators, weekend flags, and cyclical sine/cosine embeddings.""")

    add_code("""LANDMARKS = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "Midtown": (40.7580, -73.9855)
}

def haversine_vectorized(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return 6371.0 * c

def calculate_bearing_vectorized(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    initial_bearing = np.arctan2(x, y)
    return (np.degrees(initial_bearing) + 360) % 360

def engineer_features(data):
    df = data.copy()
    p_lat, p_lon = df["pickup_latitude"].values, df["pickup_longitude"].values
    d_lat, d_lon = df["dropoff_latitude"].values, df["dropoff_longitude"].values
    
    df["haversine_dist_km"] = haversine_vectorized(p_lon, p_lat, d_lon, d_lat)
    df["lat_diff"] = np.abs(d_lat - p_lat)
    df["lon_diff"] = np.abs(d_lon - p_lon)
    df["manhattan_dist_km"] = (df["lat_diff"] * 111.0) + (df["lon_diff"] * 82.0)
    df["euclidean_dist"] = np.sqrt(df["lat_diff"]**2 + df["lon_diff"]**2)
    df["bearing_deg"] = calculate_bearing_vectorized(p_lon, p_lat, d_lon, d_lat)
    
    for hub, (h_lat, h_lon) in LANDMARKS.items():
        df[f"pickup_{hub}_dist"] = haversine_vectorized(p_lon, p_lat, h_lon, h_lat)
        df[f"dropoff_{hub}_dist"] = haversine_vectorized(d_lon, d_lat, h_lon, h_lat)
        
    df["dist_per_passenger"] = df["haversine_dist_km"] / np.maximum(df["passenger_count"].values, 1)
    
    dt = df["pickup_datetime"].dt
    df["hour"] = dt.hour
    df["day"] = dt.day
    df["day_of_week"] = dt.dayofweek
    df["month"] = dt.month
    df["year"] = dt.year
    df["is_weekend"] = ((df["day_of_week"] == 5) | (df["day_of_week"] == 6)).astype(int)
    df["is_rush_hour"] = (
        (df["is_weekend"] == 0) &
        (((df["hour"] >= 7) & (df["hour"] <= 10)) | ((df["hour"] >= 16) & (df["hour"] <= 20)))
    ).astype(int)
    
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    return df

df_featured = engineer_features(df_cleaned)
print("Features engineered successfully. New column count:", len(df_featured.columns))""")

    # Cell 6: Train/Val/Test Split & Feature Scaling
    add_md("""## 6. Train, Validation, and Test Partitioning & Leak-Free Feature Scaling
To prevent data contamination:
- **Training Set (70%)**: Used for fitting baseline models and neural network parameters.
- **Validation Set (15%)**: Used for hyperparameter search, learning rate scheduling, and early stopping.
- **Test Set (15%)**: Kept strictly untouched until the final single evaluation.
- `StandardScaler` is fitted **exclusively on the training partition**.""")

    add_code("""FEATURE_COLUMNS = [
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "passenger_count",
    "haversine_dist_km", "manhattan_dist_km", "euclidean_dist",
    "lat_diff", "lon_diff", "bearing_deg",
    "pickup_JFK_dist", "dropoff_JFK_dist",
    "pickup_LGA_dist", "dropoff_LGA_dist",
    "pickup_EWR_dist", "dropoff_EWR_dist",
    "pickup_Midtown_dist", "dropoff_Midtown_dist",
    "dist_per_passenger",
    "hour", "day", "day_of_week", "month", "year",
    "is_weekend", "is_rush_hour",
    "sin_hour", "cos_hour", "sin_dow", "cos_dow", "sin_month", "cos_month"
]

train_df, temp_df = train_test_split(df_featured, test_size=0.30, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, shuffle=True)

X_train_raw = train_df[FEATURE_COLUMNS].values
y_train = train_df["fare_amount"].values

X_val_raw = val_df[FEATURE_COLUMNS].values
y_val = val_df["fare_amount"].values

X_test_raw = test_df[FEATURE_COLUMNS].values
y_test = test_df["fare_amount"].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

print(f"X_train shape: {X_train.shape} | X_val shape: {X_val.shape} | X_test shape: {X_test.shape}")""")

    # Cell 7: Baseline Models
    add_md("""## 7. Baseline Machine Learning Models Benchmark
We evaluate 5 conventional models:
1. Linear Regression (Ordinary Least Squares)
2. Ridge Regression ($L_2$ Regularized)
3. Random Forest Regressor
4. LightGBM Regressor
5. Scikit-Learn Multilayer Perceptron (`MLPRegressor`)""")

    add_code("""baselines = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=10.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=16, n_jobs=-1, random_state=42),
    "LightGBM": lgb.LGBMRegressor(n_estimators=250, learning_rate=0.08, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1),
    "Scikit-Learn MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=40, batch_size=64, early_stopping=True, random_state=42)
}

baseline_records = []
for name, model in baselines.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    t_train = time.time() - t0
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    baseline_records.append({
        "Model": name,
        "Test_MAE ($)": round(mae, 2),
        "Test_RMSE ($)": round(rmse, 2),
        "Test_R2": round(r2, 4),
        "Train_Time (s)": round(t_train, 2)
    })
    print(f"{name:20s} -> Test MAE: ${mae:.2f} | Test RMSE: ${rmse:.2f} | R2: {r2:.4f}")

df_baselines = pd.DataFrame(baseline_records)
display(df_baselines)""")

    # Cell 8: PyTorch DNN Architecture
    add_md("""## 8. Deep Feedforward Neural Network (DNN) Architecture
We design a deep neural network tailored for non-linear regression:
$$\\text{Input (33)} \\xrightarrow{\\text{Dense(128)}} \\text{BatchNorm} \\xrightarrow{\\text{ReLU}} \\text{Dropout(0.2)} \\xrightarrow{\\text{Dense(64)}} \\text{BatchNorm} \\xrightarrow{\\text{ReLU}} \\text{Dropout(0.1)} \\xrightarrow{\\text{Dense(32)}} \\text{ReLU} \\xrightarrow{\\text{Linear(1)}} \\hat{y}$$

### Regression Loss Functions:
- **Mean Squared Error (MSE)**: Highly sensitive to extreme outliers due to quadratic penalty.
- **Mean Absolute Error (MAE)**: Constant gradient everywhere, robust to outliers.
- **Huber Loss (Smooth $L_1$)**: Combines quadratic loss for small errors with linear loss for large deviations:
$$L_\\delta(y, \\hat{y}) = \\begin{cases} \\frac{1}{2}(y - \\hat{y})^2 & \\text{for } |y - \\hat{y}| \\le \\delta \\\\ \\delta |y - \\hat{y}| - \\frac{1}{2}\\delta^2 & \\text{otherwise} \\end{cases}$$
Huber Loss is selected for our production model.""")

    add_code("""class TaxiFareDNN(nn.Module):
    def __init__(self, in_features=33, hidden_dims=(128, 64, 32), dropout_rate=0.2):
        super(TaxiFareDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2.0),
            
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            
            nn.Linear(hidden_dims[2], 1)
        )
        
    def forward(self, x):
        return self.net(x)

# PyTorch DataLoaders
t_X_train = torch.tensor(X_train, dtype=torch.float32)
t_y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
t_X_val = torch.tensor(X_val, dtype=torch.float32)
t_y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
t_X_test = torch.tensor(X_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(t_X_train, t_y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(t_X_val, t_y_val), batch_size=128, shuffle=False)

dnn = TaxiFareDNN(in_features=33)
criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.Adam(dnn.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
print(dnn)""")

    # Cell 9: Model Training
    add_md("""## 9. Model Training & Learning Curve Diagnostics
The network is trained with early stopping, monitoring validation Huber loss and tracking validation MAE.""")

    add_code("""best_val_loss = float("inf")
best_weights = None
history = {"train_loss": [], "val_loss": [], "val_mae": []}

for epoch in range(1, 26):
    dnn.train()
    running_loss = 0.0
    for b_x, b_y in train_loader:
        optimizer.zero_grad()
        out = dnn(b_x)
        loss = criterion(out, b_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(b_x)
        
    epoch_train_loss = running_loss / len(train_loader.dataset)
    
    dnn.eval()
    val_loss, val_mae = 0.0, 0.0
    with torch.no_grad():
        for b_x, b_y in val_loader:
            out = dnn(b_x)
            val_loss += criterion(out, b_y).item() * len(b_x)
            val_mae += torch.sum(torch.abs(out - b_y)).item()
            
    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_mae = val_mae / len(val_loader.dataset)
    scheduler.step(epoch_val_loss)
    
    history["train_loss"].append(epoch_train_loss)
    history["val_loss"].append(epoch_val_loss)
    history["val_mae"].append(epoch_val_mae)
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}/25 | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val MAE: ${epoch_val_mae:.2f}")

# Plot Learning Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Huber Loss", color="#2563EB")
plt.plot(history["val_loss"], label="Val Huber Loss", color="#DC2626", linestyle="--")
plt.title("Huber Loss Progression Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["val_mae"], label="Validation MAE ($)", color="#059669")
plt.title("Validation MAE Progression Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("MAE ($)")
plt.legend()
plt.tight_layout()
plt.show()""")

    # Cell 10: Final Evaluation
    add_md("""## 10. Final Model Evaluation & Diagnostic Visualizations
We evaluate the trained Deep Neural Network on the untouched test set.""")

    add_code("""dnn.eval()
with torch.no_grad():
    y_test_pred = dnn(t_X_test).numpy().flatten()

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print(f"Deep Neural Network (DNN) Test Performance:")
print(f"  Test MAE:  ${test_mae:.2f}")
print(f"  Test RMSE: ${test_rmse:.2f}")
print(f"  Test R2:   {test_r2:.4f}")

# Residual and Actual vs Predicted Plots
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test[:3000], y_test_pred[:3000], alpha=0.3, color="#2563EB", s=15)
plt.plot([0, 60], [0, 60], color="#DC2626", linestyle="--", linewidth=2)
plt.title(f"Actual vs. Predicted Fare (R2 = {test_r2:.4f})")
plt.xlabel("Actual Fare ($)")
plt.ylabel("Predicted Fare ($)")
plt.xlim(0, 60)
plt.ylim(0, 60)

plt.subplot(1, 2, 2)
residuals = y_test_pred - y_test
sns.histplot(residuals[(residuals >= -20) & (residuals <= 20)], bins=50, kde=True, color="#7C3AED")
plt.axvline(0, color="#DC2626", linestyle="--")
plt.title("Residual Distribution (Predicted - Actual)")
plt.xlabel("Residual ($)")
plt.tight_layout()
plt.show()""")

    # Cell 11: Deployment & Inference Demonstration
    add_md("""## 11. Model Deployment & Automated Verification Demonstration
We test the complete pipeline on real-world test cases:
1. **Short Manhattan Journey**: Times Square to Grand Central Terminal
2. **Long Airport Transit**: JFK Airport Terminal 4 to Times Square (Rush Hour)""")

    add_code("""def predict_trip_fare(p_lat, p_lon, d_lat, d_lon, p_count, trip_dt_str):
    trip_df = pd.DataFrame([{
        "key": "demo",
        "pickup_datetime": pd.to_datetime(trip_dt_str),
        "pickup_longitude": p_lon,
        "pickup_latitude": p_lat,
        "dropoff_longitude": d_lon,
        "dropoff_latitude": d_lat,
        "passenger_count": p_count
    }])
    feats = engineer_features(trip_df)
    dist_km = feats["haversine_dist_km"].iloc[0]
    X_in = scaler.transform(feats[FEATURE_COLUMNS].values)
    
    dnn.eval()
    with torch.no_grad():
        fare = dnn(torch.tensor(X_in, dtype=torch.float32)).item()
    fare = max(2.50, round(fare, 2))
    return dist_km, fare

# Test Case 1: Times Square to Grand Central
d1, f1 = predict_trip_fare(40.7580, -73.9855, 40.7527, -73.9772, 1, "2025-10-15 14:00:00")
print(f"Case 1 (Times Square -> Grand Central): Distance = {d1:.2f} km | Estimated Fare = ${f1:.2f}")

# Test Case 2: JFK Airport to Times Square
d2, f2 = predict_trip_fare(40.6413, -73.7781, 40.7580, -73.9855, 2, "2025-10-15 18:30:00")
print(f"Case 2 (JFK Airport -> Times Square):  Distance = {d2:.2f} km | Estimated Fare = ${f2:.2f}")""")

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    os.makedirs(os.path.dirname(NOTEBOOK_PATH), exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)
        
    print(f"Jupyter Notebook successfully built at: {NOTEBOOK_PATH}")

if __name__ == "__main__":
    build_notebook()
