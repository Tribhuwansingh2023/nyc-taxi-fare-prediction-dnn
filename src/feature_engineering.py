"""
Feature Engineering and Scaling Pipeline for NYC Taxi Fare Prediction.
Extracts rich spatial, geodesic, temporal, and cyclical representations,
and provides leak-free standard scaling.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
SCALER_PATH = os.path.join(MODELS_DIR, "taxi_fare_scaler.pkl")
META_PATH = os.path.join(MODELS_DIR, "feature_metadata.json")

os.makedirs(MODELS_DIR, exist_ok=True)

# Key NYC Reference Hubs
LANDMARKS = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "Midtown": (40.7580, -73.9855)
}

def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculates great-circle distance between two points in kilometers."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return 6371.0 * c

def calculate_bearing(lon1, lat1, lon2, lat2):
    """Calculates bearing angle in degrees from pickup to dropoff."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all geospatial and temporal engineered features from raw attributes.
    """
    data = df.copy()
    
    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(data["pickup_datetime"]):
        data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
        
    # 1. Geodesic & Spatial Features
    p_lat = data["pickup_latitude"].values
    p_lon = data["pickup_longitude"].values
    d_lat = data["dropoff_latitude"].values
    d_lon = data["dropoff_longitude"].values
    
    # Haversine distance
    data["haversine_dist_km"] = haversine_distance(p_lon, p_lat, d_lon, d_lat)
    
    # Manhattan grid distance (approx 111 km per deg lat, ~82 km per deg lon in NYC)
    data["lat_diff"] = np.abs(d_lat - p_lat)
    data["lon_diff"] = np.abs(d_lon - p_lon)
    data["manhattan_dist_km"] = (data["lat_diff"] * 111.0) + (data["lon_diff"] * 82.0)
    
    # Euclidean coordinate distance
    data["euclidean_dist"] = np.sqrt(data["lat_diff"]**2 + data["lon_diff"]**2)
    
    # Direction / Bearing angle
    data["bearing_deg"] = calculate_bearing(p_lon, p_lat, d_lon, d_lat)
    
    # Distance to NYC Landmarks & Airports
    for hub_name, (hub_lat, hub_lon) in LANDMARKS.items():
        data[f"pickup_{hub_name}_dist"] = haversine_distance(p_lon, p_lat, hub_lon, hub_lat)
        data[f"dropoff_{hub_name}_dist"] = haversine_distance(d_lon, d_lat, hub_lon, hub_lat)
        
    # Interaction: distance per passenger
    passengers = np.maximum(data["passenger_count"].values, 1)
    data["dist_per_passenger"] = data["haversine_dist_km"] / passengers
    
    # 2. Temporal Features
    dt = data["pickup_datetime"].dt
    data["hour"] = dt.hour
    data["day"] = dt.day
    data["day_of_week"] = dt.dayofweek
    data["month"] = dt.month
    data["year"] = np.clip(dt.year, 2009, 2015)
    
    # Flags
    data["is_weekend"] = ((data["day_of_week"] == 5) | (data["day_of_week"] == 6)).astype(int)
    data["is_rush_hour"] = (
        (data["is_weekend"] == 0) &
        (((data["hour"] >= 7) & (data["hour"] <= 10)) | ((data["hour"] >= 16) & (data["hour"] <= 20)))
    ).astype(int)
    
    # Cyclical representations
    data["sin_hour"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["cos_hour"] = np.cos(2 * np.pi * data["hour"] / 24.0)
    data["sin_dow"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
    data["cos_dow"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)
    data["sin_month"] = np.sin(2 * np.pi * (data["month"] - 1) / 12.0)
    data["cos_month"] = np.cos(2 * np.pi * (data["month"] - 1) / 12.0)
    
    return data

# Canonical list of numerical model input features
FEATURE_COLS = [
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

def prepare_and_scale_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Applies feature extraction and fits StandardScaler on train_df ONLY.
    Returns scaled feature matrices and target vectors.
    """
    print(f"[FeatureEngineering] Extracting features for Train ({len(train_df):,}), Val ({len(val_df):,}), Test ({len(test_df):,})...")
    
    train_feat = extract_features(train_df)
    val_feat = extract_features(val_df)
    test_feat = extract_features(test_df)
    
    X_train = train_feat[FEATURE_COLS].values
    y_train = train_feat["fare_amount"].values
    
    X_val = val_feat[FEATURE_COLS].values
    y_val = val_feat["fare_amount"].values
    
    X_test = test_feat[FEATURE_COLS].values
    y_test = test_feat["fare_amount"].values
    
    # Fit scaler on training partition only to prevent data leakage
    print(f"[FeatureEngineering] Fitting StandardScaler on {len(FEATURE_COLS)} features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler and metadata
    joblib.dump(scaler, SCALER_PATH)
    meta = {
        "feature_cols": FEATURE_COLS,
        "num_features": len(FEATURE_COLS),
        "scaler_path": SCALER_PATH,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df)
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=4)
        
    print(f"[FeatureEngineering] Scaler saved to {SCALER_PATH}")
    print(f"[FeatureEngineering] Feature metadata saved to {META_PATH}")
    
    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test), FEATURE_COLS

if __name__ == "__main__":
    from preprocessing import create_dataset_splits
    train_df, val_df, test_df = create_dataset_splits()
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), cols = prepare_and_scale_data(train_df, val_df, test_df)
    print(f"X_train shape: {X_tr.shape}, y_train shape: {y_tr.shape}")
    print(f"Number of engineered features: {len(cols)}")
