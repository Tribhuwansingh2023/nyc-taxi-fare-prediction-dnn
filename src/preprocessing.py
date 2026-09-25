"""
Data Preprocessing Pipeline for NYC Taxi Fare Prediction.
Handles missing values, cleans geographical and numerical anomalies,
and partitions data into Train (70%), Validation (15%), and Test (15%) subsets.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "train_sample_100k.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# NYC Metropolitan Bounding Box
NYC_LAT_MIN, NYC_LAT_MAX = 40.50, 40.95
NYC_LON_MIN, NYC_LON_MAX = -74.25, -73.70

def haversine_km(lon1, lat1, lon2, lat2):
    """Vectorized Haversine distance in kilometers."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return 6371.0 * c

def clean_taxi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw taxi trip records with strict validation rules.
    """
    initial_len = len(df)
    
    # Drop rows with nulls
    df = df.dropna().copy()
    
    # Parse datetimes
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime"]).copy()
    
    # Filter valid fare amounts (NYC base fare is $2.50)
    df = df[(df["fare_amount"] >= 2.50) & (df["fare_amount"] <= 200.0)].copy()
    
    # Filter valid passenger counts (1 to 6 in yellow cabs)
    df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)].copy()
    
    # Filter valid NYC coordinates
    df = df[
        (df["pickup_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX)) &
        (df["pickup_longitude"].between(NYC_LON_MIN, NYC_LON_MAX)) &
        (df["dropoff_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX)) &
        (df["dropoff_longitude"].between(NYC_LON_MIN, NYC_LON_MAX))
    ].copy()
    
    # Calculate initial distance and drop zero-distance trips with no movement
    d_km = haversine_km(
        df["pickup_longitude"].values, df["pickup_latitude"].values,
        df["dropoff_longitude"].values, df["dropoff_latitude"].values
    )
    df["temp_dist"] = d_km
    df = df[(df["temp_dist"] >= 0.05) & (df["temp_dist"] <= 80.0)].copy()
    df = df.drop(columns=["temp_dist"]).copy()
    
    final_len = len(df)
    retention = (final_len / initial_len) * 100
    print(f"[Preprocessing] Cleaned dataset: {final_len:,} / {initial_len:,} rows retained ({retention:.2f}%)")
    return df

def create_dataset_splits(random_state: int = 42):
    """
    Loads raw data, applies cleaning, and generates 70/15/15 Train/Validation/Test splits.
    Saves CSV files to data/processed/.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DIR, "val.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print(f"[Preprocessing] Processed partitions already exist in {PROCESSED_DIR}")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        return train_df, val_df, test_df
        
    print(f"[Preprocessing] Loading raw data from {RAW_CSV_PATH}...")
    df_raw = pd.read_csv(RAW_CSV_PATH)
    df_clean = clean_taxi_data(df_raw)
    
    # 70% Train, 30% Temp (which is split into 15% Val and 15% Test)
    train_df, temp_df = train_test_split(df_clean, test_size=0.30, random_state=random_state, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=random_state, shuffle=True)
    
    print(f"[Preprocessing] Train split: {len(train_df):,} records ({len(train_df)/len(df_clean)*100:.1f}%)")
    print(f"[Preprocessing] Val split:   {len(val_df):,} records ({len(val_df)/len(df_clean)*100:.1f}%)")
    print(f"[Preprocessing] Test split:  {len(test_df):,} records ({len(test_df)/len(df_clean)*100:.1f}%)")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[Preprocessing] Saved splits to {PROCESSED_DIR}")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = create_dataset_splits()
    print("Preprocessing completed successfully.")
