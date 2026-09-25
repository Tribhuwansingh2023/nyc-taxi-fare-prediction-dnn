"""
Automated Deployment Verification Test Suite for NYC Taxi Fare Prediction.
Tests the saved models, preprocessing pipeline, and inference endpoints across:
  1. Standard short Manhattan trip
  2. Long Airport transit (JFK to Midtown)
  3. Borderline multi-passenger trip
  4. Robustness against boundary edge cases
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import datetime
import joblib
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from feature_engineering import extract_features, FEATURE_COLS
from dnn_model import TaxiFareDNN

def run_deployment_tests():
    print("="*80)
    print("RUNNING AUTOMATED DEPLOYMENT AND INFERENCE VERIFICATION TESTS")
    print("="*80)
    
    scaler_path = os.path.join(BASE_DIR, "saved_models", "taxi_fare_scaler.pkl")
    dnn_path = os.path.join(BASE_DIR, "saved_models", "taxi_fare_dnn.pt")
    
    assert os.path.exists(scaler_path), f"Missing scaler at {scaler_path}"
    assert os.path.exists(dnn_path), f"Missing model checkpoint at {dnn_path}"
    
    scaler = joblib.load(scaler_path)
    checkpoint = torch.load(dnn_path, map_location="cpu", weights_only=False)
    
    model = TaxiFareDNN(in_features=checkpoint["in_features"], hidden_dims=checkpoint["hidden_dims"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("[TestSetup] Scaler and PyTorch DNN successfully loaded into memory.")
    
    test_cases = [
        {
            "name": "Case 1: Standard Short Manhattan Trip (Times Square -> Grand Central)",
            "pickup_lat": 40.7580, "pickup_lon": -73.9855,
            "dropoff_lat": 40.7527, "dropoff_lon": -73.9772,
            "passenger_count": 1,
            "datetime": "2025-10-15 14:00:00",
            "expected_range": (4.0, 15.0)
        },
        {
            "name": "Case 2: Long Airport Journey (JFK Terminal 4 -> Times Square)",
            "pickup_lat": 40.6413, "pickup_lon": -73.7781,
            "dropoff_lat": 40.7580, "dropoff_lon": -73.9855,
            "passenger_count": 2,
            "datetime": "2025-10-15 18:30:00", # Rush hour
            "expected_range": (35.0, 75.0)
        },
        {
            "name": "Case 3: Borderline Ultra-Short Trip (200m hop)",
            "pickup_lat": 40.7580, "pickup_lon": -73.9855,
            "dropoff_lat": 40.7595, "dropoff_lon": -73.9840,
            "passenger_count": 5,
            "datetime": "2025-10-15 10:00:00",
            "expected_range": (2.5, 12.0)
        },
        {
            "name": "Case 4: LaGuardia Airport to Lower Manhattan / Wall St",
            "pickup_lat": 40.7769, "pickup_lon": -73.8740,
            "dropoff_lat": 40.7075, "dropoff_lon": -74.0090,
            "passenger_count": 3,
            "datetime": "2025-10-18 22:00:00", # Weekend night
            "expected_range": (25.0, 55.0)
        }
    ]
    
    passed = 0
    for idx, tc in enumerate(test_cases, 1):
        print(f"\nEvaluating {tc['name']}...")
        input_df = pd.DataFrame([{
            "key": f"test_{idx}",
            "pickup_datetime": pd.to_datetime(tc["datetime"]),
            "pickup_longitude": tc["pickup_lon"],
            "pickup_latitude": tc["pickup_lat"],
            "dropoff_longitude": tc["dropoff_lon"],
            "dropoff_latitude": tc["dropoff_lat"],
            "passenger_count": tc["passenger_count"]
        }])
        
        feats = extract_features(input_df)
        dist_km = feats["haversine_dist_km"].iloc[0]
        X_in = feats[FEATURE_COLS].values
        X_sc = scaler.transform(X_in)
        
        with torch.no_grad():
            fare_pred = model(torch.tensor(X_sc, dtype=torch.float32)).item()
        fare_pred = max(2.50, round(fare_pred, 2))
        
        low, high = tc["expected_range"]
        status = "PASSED [OK]" if (low <= fare_pred <= high) else "WARNING [OUT OF RANGE]"
        if low <= fare_pred <= high:
            passed += 1
            
        print(f"  Distance: {dist_km:.2f} km | Predicted Fare: ${fare_pred:.2f} (Expected: ${low:.2f} - ${high:.2f}) -> {status}")
        
    print("\n" + "="*80)
    print(f"DEPLOYMENT TEST SUMMARY: {passed} / {len(test_cases)} Test Cases Passed.")
    print("="*80)
    return passed == len(test_cases)

if __name__ == "__main__":
    success = run_deployment_tests()
    sys.exit(0 if success else 1)
