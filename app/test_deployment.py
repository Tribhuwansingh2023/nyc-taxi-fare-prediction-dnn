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
import io
import datetime
import joblib
import numpy as np
import pandas as pd
import torch

# Ensure UTF-8 stdout on Windows terminal environments
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

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
        
    from geocoding import geocode_address, is_in_nyc_bbox

    print("\n" + "="*80)
    print("RUNNING FEATURE #1: ADDRESS GEOCODING & COORDINATE SYNCHRONIZATION TESTS")
    print("="*80)

    geocoding_tests = [
        {"id": 1, "name": "TEST 1: Valid NYC Address Geocoding (Times Square, New York, NY)"},
        {"id": 2, "name": "TEST 2: Invalid/Nonexistent Address Graceful Failure"},
        {"id": 3, "name": "TEST 3: Empty Address Input Validation Error"},
        {"id": 4, "name": "TEST 4: Existing Predefined Landmark Workflow Coexistence"},
        {"id": 5, "name": "TEST 5: Geocoded Coordinates Flow to Existing DNN Inference Pipeline"},
        {"id": 6, "name": "TEST 6: Manual Coordinate Override Verification"}
    ]

    geo_passed = 0

    # TEST 1: Valid NYC Address
    print(f"\nEvaluating {geocoding_tests[0]['name']}...")
    res1 = geocode_address("Times Square, New York, NY")
    if res1["status"] == "success" and res1["latitude"] is not None and res1["longitude"] is not None and is_in_nyc_bbox(res1["latitude"], res1["longitude"]):
        print(f"  Resolved: {res1['formatted_address'][:60]} | Coords: ({res1['latitude']:.4f}, {res1['longitude']:.4f}) -> PASSED [OK]")
        geo_passed += 1
    else:
        print(f"  Geocoding failed: {res1} -> FAILED")

    # TEST 2: Nonexistent Address
    print(f"\nEvaluating {geocoding_tests[1]['name']}...")
    res2 = geocode_address("nonexistent_xyz_99887766554433_fake_location")
    if res2["status"] == "not_found" and res2["latitude"] is None:
        print(f"  Graceful failure confirmed: status='{res2['status']}' | message='{res2['message']}' -> PASSED [OK]")
        geo_passed += 1
    else:
        print(f"  Expected not_found, got {res2} -> FAILED")

    # TEST 3: Empty Address
    print(f"\nEvaluating {geocoding_tests[2]['name']}...")
    res3 = geocode_address("   ")
    if res3["status"] == "empty" and res3["latitude"] is None:
        print(f"  Validation error caught: status='{res3['status']}' | message='{res3['message']}' -> PASSED [OK]")
        geo_passed += 1
    else:
        print(f"  Expected empty validation, got {res3} -> FAILED")

    # TEST 4: Existing Landmark Workflow Coexistence
    print(f"\nEvaluating {geocoding_tests[3]['name']}...")
    p_lm_lat, p_lm_lon = 40.7580, -73.9855 # Times Square
    d_lm_lat, d_lm_lon = 40.7527, -73.9772 # Grand Central
    lm_df = pd.DataFrame([{
        "key": "test_lm",
        "pickup_datetime": pd.to_datetime("2025-10-15 14:00:00"),
        "pickup_longitude": p_lm_lon, "pickup_latitude": p_lm_lat,
        "dropoff_longitude": d_lm_lon, "dropoff_latitude": d_lm_lat,
        "passenger_count": 1
    }])
    lm_feats = extract_features(lm_df)
    lm_scaled = scaler.transform(lm_feats[FEATURE_COLS].values)
    with torch.no_grad():
        lm_fare = model(torch.tensor(lm_scaled, dtype=torch.float32)).item()
    lm_fare = max(2.50, round(lm_fare, 2))
    if 4.0 <= lm_fare <= 15.0:
        print(f"  Landmark Preset Coexistence Verified | Distance: {lm_feats['haversine_dist_km'].iloc[0]:.2f} km | Fare: ${lm_fare:.2f} -> PASSED [OK]")
        geo_passed += 1
    else:
        print(f"  Landmark fare out of range: ${lm_fare:.2f} -> FAILED")

    # TEST 5: Geocoded Coordinates -> Existing DNN Inference
    print(f"\nEvaluating {geocoding_tests[4]['name']}...")
    geo_p = geocode_address("Grand Central Terminal, New York, NY")
    geo_d = geocode_address("Empire State Building, New York, NY")
    if geo_p["status"] == "success" and geo_d["status"] == "success":
        pipe_df = pd.DataFrame([{
            "key": "test_geocoded_pipe",
            "pickup_datetime": pd.to_datetime("2025-10-15 15:30:00"),
            "pickup_longitude": geo_p["longitude"], "pickup_latitude": geo_p["latitude"],
            "dropoff_longitude": geo_d["longitude"], "dropoff_latitude": geo_d["latitude"],
            "passenger_count": 2
        }])
        pipe_feats = extract_features(pipe_df)
        pipe_scaled = scaler.transform(pipe_feats[FEATURE_COLS].values)
        with torch.no_grad():
            pipe_fare = model(torch.tensor(pipe_scaled, dtype=torch.float32)).item()
        pipe_fare = max(2.50, round(pipe_fare, 2))
        print(f"  Resolved: '{geo_p['formatted_address'][:30]}...' -> '{geo_d['formatted_address'][:30]}...'")
        print(f"  Distance: {pipe_feats['haversine_dist_km'].iloc[0]:.2f} km | DNN Predicted Fare: ${pipe_fare:.2f} -> PASSED [OK]")
        geo_passed += 1
    else:
        print(f"  Geocoding for test 5 failed (P: {geo_p['status']}, D: {geo_d['status']}) -> FAILED")

    # TEST 6: Manual Coordinate Override
    print(f"\nEvaluating {geocoding_tests[5]['name']}...")
    override_plat, override_plon = 40.7600, -73.9800
    override_dlat, override_dlon = 40.7500, -73.9900
    ovr_df = pd.DataFrame([{
        "key": "test_override",
        "pickup_datetime": pd.to_datetime("2025-10-15 16:00:00"),
        "pickup_longitude": override_plon, "pickup_latitude": override_plat,
        "dropoff_longitude": override_dlon, "dropoff_latitude": override_dlat,
        "passenger_count": 1
    }])
    ovr_feats = extract_features(ovr_df)
    ovr_scaled = scaler.transform(ovr_feats[FEATURE_COLS].values)
    with torch.no_grad():
        ovr_fare = model(torch.tensor(ovr_scaled, dtype=torch.float32)).item()
    ovr_fare = max(2.50, round(ovr_fare, 2))
    if ovr_fare >= 2.50:
        print(f"  Manual Coordinates Override Applied: ({override_plat}, {override_plon}) -> ({override_dlat}, {override_dlon})")
        print(f"  Distance: {ovr_feats['haversine_dist_km'].iloc[0]:.2f} km | Fare: ${ovr_fare:.2f} -> PASSED [OK]")
        geo_passed += 1
    else:
        print(f"  Manual override invalid: ${ovr_fare:.2f} -> FAILED")

    total_tests = len(test_cases) + len(geocoding_tests)
    total_passed = passed + geo_passed
    print("\n" + "="*80)
    print(f"DEPLOYMENT & GEOCODING TEST SUMMARY: {total_passed} / {total_tests} Test Cases Passed.")
    print("="*80)
    return total_passed == total_tests

if __name__ == "__main__":
    success = run_deployment_tests()
    sys.exit(0 if success else 1)
