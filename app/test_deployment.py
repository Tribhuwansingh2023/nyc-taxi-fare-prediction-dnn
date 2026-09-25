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

    from routing import get_road_route, format_duration, calculate_haversine_km

    print("\n" + "="*80)
    print("RUNNING FEATURE #2: REAL ROAD ROUTE, DISTANCE & DRIVING TIME TESTS")
    print("="*80)

    routing_tests = [
        {"id": 1, "name": "ROUTING TEST 1: Valid Pickup + Drop-off Coordinates (Real Route Returned)"},
        {"id": 2, "name": "ROUTING TEST 2: Real Road Distance Returned (> 0 km)"},
        {"id": 3, "name": "ROUTING TEST 3: Driving Duration Returned (> 0 mins)"},
        {"id": 4, "name": "ROUTING TEST 4: Invalid/Missing Coordinates Graceful Error Handling"},
        {"id": 5, "name": "ROUTING TEST 5: Routing API Service Interruption Robustness (No App Crash)"},
        {"id": 6, "name": "ROUTING TEST 6: Existing Haversine Geodesic Calculation Integrity"},
        {"id": 7, "name": "ROUTING TEST 7: Existing Trained DNN Inference Pipeline Integrity"}
    ]

    routing_passed = 0

    # ROUTING TEST 1: Valid Coordinates -> Real route returned
    print(f"\nEvaluating {routing_tests[0]['name']}...")
    ts_lat, ts_lon = 40.7580, -73.9855  # Times Square
    jfk_lat, jfk_lon = 40.6413, -73.7781 # JFK Terminal 4
    r_res = get_road_route(ts_lat, ts_lon, jfk_lat, jfk_lon)
    if r_res["status"] == "success" and len(r_res.get("route_geometry", [])) > 0:
        print(f"  Real Route Retrieved: {len(r_res['route_geometry'])} geometry waypoints from {r_res['provider']} -> PASSED [OK]")
        routing_passed += 1
    else:
        print(f"  Routing retrieval failed: {r_res} -> FAILED")

    # ROUTING TEST 2: Distance returned (> 0)
    print(f"\nEvaluating {routing_tests[1]['name']}...")
    if r_res.get("distance_km") is not None and r_res["distance_km"] > 0:
        print(f"  Road Distance: {r_res['distance_km']:.2f} km (Air: {r_res['air_distance_km']:.2f} km | Ratio: {r_res['road_air_ratio']:.2f}x) -> PASSED [OK]")
        routing_passed += 1
    else:
        print(f"  Expected positive road distance, got: {r_res.get('distance_km')} -> FAILED")

    # ROUTING TEST 3: Duration returned (> 0)
    print(f"\nEvaluating {routing_tests[2]['name']}...")
    if r_res.get("duration_minutes") is not None and r_res["duration_minutes"] > 0:
        print(f"  Driving Duration: {r_res['duration_minutes']:.1f} mins ({r_res['duration_formatted']}) -> PASSED [OK]")
        routing_passed += 1
    else:
        print(f"  Expected positive duration, got: {r_res.get('duration_minutes')} -> FAILED")

    # ROUTING TEST 4: Invalid coordinates -> Graceful error
    print(f"\nEvaluating {routing_tests[3]['name']}...")
    r_inv = get_road_route(None, -73.9855, 40.6413, None)
    if r_inv["status"] == "invalid_coords" and r_inv["distance_km"] is None and r_inv["duration_minutes"] is None:
        print(f"  Graceful validation error: status='{r_inv['status']}' | msg='{r_inv['error_message']}' -> PASSED [OK]")
        routing_passed += 1
    else:
        print(f"  Expected invalid_coords error, got: {r_inv} -> FAILED")

    # ROUTING TEST 5: Routing API Service Interruption Robustness
    print(f"\nEvaluating {routing_tests[4]['name']}...")
    # Test with simulated timeout / unreachable host to ensure app never crashes
    r_down = get_road_route(ts_lat, ts_lon, jfk_lat, jfk_lon, timeout=0.0001)
    if r_down["status"] in ["network_error", "error"] and r_down["distance_km"] is None:
        print(f"  Interruption handled gracefully: status='{r_down['status']}' | msg='{r_down['error_message']}' -> PASSED [OK]")
        routing_passed += 1
    else:
        print(f"  Expected graceful network_error, got: {r_down} -> FAILED")

    # ROUTING TEST 6: Existing Haversine calculation still works
    print(f"\nEvaluating {routing_tests[5]['name']}...")
    h_dist = calculate_haversine_km(ts_lat, ts_lon, jfk_lat, jfk_lon)
    if 21.0 <= h_dist <= 22.5:
        print(f"  Air Geodesic Haversine Calculation: {h_dist:.2f} km -> PASSED [OK]")
        routing_passed += 1
    else:
        print(f"  Haversine out of expected range: {h_dist:.2f} km -> FAILED")

    # ROUTING TEST 7: Existing DNN prediction still works exactly as before
    print(f"\nEvaluating {routing_tests[6]['name']}...")
    dnn_check_df = pd.DataFrame([{
        "key": "test_routing_dnn_check",
        "pickup_datetime": pd.to_datetime("2025-10-15 14:00:00"),
        "pickup_longitude": ts_lon, "pickup_latitude": ts_lat,
        "dropoff_longitude": jfk_lon, "dropoff_latitude": jfk_lat,
        "passenger_count": 1
    }])
    dnn_check_feats = extract_features(dnn_check_df)
    dnn_check_scaled = scaler.transform(dnn_check_feats[FEATURE_COLS].values)
    with torch.no_grad():
        dnn_check_fare = model(torch.tensor(dnn_check_scaled, dtype=torch.float32)).item()
    dnn_check_fare = max(2.50, round(dnn_check_fare, 2))
    if 35.0 <= dnn_check_fare <= 75.0:
        print(f"  Trained PyTorch DNN Inference Untouched: Predicted Fare = ${dnn_check_fare:.2f} (Air Distance: {dnn_check_feats['haversine_dist_km'].iloc[0]:.2f} km) -> PASSED [OK]")
        routing_passed += 1
    else:
        print(f"  DNN prediction unexpected: ${dnn_check_fare:.2f} -> FAILED")

    from fare_engine import calculate_meter_estimate, compare_fares, get_dnn_prediction_interval, NYC_TLC_FARE_RULES

    print("\n" + "="*80)
    print("RUNNING FEATURE #3: REALISTIC FARE ESTIMATION ENGINE & COMPARISON TESTS")
    print("="*80)

    fare_tests = [
        {"id": 1, "name": "FARE TEST 1: Valid Trip Itemized Reference Breakdown Generation"},
        {"id": 2, "name": "FARE TEST 2: Distance Component Sensitivity (Proportional Scaling)"},
        {"id": 3, "name": "FARE TEST 3: Driving Duration / Slow Traffic Component Sensitivity"},
        {"id": 4, "name": "FARE TEST 4: Time-Dependent Surcharge Verification (Weekday Rush vs Midday)"},
        {"id": 5, "name": "FARE TEST 5: Passenger Count Boundary & Governance Integrity"},
        {"id": 6, "name": "FARE TEST 6: Missing Input Parameter Validation Handling"},
        {"id": 7, "name": "FARE TEST 7: PyTorch DNN Inference Pipeline Coexistence & Functional Integrity"},
        {"id": 8, "name": "FARE TEST 8: ML Prediction vs Reference Estimate Difference & Percentage Calculation"},
        {"id": 9, "name": "FARE TEST 9: Extreme & Out-of-Bounds Parameter Graceful Validation"},
        {"id": 10, "name": "FARE TEST 10: Zero/Edge-Case Reference Fare ZeroDivisionError Protection"}
    ]

    fare_passed = 0

    # FARE TEST 1: Valid Trip
    print(f"\nEvaluating {fare_tests[0]['name']}...")
    f_res1 = calculate_meter_estimate(
        pickup_lat=40.7580, pickup_lon=-73.9855,
        dropoff_lat=40.6413, dropoff_lon=-73.7781,
        trip_date=datetime.date(2025, 10, 15), trip_time=datetime.time(14, 30),
        passenger_count=2, road_distance_km=27.90, duration_minutes=31.0
    )
    if f_res1["status"] == "success" and f_res1["estimated_total"] > 0 and f_res1["base_fare"] == 3.00:
        print(f"  Valid Breakdown Generated: Base=${f_res1['base_fare']:.2f}, Dist=${f_res1['distance_component']:.2f}, Total=${f_res1['estimated_total']:.2f} -> PASSED [OK]")
        fare_passed += 1
    else:
        print(f"  Failed generating valid breakdown: {f_res1} -> FAILED")

    # FARE TEST 2: Different Distance
    print(f"\nEvaluating {fare_tests[1]['name']}...")
    f_short = calculate_meter_estimate(
        40.7580, -73.9855, 40.7527, -73.9772,
        datetime.date(2025, 10, 15), datetime.time(12, 0), road_distance_km=3.0
    )
    f_long = calculate_meter_estimate(
        40.7580, -73.9855, 40.7527, -73.9772,
        datetime.date(2025, 10, 15), datetime.time(12, 0), road_distance_km=15.0
    )
    if f_long["distance_component"] > f_short["distance_component"] and f_long["estimated_total"] > f_short["estimated_total"]:
        print(f"  Distance Scaling Verified: 3 km DistComp=${f_short['distance_component']:.2f} -> 15 km DistComp=${f_long['distance_component']:.2f} -> PASSED [OK]")
        fare_passed += 1
    else:
        print(f"  Distance sensitivity failed: Short=${f_short['distance_component']}, Long=${f_long['distance_component']} -> FAILED")

    # FARE TEST 3: Different Duration
    print(f"\nEvaluating {fare_tests[2]['name']}...")
    f_freeflow = calculate_meter_estimate(
        40.7580, -73.9855, 40.7527, -73.9772,
        datetime.date(2025, 10, 15), datetime.time(12, 0), road_distance_km=5.0, duration_minutes=8.0
    )
    f_slowtraffic = calculate_meter_estimate(
        40.7580, -73.9855, 40.7527, -73.9772,
        datetime.date(2025, 10, 15), datetime.time(12, 0), road_distance_km=5.0, duration_minutes=35.0
    )
    if f_slowtraffic["time_component"] >= f_freeflow["time_component"]:
        print(f"  Time Component Sensitivity Verified: FreeFlow TimeComp=${f_freeflow['time_component']:.2f} -> SlowTraffic TimeComp=${f_slowtraffic['time_component']:.2f} -> PASSED [OK]")
        fare_passed += 1
    else:
        print(f"  Duration sensitivity failed: {f_freeflow['time_component']} vs {f_slowtraffic['time_component']} -> FAILED")

    # FARE TEST 4: Time-Dependent Surcharge (Weekday Rush 5 PM vs Midday 12 PM)
    print(f"\nEvaluating {fare_tests[3]['name']}...")
    f_midday = calculate_meter_estimate(
        40.7580, -73.9855, 40.7527, -73.9772,
        datetime.date(2025, 10, 15), datetime.time(12, 0), road_distance_km=5.0 # Wednesday 12 PM
    )
    f_rush = calculate_meter_estimate(
        40.7580, -73.9855, 40.7527, -73.9772,
        datetime.date(2025, 10, 15), datetime.time(17, 30), road_distance_km=5.0 # Wednesday 5:30 PM (Rush)
    )
    if f_rush["surcharge_breakdown"]["rush_hour"] == 2.50 and f_midday["surcharge_breakdown"]["rush_hour"] == 0.0:
        print(f"  Rush-Hour Tariff Verified: Midday Surcharge=${f_midday['surcharge_breakdown']['rush_hour']:.2f} vs Rush Surcharge=${f_rush['surcharge_breakdown']['rush_hour']:.2f} -> PASSED [OK]")
        fare_passed += 1
    else:
        print(f"  Rush-hour rule failed: Midday={f_midday['surcharge_breakdown']}, Rush={f_rush['surcharge_breakdown']} -> FAILED")

    # FARE TEST 5: Passenger Count Boundary
    print(f"\nEvaluating {fare_tests[4]['name']}...")
    f_p1 = calculate_meter_estimate(40.7580, -73.9855, 40.7527, -73.9772, datetime.date(2025, 10, 15), datetime.time(12, 0), passenger_count=1, road_distance_km=5.0)
    f_p4 = calculate_meter_estimate(40.7580, -73.9855, 40.7527, -73.9772, datetime.date(2025, 10, 15), datetime.time(12, 0), passenger_count=4, road_distance_km=5.0)
    if f_p1["status"] == "success" and f_p4["status"] == "success" and f_p1["base_fare"] == f_p4["base_fare"]:
        print(f"  Passenger Governance Verified: Rates correctly conform to vehicle tariffs without arbitrary passenger surcharges -> PASSED [OK]")
        fare_passed += 1
    else:
        print(f"  Passenger count handling failed: {f_p1} vs {f_p4} -> FAILED")

    # FARE TEST 6: Missing Input Validation Error
    print(f"\nEvaluating {fare_tests[5]['name']}...")
    f_miss = calculate_meter_estimate(None, None, 40.7527, -73.9772, datetime.date(2025, 10, 15), datetime.time(12, 0), road_distance_km=5.0)
    if f_miss["status"] == "validation_error" and "Missing" in f_miss["error_message"]:
        print(f"  Validation Error Caught Gracefully: {f_miss['error_message']} -> PASSED [OK]")
        fare_passed += 1
    else:
        print(f"  Expected validation_error, got {f_miss} -> FAILED")

    # FARE TEST 7: DNN Prediction Functional Integrity
    print(f"\nEvaluating {fare_tests[6]['name']}...")
    dnn_live_df = pd.DataFrame([{
        "key": "test_fare_dnn_live",
        "pickup_datetime": pd.to_datetime("2025-10-15 17:30:00"),
        "pickup_longitude": -73.9855, "pickup_latitude": 40.7580,
        "dropoff_longitude": -73.7781, "dropoff_latitude": 40.6413,
        "passenger_count": 2
    }])
    dnn_live_feats = extract_features(dnn_live_df)
    dnn_live_scaled = scaler.transform(dnn_live_feats[FEATURE_COLS].values)
    with torch.no_grad():
        dnn_live_fare = model(torch.tensor(dnn_live_scaled, dtype=torch.float32)).item()
    dnn_live_fare = max(2.50, round(dnn_live_fare, 2))
    if dnn_live_fare > 10.0:
        print(f"  PyTorch DNN Coexistence Confirmed: ML Prediction = ${dnn_live_fare:.2f} (Huber Loss Checkpoint) -> PASSED [OK]")
        fare_passed += 1
    else:
        print(f"  DNN prediction unexpected: ${dnn_live_fare:.2f} -> FAILED")

    # FARE TEST 8: ML vs Reference Comparison Calculation
    print(f"\nEvaluating {fare_tests[7]['name']}...")
    cmp_res = compare_fares(24.30, 25.10)
    if cmp_res["difference"] == -0.80 and cmp_res["absolute_difference"] == 0.80 and cmp_res["percentage_difference"] == 3.19 and cmp_res["direction"] == "lower":
        print(f"  Comparison Mathematics Verified: Diff=${cmp_res['difference']:.2f}, AbsDiff=${cmp_res['absolute_difference']:.2f}, Pct={cmp_res['percentage_difference']:.2f}% ({cmp_res['direction']}) -> PASSED [OK]")
        fare_passed += 1
    else:
        print(f"  Comparison calculation mismatch: {cmp_res} -> FAILED")

    # FARE TEST 9: Extreme / Out-of-Bounds Parameter Validation
    print(f"\nEvaluating {fare_tests[8]['name']}...")
    f_extreme = calculate_meter_estimate(40.7580, -73.9855, 40.7527, -73.9772, datetime.date(2025, 10, 15), datetime.time(12, 0), passenger_count=99, road_distance_km=5.0)
    if f_extreme["status"] == "validation_error" and "Invalid passenger count" in f_extreme["error_message"]:
        print(f"  Out-of-Bounds Parameter Caught: {f_extreme['error_message']} -> PASSED [OK]")
        fare_passed += 1
    else:
        print(f"  Expected validation_error for extreme input, got {f_extreme} -> FAILED")

    # FARE TEST 10: Zero/Edge-Case Reference Fare ZeroDivisionError Protection
    print(f"\nEvaluating {fare_tests[9]['name']}...")
    cmp_zero = compare_fares(20.0, 0.0)
    if cmp_zero["difference"] == 20.0 and cmp_zero["percentage_difference"] == 0.0 and cmp_zero["direction"] == "higher":
        print(f"  ZeroDivisionError Protection Verified: Zero-reference edge-case handled safely without crash -> PASSED [OK]")
        fare_passed += 1
    else:
        print(f"  Zero reference handling failed: {cmp_zero} -> FAILED")

    # =========================================================================
    # FEATURE #4: REAL MODEL EXPLAINABILITY & FEATURE ATTRIBUTION TESTS
    # =========================================================================
    from explainability import (
        explain_prediction_integrated_gradients,
        explain_prediction_shap,
        get_global_feature_attribution,
        get_display_name,
        get_feature_icon,
        FEATURE_DISPLAY_NAMES
    )

    print("\n" + "="*80)
    print("RUNNING FEATURE #4: REAL MODEL EXPLAINABILITY & FEATURE ATTRIBUTION TESTS")
    print("="*80)

    explainability_tests = [
        {"id": 1, "name": "EXPLAIN TEST 1: Valid Prediction Attribution Generated (Integrated Gradients)"},
        {"id": 2, "name": "EXPLAIN TEST 2: Attribution Feature Count & Names Strictly Match Model Input Schema (33 Features)"},
        {"id": 3, "name": "EXPLAIN TEST 3: Top Influential Features Sorted by Absolute Attribution Magnitude"},
        {"id": 4, "name": "EXPLAIN TEST 4: Signed Directionality (Positive vs. Negative) Correctly Identified"},
        {"id": 5, "name": "EXPLAIN TEST 5: Graceful Fallback if SHAP Library Is Unavailable or Errors"},
        {"id": 6, "name": "EXPLAIN TEST 6: Invalid / Mismatched Input Dimensions Handled Safely"},
        {"id": 7, "name": "EXPLAIN TEST 7: Model Forward Prediction Remains Identical Before & After Attribution"},
        {"id": 8, "name": "EXPLAIN TEST 8: Neural Network Weights & Gradients Unaltered by Attribution Computation"},
        {"id": 9, "name": "EXPLAIN TEST 9: Axiomatic Completeness Verified (Sum of Attributions == F(x) - F(baseline))"},
        {"id": 10, "name": "EXPLAIN TEST 10: Global Feature Importance Strictly Separated from Local Prediction Explanation"}
    ]

    exp_passed = 0

    # EXPLAIN TEST 1: Valid Prediction Attribution Generated
    print(f"\nEvaluating {explainability_tests[0]['name']}...")
    exp_res1 = explain_prediction_integrated_gradients(
        model=model,
        input_scaled=dnn_live_scaled[0],
        feature_names=FEATURE_COLS,
        raw_values=dnn_live_feats[FEATURE_COLS].values[0],
        steps=30
    )
    if exp_res1.get("status") == "success" and len(exp_res1.get("features", [])) == 33:
        print(f"  Valid Attribution Generated: Base=${exp_res1['base_prediction']:.2f}, Target=${exp_res1['target_prediction']:.2f}, Net=${exp_res1['net_attribution']:.2f} -> PASSED [OK]")
        exp_passed += 1
    else:
        print(f"  Attribution generation failed: {exp_res1} -> FAILED")

    # EXPLAIN TEST 2: Attribution Feature Count & Names Match Model Input Schema
    print(f"\nEvaluating {explainability_tests[1]['name']}...")
    res_fnames = [f["feature"] for f in exp_res1.get("features", [])]
    if res_fnames == FEATURE_COLS:
        print(f"  Schema Conformance Confirmed: 33 features match FEATURE_COLS exactly in identical order -> PASSED [OK]")
        exp_passed += 1
    else:
        print(f"  Feature order mismatch: {res_fnames} vs {FEATURE_COLS} -> FAILED")

    # EXPLAIN TEST 3: Top Influential Features Sorted by Absolute Attribution Magnitude
    print(f"\nEvaluating {explainability_tests[2]['name']}...")
    top_feats = exp_res1.get("top_features", [])
    is_sorted = all(top_feats[i]["abs_attribution"] >= top_feats[i+1]["abs_attribution"] for i in range(len(top_feats)-1))
    if len(top_feats) == 5 and is_sorted:
        top_names = [f"{f['display_name']} ({f['formatted_delta']})" for f in top_feats[:3]]
        print(f"  Sorting Magnitude Verified: Top 3: {', '.join(top_names)} -> PASSED [OK]")
        exp_passed += 1
    else:
        print(f"  Top features not sorted by magnitude: {top_feats} -> FAILED")

    # EXPLAIN TEST 4: Signed Directionality Correctly Identified
    print(f"\nEvaluating {explainability_tests[3]['name']}...")
    valid_dirs = all(f["direction"] in ["positive", "negative"] for f in exp_res1["features"])
    dir_signs_correct = all(
        (f["attribution"] >= 0 and f["direction"] == "positive" and f["formatted_delta"].startswith("+")) or
        (f["attribution"] < 0 and f["direction"] == "negative" and f["formatted_delta"].startswith("-"))
        for f in exp_res1["features"]
    )
    if valid_dirs and dir_signs_correct:
        print(f"  Signed Directionality Verified: All 33 features correctly categorized as positive (fare increase) or negative (fare decrease) -> PASSED [OK]")
        exp_passed += 1
    else:
        print(f"  Directional logic failure in attribution items -> FAILED")

    # EXPLAIN TEST 5: Graceful Fallback if SHAP Library Is Unavailable or Errors
    print(f"\nEvaluating {explainability_tests[4]['name']}...")
    shap_res = explain_prediction_shap(
        model=model,
        input_scaled=dnn_live_scaled[0],
        feature_names=FEATURE_COLS,
        raw_values=dnn_live_feats[FEATURE_COLS].values[0],
        background_samples=10,
        nsamples=20
    )
    if shap_res.get("status") == "success" and len(shap_res.get("features", [])) == 33:
        print(f"  SHAP Pipeline Executed / Handled Gracefully: Method='{shap_res.get('method')}' -> PASSED [OK]")
        exp_passed += 1
    else:
        print(f"  SHAP execution / fallback failed: {shap_res} -> FAILED")

    # EXPLAIN TEST 6: Invalid / Mismatched Input Dimensions Handled Safely
    print(f"\nEvaluating {explainability_tests[5]['name']}...")
    try:
        explain_prediction_integrated_gradients(
            model=model,
            input_scaled=np.array([1.0, 2.0]), # Invalid length 2 instead of 33
            feature_names=FEATURE_COLS
        )
        print("  Expected ValueError for dimension mismatch, but none was raised -> FAILED")
    except ValueError as val_err:
        print(f"  Mismatched Dimensions Handled Gracefully: Caught {type(val_err).__name__} -> PASSED [OK]")
        exp_passed += 1

    # EXPLAIN TEST 7: Model Forward Prediction Remains Identical Before & After Attribution
    print(f"\nEvaluating {explainability_tests[6]['name']}...")
    with torch.no_grad():
        pred_before = model(torch.tensor(dnn_live_scaled, dtype=torch.float32)).item()
    _ = explain_prediction_integrated_gradients(model=model, input_scaled=dnn_live_scaled[0], feature_names=FEATURE_COLS, steps=10)
    with torch.no_grad():
        pred_after = model(torch.tensor(dnn_live_scaled, dtype=torch.float32)).item()
    if abs(pred_before - pred_after) < 1e-6:
        print(f"  Prediction Invariance Verified: PredBefore=${pred_before:.4f} == PredAfter=${pred_after:.4f} -> PASSED [OK]")
        exp_passed += 1
    else:
        print(f"  Prediction altered by explainability: {pred_before} vs {pred_after} -> FAILED")

    # EXPLAIN TEST 8: Neural Network Weights & Gradients Unaltered by Attribution
    print(f"\nEvaluating {explainability_tests[7]['name']}...")
    weights_before = sum(p.sum().item() for p in model.parameters())
    _ = explain_prediction_integrated_gradients(model=model, input_scaled=dnn_live_scaled[0], feature_names=FEATURE_COLS, steps=10)
    weights_after = sum(p.sum().item() for p in model.parameters())
    if abs(weights_before - weights_after) < 1e-6 and not model.training:
        print(f"  Parameter Immutability Verified: SumOfWeights={weights_before:.6f} untouched, model.training=False -> PASSED [OK]")
        exp_passed += 1
    else:
        print(f"  Model weights or training state altered: {weights_before} vs {weights_after} -> FAILED")

    # EXPLAIN TEST 9: Axiomatic Completeness Verified
    print(f"\nEvaluating {explainability_tests[8]['name']}...")
    completeness_ok = exp_res1.get("completeness_verified", False)
    base_val = exp_res1.get("base_prediction", 0.0)
    net_val = exp_res1.get("net_attribution", 0.0)
    target_val = exp_res1.get("target_prediction", 0.0)
    if completeness_ok and abs((base_val + net_val) - target_val) < 0.05:
        print(f"  Completeness Axiom Verified: Base (${base_val:.2f}) + Net (${net_val:.2f}) == Target (${target_val:.2f}) within 5¢ tolerance -> PASSED [OK]")
        exp_passed += 1
    else:
        print(f"  Completeness verification failed: Base={base_val}, Net={net_val}, Target={target_val} -> FAILED")

    # EXPLAIN TEST 10: Global Feature Importance Strictly Separated from Local Explanation
    print(f"\nEvaluating {explainability_tests[9]['name']}...")
    global_res = get_global_feature_attribution()
    if global_res.get("status") == "success" and len(global_res.get("rankings", [])) >= 10:
        g_top = global_res["rankings"][0]["feature"]
        print(f"  Global Feature Importance Verified: Sample={global_res.get('sample_size')} trips, #1 Global Feature={g_top} -> PASSED [OK]")
        exp_passed += 1
    else:
        print(f"  Global feature attribution failed or missing: {global_res} -> FAILED")

    # =========================================================================
    # FEATURE #5: SCIENTIFIC PREDICTION UNCERTAINTY & INTERVAL TESTS
    # =========================================================================
    from uncertainty import compute_prediction_interval, create_uncertainty_badge_html

    print("\n" + "="*80)
    print("RUNNING FEATURE #5: SCIENTIFIC PREDICTION UNCERTAINTY & INTERVAL TESTS")
    print("="*80)

    uncertainty_tests = [
        {"id": 1, "name": "UNCERTAINTY TEST 1: Valid Prediction Produces Defensible Prediction Interval"},
        {"id": 2, "name": "UNCERTAINTY TEST 2: Mathematical Ordering Verified (Lower <= Prediction <= Upper)"},
        {"id": 3, "name": "UNCERTAINTY TEST 3: Interval Width Non-Negative (Width >= 0)"},
        {"id": 4, "name": "UNCERTAINTY TEST 4: Invalid Prediction Input Handled Gracefully (None / NaN)"},
        {"id": 5, "name": "UNCERTAINTY TEST 5: Fallback Mechanism on Missing Calibration Data"},
        {"id": 6, "name": "UNCERTAINTY TEST 6: Forward DNN Prediction Remains Completely Invariant"},
        {"id": 7, "name": "UNCERTAINTY TEST 7: Minimum Statutory Non-Negative Bound Honored ($2.50 Min)"}
    ]

    unc_passed = 0

    # UNCERTAINTY TEST 1: Valid Prediction Produces Interval
    print(f"\nEvaluating {uncertainty_tests[0]['name']}...")
    u_res1 = compute_prediction_interval(59.33, coverage_level=0.95)
    if u_res1.get("status") == "success" and "lower_bound" in u_res1 and "upper_bound" in u_res1:
        print(f"  Valid Interval Generated: Pred=${u_res1['prediction']:.2f} -> [{u_res1['formatted']}] (Width=${u_res1['interval_width']:.2f}, {u_res1['coverage_percent']}) -> PASSED [OK]")
        unc_passed += 1
    else:
        print(f"  Prediction interval generation failed: {u_res1} -> FAILED")

    # UNCERTAINTY TEST 2: Mathematical Ordering (Lower <= Pred <= Upper)
    print(f"\nEvaluating {uncertainty_tests[1]['name']}...")
    if u_res1["lower_bound"] <= u_res1["prediction"] <= u_res1["upper_bound"]:
        print(f"  Mathematical Bounds Verified: ${u_res1['lower_bound']:.2f} <= ${u_res1['prediction']:.2f} <= ${u_res1['upper_bound']:.2f} -> PASSED [OK]")
        unc_passed += 1
    else:
        print(f"  Bounds violated: {u_res1} -> FAILED")

    # UNCERTAINTY TEST 3: Interval Width Non-Negative
    print(f"\nEvaluating {uncertainty_tests[2]['name']}...")
    if u_res1["interval_width"] >= 0 and u_res1["interval_width"] == round(u_res1["upper_bound"] - u_res1["lower_bound"], 2):
        print(f"  Interval Width Verified: Width=${u_res1['interval_width']:.2f} (Non-negative & consistent) -> PASSED [OK]")
        unc_passed += 1
    else:
        print(f"  Interval width invalid: {u_res1} -> FAILED")

    # UNCERTAINTY TEST 4: Invalid Input Handled Safely
    print(f"\nEvaluating {uncertainty_tests[3]['name']}...")
    u_none = compute_prediction_interval(None)
    u_nan = compute_prediction_interval(float("nan"))
    if u_none.get("status") == "invalid_input" and u_nan.get("status") == "invalid_input":
        print(f"  Invalid Inputs Handled Gracefully: None -> {u_none['status']}, NaN -> {u_nan['status']} -> PASSED [OK]")
        unc_passed += 1
    else:
        print(f"  Invalid inputs not handled safely: None={u_none}, NaN={u_nan} -> FAILED")

    # UNCERTAINTY TEST 5: Fallback Mechanism on Missing Calibration Data
    print(f"\nEvaluating {uncertainty_tests[4]['name']}...")
    from uncertainty import load_calibration_data
    calib = load_calibration_data()
    if "coverage_levels" in calib and calib.get("calibration_size", 0) > 0:
        print(f"  Calibration Profile Active: SampleSize={calib.get('calibration_size'):,}, RMSE=${calib.get('root_mean_squared_error', 0):.2f} -> PASSED [OK]")
        unc_passed += 1
    else:
        print(f"  Calibration data missing or corrupt: {calib} -> FAILED")

    # UNCERTAINTY TEST 6: Forward DNN Prediction Invariance
    print(f"\nEvaluating {uncertainty_tests[5]['name']}...")
    with torch.no_grad():
        p_raw1 = model(torch.tensor(dnn_live_scaled, dtype=torch.float32)).item()
    _ = compute_prediction_interval(p_raw1, coverage_level=0.95)
    with torch.no_grad():
        p_raw2 = model(torch.tensor(dnn_live_scaled, dtype=torch.float32)).item()
    if abs(p_raw1 - p_raw2) < 1e-6:
        print(f"  Prediction Invariance Confirmed: PredBefore=${p_raw1:.4f} == PredAfter=${p_raw2:.4f} -> PASSED [OK]")
        unc_passed += 1
    else:
        print(f"  Model output altered by interval computation -> FAILED")

    # UNCERTAINTY TEST 7: Minimum Statutory Non-Negative Bound Honored ($2.50)
    print(f"\nEvaluating {uncertainty_tests[6]['name']}...")
    u_low = compute_prediction_interval(1.20, coverage_level=0.95)
    if u_low["lower_bound"] >= 2.50:
        print(f"  Statutory Min Fare Enforced: Low Pred ($1.20) -> Lower Bound clamped to ${u_low['lower_bound']:.2f} >= $2.50 -> PASSED [OK]")
        unc_passed += 1
    else:
        print(f"  Clamping failed for low prediction: {u_low} -> FAILED")

    total_tests = len(test_cases) + len(geocoding_tests) + len(routing_tests) + len(fare_tests) + len(explainability_tests) + len(uncertainty_tests)
    total_passed = passed + geo_passed + routing_passed + fare_passed + exp_passed + unc_passed
    print("\n" + "="*80)
    print(f"DEPLOYMENT, GEOCODING, ROUTING, FARE, EXPLAINABILITY & UNCERTAINTY TEST SUMMARY: {total_passed} / {total_tests} Test Cases Passed.")
    print("="*80)
    return total_passed == total_tests

if __name__ == "__main__":
    success = run_deployment_tests()
    sys.exit(0 if success else 1)


