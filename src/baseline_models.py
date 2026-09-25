"""
Baseline Machine Learning Models for NYC Taxi Fare Prediction.
Trains and benchmarks multiple conventional regression models:
  1. Linear Regression (OLS)
  2. Ridge Regression (L2)
  3. Random Forest Regressor
  4. LightGBM Regressor
  5. Scikit-Learn MLPRegressor
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

from preprocessing import create_dataset_splits
from feature_engineering import prepare_and_scale_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

def evaluate_predictions(y_true, y_pred):
    """Calculates MAE, MSE, RMSE, and R2."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4)
    }

def train_and_evaluate_baselines():
    train_df, val_df, test_df = create_dataset_splits()
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feat_cols = prepare_and_scale_data(
        train_df, val_df, test_df
    )
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=16, min_samples_split=5, n_jobs=-1, random_state=42
        ),
        "LightGBM Regressor": lgb.LGBMRegressor(
            n_estimators=250, learning_rate=0.08, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1
        ),
        "MLPRegressor (Scikit-Learn)": MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", max_iter=40,
            batch_size=64, early_stopping=True, random_state=42
        )
    }
    
    results = []
    trained_models = {}
    
    print("\n" + "="*80)
    print("BENCHMARKING BASELINE REGRESSION MODELS ON NYC TAXI DATASET")
    print("="*80)
    
    for name, model in models.items():
        print(f"\n[Baseline] Training {name} on {X_train.shape[0]:,} samples...")
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        val_metrics = evaluate_predictions(y_val, y_val_pred)
        test_metrics = evaluate_predictions(y_test, y_test_pred)
        
        print(f"  Training Time: {elapsed:.2f}s")
        print(f"  Val  -> MAE: ${val_metrics['MAE']:.2f} | RMSE: ${val_metrics['RMSE']:.2f} | R2: {val_metrics['R2']:.4f}")
        print(f"  Test -> MAE: ${test_metrics['MAE']:.2f} | RMSE: ${test_metrics['RMSE']:.2f} | R2: {test_metrics['R2']:.4f}")
        
        results.append({
            "Model": name,
            "Train_Time_Sec": round(elapsed, 2),
            "Val_MAE": val_metrics["MAE"],
            "Val_MSE": val_metrics["MSE"],
            "Val_RMSE": val_metrics["RMSE"],
            "Val_R2": val_metrics["R2"],
            "Test_MAE": test_metrics["MAE"],
            "Test_MSE": test_metrics["MSE"],
            "Test_RMSE": test_metrics["RMSE"],
            "Test_R2": test_metrics["R2"]
        })
        trained_models[name] = model
        
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "baseline_comparison.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[Baseline] Comparison table saved to {csv_path}")
    print(df_results[["Model", "Test_MAE", "Test_RMSE", "Test_R2", "Train_Time_Sec"]].to_string())
    
    # Save trained baseline bundle (LightGBM and Random Forest for quick inference)
    bundle_path = os.path.join(MODELS_DIR, "baseline_models.pkl")
    joblib.dump({
        "LightGBM": trained_models["LightGBM Regressor"],
        "Linear Regression": trained_models["Linear Regression"]
    }, bundle_path)
    print(f"[Baseline] Saved inference baselines to {bundle_path}")
    
    return df_results

if __name__ == "__main__":
    train_and_evaluate_baselines()
