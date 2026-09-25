"""
Multi-Model Comparison and Benchmarking Engine for NYC Taxi Fare Prediction.
Loads real trained models (PyTorch DNN, LightGBM, Linear Regression), measures live inference latency,
evaluates prediction spread across models on identical trip telemetry, and retrieves empirical validation benchmarks.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import torch
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_COMP_CSV = os.path.join(RESULTS_DIR, "model_comparison.csv")
BASELINE_COMP_CSV = os.path.join(RESULTS_DIR, "baseline_comparison.csv")
DNN_SUMMARY_JSON = os.path.join(RESULTS_DIR, "dnn_performance_summary.json")

# In-memory cache for loaded model artifacts to prevent repeated disk I/O
_LOADED_MODELS: Optional[Dict[str, Any]] = None


def get_available_models_bundle() -> Tuple[Dict[str, Any], List[str]]:
    """
    Loads all trained model artifacts available in the repository.
    Returns:
      (available_models_dict, list_of_unavailable_model_names)
    """
    global _LOADED_MODELS
    if _LOADED_MODELS is not None:
        return _LOADED_MODELS["available"], _LOADED_MODELS["unavailable"]

    available = {}
    unavailable = []

    # 1. PyTorch Deep Feedforward Neural Network
    dnn_path = os.path.join(MODELS_DIR, "taxi_fare_dnn.pt")
    if os.path.exists(dnn_path):
        try:
            from dnn_model import TaxiFareDNN
            checkpoint = torch.load(dnn_path, map_location=torch.device("cpu"), weights_only=False)
            in_feats = checkpoint.get("in_features", 33) if isinstance(checkpoint, dict) and "in_features" in checkpoint else 33
            hidden_dims = checkpoint.get("hidden_dims", (128, 64, 32)) if isinstance(checkpoint, dict) and "hidden_dims" in checkpoint else (128, 64, 32)
            model = TaxiFareDNN(in_features=in_feats, hidden_dims=hidden_dims)
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            available["Deep Neural Network (PyTorch)"] = {
                "model": model,
                "type": "Deep Learning (PyTorch MLP)",
                "framework": "PyTorch",
                "features_expected": in_feats
            }
        except Exception as e:
            logger.error(f"Failed loading PyTorch DNN: {e}")
            unavailable.append("Deep Neural Network (PyTorch)")
    else:
        unavailable.append("Deep Neural Network (PyTorch)")

    # 2. Baseline models bundle (LightGBM, Linear Regression)
    baselines_path = os.path.join(MODELS_DIR, "baseline_models.pkl")
    if os.path.exists(baselines_path):
        try:
            baselines = joblib.load(baselines_path)
            if "LightGBM" in baselines:
                available["LightGBM Regressor"] = {
                    "model": baselines["LightGBM"],
                    "type": "Gradient Boosting",
                    "framework": "LightGBM",
                    "features_expected": 33
                }
            if "Linear Regression" in baselines:
                available["Linear Regression (OLS)"] = {
                    "model": baselines["Linear Regression"],
                    "type": "Linear Baseline",
                    "framework": "Scikit-Learn",
                    "features_expected": 33
                }
        except Exception as e:
            logger.error(f"Failed loading baseline models bundle: {e}")
            unavailable.append("LightGBM Regressor")
            unavailable.append("Linear Regression (OLS)")

    # 3. Known models that exist in offline benchmark evaluation but without heavy live weight files
    all_benchmark_models = ["Random Forest Regressor", "Ridge Regression", "MLPRegressor (Scikit-Learn)"]
    for m in all_benchmark_models:
        if m not in available:
            unavailable.append(m)

    _LOADED_MODELS = {
        "available": available,
        "unavailable": unavailable
    }
    return available, unavailable


def load_benchmark_metrics() -> pd.DataFrame:
    """
    Loads empirical test and validation regression metrics from saved evaluation results.
    Does NOT fabricate metrics; reads directly from results/model_comparison.csv.
    """
    if os.path.exists(MODEL_COMP_CSV):
        try:
            df = pd.read_csv(MODEL_COMP_CSV)
            return df
        except Exception as e:
            logger.error(f"Error reading {MODEL_COMP_CSV}: {e}")

    if os.path.exists(BASELINE_COMP_CSV):
        try:
            df = pd.read_csv(BASELINE_COMP_CSV)
            return df
        except Exception as e:
            logger.error(f"Error reading {BASELINE_COMP_CSV}: {e}")

    # Fallback to empty DataFrame if no files exist
    return pd.DataFrame(columns=["Model", "Test_MAE", "Test_MSE", "Test_RMSE", "Test_R2", "Train_Time_Sec"])


def benchmark_single_trip(
    X_scaled: np.ndarray,
    weather_mult: float = 1.0,
    is_jfk_flat: bool = False
) -> Dict[str, Any]:
    """
    Executes live forward inference across all currently available trained models
    using the exact same 33-feature scaled input vector.
    
    Measures:
      - Exact predicted fare ($)
      - Exact inference latency in milliseconds (ms) using high-precision performance counter
      - Prediction spread (Min, Max, Range, Mean)
    """
    available, unavailable = get_available_models_bundle()
    if X_scaled.ndim == 1:
        X_scaled = X_scaled.reshape(1, -1)

    results = []

    # 1. Evaluate PyTorch DNN
    if "Deep Neural Network (PyTorch)" in available:
        dnn_entry = available["Deep Neural Network (PyTorch)"]
        model = dnn_entry["model"]
        try:
            t_start = time.perf_counter()
            t_input = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                raw_pred = float(model(t_input).item())
            lat_ms = (time.perf_counter() - t_start) * 1000.0

            if is_jfk_flat:
                fare = max(2.50, round(70.00 * weather_mult, 2))
            else:
                fare = max(2.50, round(raw_pred * weather_mult, 2))

            results.append({
                "model_name": "Deep Neural Network (PyTorch)",
                "display_name": "Deep Neural Network (PyTorch)",
                "predicted_fare": fare,
                "latency_ms": round(lat_ms, 2),
                "type": "Deep Learning",
                "framework": "PyTorch",
                "color": "#10B981",
                "status": "success"
            })
        except Exception as e:
            logger.error(f"DNN inference failed: {e}")
            results.append({
                "model_name": "Deep Neural Network (PyTorch)",
                "display_name": "Deep Neural Network (PyTorch)",
                "predicted_fare": None,
                "latency_ms": None,
                "type": "Deep Learning",
                "status": "error",
                "error": str(e)
            })

    # 2. Evaluate LightGBM
    if "LightGBM Regressor" in available:
        lgb_model = available["LightGBM Regressor"]["model"]
        try:
            t_start = time.perf_counter()
            raw_pred = float(lgb_model.predict(X_scaled)[0])
            lat_ms = (time.perf_counter() - t_start) * 1000.0

            if is_jfk_flat:
                fare = max(2.50, round(70.00 * weather_mult, 2))
            else:
                fare = max(2.50, round(raw_pred * weather_mult, 2))

            results.append({
                "model_name": "LightGBM Regressor",
                "display_name": "LightGBM Regressor",
                "predicted_fare": fare,
                "latency_ms": round(lat_ms, 2),
                "type": "Gradient Boosting",
                "framework": "LightGBM",
                "color": "#38BDF8",
                "status": "success"
            })
        except Exception as e:
            logger.error(f"LightGBM inference failed: {e}")

    # 3. Evaluate Linear Regression (OLS)
    if "Linear Regression (OLS)" in available:
        lr_model = available["Linear Regression (OLS)"]["model"]
        try:
            t_start = time.perf_counter()
            raw_pred = float(lr_model.predict(X_scaled)[0])
            lat_ms = (time.perf_counter() - t_start) * 1000.0

            if is_jfk_flat:
                fare = max(2.50, round(70.00 * weather_mult, 2))
            else:
                fare = max(2.50, round(raw_pred * weather_mult, 2))

            results.append({
                "model_name": "Linear Regression (OLS)",
                "display_name": "Linear Regression (OLS)",
                "predicted_fare": fare,
                "latency_ms": round(lat_ms, 2),
                "type": "Linear Baseline",
                "framework": "Scikit-Learn",
                "color": "#F59E0B",
                "status": "success"
            })
        except Exception as e:
            logger.error(f"Linear Regression inference failed: {e}")

    # Calculate spread across successful predictions
    fares = [r["predicted_fare"] for r in results if r.get("status") == "success" and r["predicted_fare"] is not None]
    if fares:
        min_fare = min(fares)
        max_fare = max(fares)
        spread = round(max_fare - min_fare, 2)
        mean_fare = round(float(np.mean(fares)), 2)
    else:
        min_fare = max_fare = spread = mean_fare = 0.0

    return {
        "status": "success" if results else "no_models_available",
        "predictions": results,
        "unavailable_models": unavailable,
        "min_fare": min_fare,
        "max_fare": max_fare,
        "spread": spread,
        "mean_fare": mean_fare,
        "model_count": len(results)
    }


def create_trip_benchmark_bar_chart(
    benchmark_results: Dict[str, Any],
    is_night_theme: bool = True
) -> go.Figure:
    """
    Renders an aesthetic horizontal bar chart comparing the predicted fare across all active models.
    """
    predictions = [p for p in benchmark_results.get("predictions", []) if p.get("status") == "success"]
    if not predictions:
        fig = go.Figure()
        fig.add_annotation(text="No model predictions available", showarrow=False)
        return fig

    # Reverse order so best / primary model appears at top
    rev_preds = list(reversed(predictions))
    y_names = [f"{p['display_name']} ({p.get('latency_ms', 0):.1f}ms)" for p in rev_preds]
    x_fares = [p["predicted_fare"] for p in rev_preds]
    colors = [p["color"] for p in rev_preds]
    text_labels = [f"${f:.2f}" for f in x_fares]

    chart_font_color = "#F8FAFC" if is_night_theme else "#0F172A"
    grid_color = "rgba(255, 255, 255, 0.08)" if is_night_theme else "#E2E8F0"

    fig = go.Figure(go.Bar(
        x=x_fares,
        y=y_names,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=text_labels,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Predicted Fare: %{text}<extra></extra>"
    ))

    max_fare = max(x_fares) * 1.22 if x_fares else 50.0

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=chart_font_color, size=11),
        xaxis=dict(
            title="Predicted Fare ($ USD)",
            gridcolor=grid_color,
            zeroline=False,
            range=[0, max_fare]
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(size=11)
        ),
        margin=dict(l=15, r=45, t=20, b=35),
        height=max(200, len(predictions) * 55)
    )
    return fig


def create_benchmark_metrics_chart(
    df_metrics: pd.DataFrame,
    is_night_theme: bool = True
) -> go.Figure:
    """
    Renders grouped bar chart showing MAE, RMSE, and R2 across models from the benchmark table.
    """
    if df_metrics.empty:
        fig = go.Figure()
        fig.add_annotation(text="Benchmark metrics unavailable", showarrow=False)
        return fig

    # Map column names if present
    mae_col = "Test_MAE" if "Test_MAE" in df_metrics.columns else "Val_MAE"
    rmse_col = "Test_RMSE" if "Test_RMSE" in df_metrics.columns else "Val_RMSE"
    r2_col = "Test_R2" if "Test_R2" in df_metrics.columns else "Val_R2"

    models = df_metrics["Model"].tolist()
    maes = df_metrics[mae_col].tolist() if mae_col in df_metrics.columns else []
    rmses = df_metrics[rmse_col].tolist() if rmse_col in df_metrics.columns else []

    chart_font_color = "#F8FAFC" if is_night_theme else "#0F172A"
    grid_color = "rgba(255, 255, 255, 0.08)" if is_night_theme else "#E2E8F0"

    fig = go.Figure()
    if maes:
        fig.add_trace(go.Bar(
            name="MAE ($)",
            x=models,
            y=maes,
            marker_color="#10B981" if is_night_theme else "#16A34A",
            text=[f"${v:.2f}" for v in maes],
            textposition="outside"
        ))
    if rmses:
        fig.add_trace(go.Bar(
            name="RMSE ($)",
            x=models,
            y=rmses,
            marker_color="#38BDF8" if is_night_theme else "#0284C7",
            text=[f"${v:.2f}" for v in rmses],
            textposition="outside"
        ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=chart_font_color, size=11),
        yaxis=dict(title="Error ($ USD — lower is better)", gridcolor=grid_color),
        xaxis=dict(tickangle=-15),
        margin=dict(l=20, r=20, t=30, b=45),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
