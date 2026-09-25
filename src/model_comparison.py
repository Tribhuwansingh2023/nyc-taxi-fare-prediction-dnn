"""
Model Comparison and Selection Module.
Combines baseline results with Deep Feedforward Neural Network results,
creates the consolidated results/model_comparison.csv,
and plots visualizations/15_model_comparison_bar.png.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")

def build_comparison_summary():
    base_csv = os.path.join(RESULTS_DIR, "baseline_comparison.csv")
    dnn_json = os.path.join(RESULTS_DIR, "dnn_performance_summary.json")
    
    if not os.path.exists(base_csv) or not os.path.exists(dnn_json):
        print("[Comparison] Waiting for baseline_comparison.csv and dnn_performance_summary.json...")
        return None
        
    df_base = pd.read_csv(base_csv)
    with open(dnn_json, "r") as f:
        dnn_data = json.load(f)
        
    dnn_row = {
        "Model": "Deep Feedforward Neural Network (PyTorch)",
        "Train_Time_Sec": dnn_data.get("training_time_sec", 65.0),
        "Val_MAE": dnn_data["val_metrics"]["MAE"],
        "Val_MSE": dnn_data["val_metrics"]["MSE"],
        "Val_RMSE": dnn_data["val_metrics"]["RMSE"],
        "Val_R2": dnn_data["val_metrics"]["R2"],
        "Test_MAE": dnn_data["test_metrics"]["MAE"],
        "Test_MSE": dnn_data["test_metrics"]["MSE"],
        "Test_RMSE": dnn_data["test_metrics"]["RMSE"],
        "Test_R2": dnn_data["test_metrics"]["R2"]
    }
    
    df_all = pd.concat([df_base, pd.DataFrame([dnn_row])], ignore_index=True)
    out_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"[Comparison] Consolidated table saved to {out_csv}")
    print(df_all.to_string())
    
    # -------------------------------------------------------------
    # Plot 15: Model Comparison Bar Chart
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    models = df_all["Model"].tolist()
    # Shorten names for clean display
    display_names = [m.replace("Regressor", "").replace(" (Scikit-Learn)", "").replace(" (PyTorch)", "").strip() for m in models]
    y_pos = np.arange(len(models))
    
    colors_mae = ["#93C5FD", "#93C5FD", "#60A5FA", "#3B82F6", "#60A5FA", "#1D4ED8"]
    bars1 = ax1.barh(y_pos, df_all["Test_MAE"], color=colors_mae, edgecolor="#1E3A8A")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(display_names, fontsize=11, fontweight="bold")
    ax1.set_xlabel("Test Mean Absolute Error ($) [Lower is Better]", fontsize=11)
    ax1.set_title("Test MAE Across Machine Learning & Deep Learning Models", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlim(0, max(df_all["Test_MAE"]) * 1.25)
    for bar in bars1:
        w = bar.get_width()
        ax1.annotate(f"${w:.2f}", xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points", va="center", fontsize=10, fontweight="bold")
                     
    colors_r2 = ["#A7F3D0", "#A7F3D0", "#34D399", "#10B981", "#34D399", "#047857"]
    bars2 = ax2.barh(y_pos, df_all["Test_R2"], color=colors_r2, edgecolor="#065F46")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(display_names, fontsize=11, fontweight="bold")
    ax2.set_xlabel("Test R² Determination Coefficient [Higher is Better]", fontsize=11)
    ax2.set_title("Test R² Score Across Models", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlim(0.75, 0.92)
    for bar in bars2:
        w = bar.get_width()
        ax2.annotate(f"{w:.4f}", xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points", va="center", fontsize=10, fontweight="bold")
                     
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "15_model_comparison_bar.png"), dpi=300)
    plt.close()
    print("[Comparison] Plot 15_model_comparison_bar.png saved.")
    
    return df_all

if __name__ == "__main__":
    build_comparison_summary()
