"""
Deep Feedforward Neural Network (DNN) for NYC Taxi Fare Prediction.
Implements a deep PyTorch regression network with Batch Normalization,
Dropout regularization, learning rate scheduling, and early stopping.
Also includes loss function comparison (MSE, MAE, Huber) and diagnostic plotting.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from preprocessing import create_dataset_splits
from feature_engineering import prepare_and_scale_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TaxiFareDNN(nn.Module):
    """
    Deep Feedforward Neural Network for Fare Regression.
    Architecture:
      Input (33) -> Linear(128) -> BatchNorm1d -> ReLU -> Dropout(0.2)
                 -> Linear(64)  -> BatchNorm1d -> ReLU -> Dropout(0.1)
                 -> Linear(32)  -> ReLU
                 -> Linear(1)   -> Output (Predicted Fare)
    """
    def __init__(self, in_features=33, hidden_dims=(128, 64, 32), dropout_rate=0.2):
        super(TaxiFareDNN, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate / 2.0)
        
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.relu3 = nn.ReLU()
        
        self.out = nn.Linear(hidden_dims[2], 1)
        
    def forward(self, x):
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        x = self.relu3(self.fc3(x))
        x = self.out(x)
        return x

def get_loss_function(name="huber", delta=1.0):
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name == "huber":
        return nn.HuberLoss(delta=delta)
    else:
        raise ValueError(f"Unknown loss function: {name}")

def get_optimizer(model, name="adam", lr=0.001, weight_decay=1e-5):
    name = name.lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None,
                epochs=40, patience=7, device="cpu"):
    """
    Trains PyTorch model with validation monitoring, early stopping, and history tracking.
    """
    model.to(device)
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_mae": [], "val_mae": [],
        "epochs": []
    }
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        running_train_mae = 0.0
        train_count = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * batch_x.size(0)
            running_train_mae += torch.sum(torch.abs(preds - batch_y)).item()
            train_count += batch_x.size(0)
            
        epoch_train_loss = running_train_loss / train_count
        epoch_train_mae = running_train_mae / train_count
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        running_val_mae = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                running_val_loss += loss.item() * batch_x.size(0)
                running_val_mae += torch.sum(torch.abs(preds - batch_y)).item()
                val_count += batch_x.size(0)
                
        epoch_val_loss = running_val_loss / val_count
        epoch_val_mae = running_val_mae / val_count
        
        if scheduler:
            scheduler.step(epoch_val_loss)
            
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_mae"].append(epoch_train_mae)
        history["val_mae"].append(epoch_val_mae)
        history["epochs"].append(epoch)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f} | Val MAE: ${epoch_val_mae:.2f}")
            
        # Early Stopping Check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch} (best Val Loss: {best_val_loss:.4f})")
                break
                
    elapsed = time.time() - start_time
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return model, history, elapsed

def evaluate_model_tensor(model, X_arr, y_arr, device="cpu"):
    """Evaluates PyTorch model on NumPy arrays."""
    model.eval()
    model.to(device)
    with torch.no_grad():
        inputs = torch.tensor(X_arr, dtype=torch.float32).to(device)
        preds = model(inputs).cpu().numpy().flatten()
        
    mae = float(np.mean(np.abs(preds - y_arr)))
    mse = float(np.mean((preds - y_arr)**2))
    rmse = float(np.sqrt(mse))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr))**2))
    ss_res = float(np.sum((y_arr - preds)**2))
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
    
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "predictions": preds
    }

def run_dnn_pipeline():
    train_df, val_df, test_df = create_dataset_splits()
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feat_cols = prepare_and_scale_data(
        train_df, val_df, test_df
    )
    
    # Prepare PyTorch Datasets
    t_X_train = torch.tensor(X_train, dtype=torch.float32)
    t_y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    t_X_val = torch.tensor(X_val, dtype=torch.float32)
    t_y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    
    train_loader = DataLoader(TensorDataset(t_X_train, t_y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(t_X_val, t_y_val), batch_size=128, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[DNN] Using device: {device} for Deep Neural Network training.")

    # -------------------------------------------------------------
    # 1. Loss Functions Comparison Experiment (MSE vs MAE vs Huber)
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("EXPERIMENT 1: COMPARATIVE STUDY OF REGRESSION LOSS FUNCTIONS")
    print("="*80)
    loss_names = ["MSE", "MAE", "Huber"]
    loss_histories = {}
    loss_results = {}
    
    for l_name in loss_names:
        print(f"\n[DNN Experiment] Training with Loss: {l_name}...")
        test_model = TaxiFareDNN(in_features=len(feat_cols), hidden_dims=(128, 64, 32), dropout_rate=0.2)
        crit = get_loss_function(l_name, delta=1.0)
        opt = optim.Adam(test_model.parameters(), lr=0.001, weight_decay=1e-5)
        
        _, hist, el = train_model(test_model, train_loader, val_loader, crit, opt, epochs=15, patience=5, device=device)
        loss_histories[l_name] = hist
        metrics = evaluate_model_tensor(test_model, X_val, y_val, device=device)
        loss_results[l_name] = {
            "Val_MAE": metrics["MAE"],
            "Val_RMSE": metrics["RMSE"],
            "Val_R2": metrics["R2"]
        }
        print(f"  Result ({l_name}) -> Val MAE: ${metrics['MAE']:.2f}, Val RMSE: ${metrics['RMSE']:.2f}, Val R2: {metrics['R2']:.4f}")

    # Plot 12: Loss functions comparison
    plt.figure(figsize=(10, 5))
    colors = {"MSE": "#DC2626", "MAE": "#2563EB", "Huber": "#059669"}
    for l_name in loss_names:
        plt.plot(loss_histories[l_name]["epochs"], loss_histories[l_name]["val_mae"],
                 label=f"{l_name} Loss (Val MAE: ${loss_results[l_name]['Val_MAE']:.2f})",
                 color=colors[l_name], marker="o", linewidth=2)
    plt.title("Validation MAE Progression Across Different Loss Functions", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Mean Absolute Error ($)", fontsize=11)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "12_loss_functions_comparison.png"), dpi=300)
    plt.close()
    print("[DNN] Plot 12_loss_functions_comparison.png saved.")

    # -------------------------------------------------------------
    # 2. Train Optimized Production Deep Feedforward Neural Network
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("TRAINING FINAL OPTIMIZED DEEP FEEDFORWARD NEURAL NETWORK (DNN)")
    print("="*80)
    final_model = TaxiFareDNN(in_features=len(feat_cols), hidden_dims=(128, 64, 32), dropout_rate=0.2)
    criterion = get_loss_function("huber", delta=1.0)
    optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    final_model, history, train_time = train_model(
        final_model, train_loader, val_loader, criterion, optimizer, scheduler=scheduler,
        epochs=35, patience=7, device=device
    )
    
    # Save trained PyTorch model state
    model_save_path = os.path.join(MODELS_DIR, "taxi_fare_dnn.pt")
    torch.save({
        "state_dict": final_model.state_dict(),
        "in_features": len(feat_cols),
        "hidden_dims": (128, 64, 32),
        "feature_cols": feat_cols
    }, model_save_path)
    print(f"\n[DNN] Model weights saved to {model_save_path}")

    # Evaluate on Train, Val, and Test
    train_metrics = evaluate_model_tensor(final_model, X_train, y_train, device=device)
    val_metrics = evaluate_model_tensor(final_model, X_val, y_val, device=device)
    test_metrics = evaluate_model_tensor(final_model, X_test, y_test, device=device)
    
    print("\n" + "-"*60)
    print("FINAL DNN PERFORMANCE EVALUATION")
    print(f"  Training Set   -> MAE: ${train_metrics['MAE']:.2f} | RMSE: ${train_metrics['RMSE']:.2f} | R2: {train_metrics['R2']:.4f}")
    print(f"  Validation Set -> MAE: ${val_metrics['MAE']:.2f} | RMSE: ${val_metrics['RMSE']:.2f} | R2: {val_metrics['R2']:.4f}")
    print(f"  Test Set       -> MAE: ${test_metrics['MAE']:.2f} | RMSE: ${test_metrics['RMSE']:.2f} | R2: {test_metrics['R2']:.4f}")
    print(f"  Training Time: {train_time:.2f} seconds")
    print("-"*60)

    # -------------------------------------------------------------
    # 3. Diagnostic Visualizations
    # -------------------------------------------------------------
    # Plot 11: Training vs Validation Loss and MAE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["epochs"], history["train_loss"], label="Train Huber Loss", color="#2563EB", linewidth=2)
    ax1.plot(history["epochs"], history["val_loss"], label="Val Huber Loss", color="#DC2626", linewidth=2, linestyle="--")
    ax1.set_title("Training vs Validation Huber Loss Curve", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Huber Loss", fontsize=11)
    ax1.legend(frameon=True, facecolor="white")
    
    ax2.plot(history["epochs"], history["train_mae"], label="Train MAE ($)", color="#059669", linewidth=2)
    ax2.plot(history["epochs"], history["val_mae"], label="Val MAE ($)", color="#D97706", linewidth=2, linestyle="--")
    ax2.set_title("Training vs Validation MAE Progression", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Mean Absolute Error ($)", fontsize=11)
    ax2.legend(frameon=True, facecolor="white")
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "11_dnn_training_validation_loss.png"), dpi=300)
    plt.close()
    print("[DNN] Plot 11_dnn_training_validation_loss.png saved.")

    # Plot 13: Actual vs Predicted Fare
    test_preds = test_metrics["predictions"]
    sample_idx = np.random.choice(len(y_test), min(5000, len(y_test)), replace=False)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test[sample_idx], test_preds[sample_idx], alpha=0.35, color="#2563EB", s=15, label="Test Samples")
    max_val = max(np.percentile(y_test, 99.5), np.percentile(test_preds, 99.5))
    plt.plot([0, max_val], [0, max_val], color="#DC2626", linestyle="--", linewidth=2.5, label="Ideal Prediction (y = x)")
    plt.title(f"DNN Predicted Fare vs. Actual Fare (Test Set, R² = {test_metrics['R2']:.4f})", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Actual Fare Amount ($)", fontsize=11)
    plt.ylabel("DNN Predicted Fare Amount ($)", fontsize=11)
    plt.xlim(0, 60)
    plt.ylim(0, 60)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "13_actual_vs_predicted_fare.png"), dpi=300)
    plt.close()
    print("[DNN] Plot 13_actual_vs_predicted_fare.png saved.")

    # Plot 14: Residual Distribution & Residual vs Predicted
    residuals = test_preds - y_test
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(residuals[(residuals >= -20) & (residuals <= 20)], bins=60, kde=True, color="#7C3AED", ax=ax1, edgecolor="white")
    ax1.set_title("Prediction Error Distribution (Residuals: ŷ - y)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Residual ($)", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.axvline(0, color="#DC2626", linestyle="--", linewidth=1.5)
    
    ax2.scatter(test_preds[sample_idx], residuals[sample_idx], alpha=0.3, color="#2563EB", s=15)
    ax2.axhline(0, color="#DC2626", linestyle="--", linewidth=2)
    ax2.set_title("Residuals vs. Predicted Fare", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Predicted Fare ($)", fontsize=11)
    ax2.set_ylabel("Residual (ŷ - y) ($)", fontsize=11)
    ax2.set_xlim(0, 60)
    ax2.set_ylim(-25, 25)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "14_residual_distribution.png"), dpi=300)
    plt.close()
    print("[DNN] Plot 14_residual_distribution.png saved.")

    # Save final results summary
    dnn_summary = {
        "architecture": "Input(33) -> Dense(128, BatchNorm, ReLU, Dropout 0.2) -> Dense(64, BatchNorm, ReLU, Dropout 0.1) -> Dense(32, ReLU) -> Linear(1)",
        "in_features": len(feat_cols),
        "loss_function": "Huber Loss (delta=1.0)",
        "optimizer": "Adam (lr=0.001, weight_decay=1e-5)",
        "scheduler": "ReduceLROnPlateau (factor=0.5, patience=3)",
        "training_time_sec": round(train_time, 2),
        "epochs_trained": len(history["epochs"]),
        "train_metrics": {k: v for k, v in train_metrics.items() if k != "predictions"},
        "val_metrics": {k: v for k, v in val_metrics.items() if k != "predictions"},
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "predictions"},
        "loss_study_val_metrics": loss_results
    }
    with open(os.path.join(RESULTS_DIR, "dnn_performance_summary.json"), "w") as f:
        json.dump(dnn_summary, f, indent=4)
        
    return dnn_summary

if __name__ == "__main__":
    run_dnn_pipeline()
