"""
Hyperparameter Tuning Module for NYC Taxi Fare Prediction Deep Neural Network.
Conducts systematic tuning across neural architecture depth, layer widths,
learning rates, batch sizes, dropout rates, optimizers, and loss functions.
Generates results/hyperparameter_tuning_results.csv.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from preprocessing import create_dataset_splits
from feature_engineering import prepare_and_scale_data
from dnn_model import get_loss_function, get_optimizer, evaluate_model_tensor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

class ConfigurableDNN(nn.Module):
    """Dynamically creates feedforward architecture from layer width tuple."""
    def __init__(self, in_features, layer_sizes, dropout=0.2):
        super(ConfigurableDNN, self).__init__()
        layers = []
        prev_dim = in_features
        for i, dim in enumerate(layer_sizes):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if dropout > 0 and i < len(layer_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def run_tuning_grid():
    train_df, val_df, test_df = create_dataset_splits()
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feat_cols = prepare_and_scale_data(
        train_df, val_df, test_df
    )
    
    t_X_train = torch.tensor(X_train, dtype=torch.float32)
    t_y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    t_X_val = torch.tensor(X_val, dtype=torch.float32)
    t_y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    
    val_loader = DataLoader(TensorDataset(t_X_val, t_y_val), batch_size=256, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 8 Representative Search Trials reflecting Section 19 table
    trials = [
        {"id": "Trial-1 (Baseline)", "layers": (64, 32), "lr": 0.001, "batch": 64, "dropout": 0.0, "opt": "adam", "loss": "mse"},
        {"id": "Trial-2 (Deeper)", "layers": (128, 64, 32), "lr": 0.001, "batch": 64, "dropout": 0.2, "opt": "adam", "loss": "mse"},
        {"id": "Trial-3 (Huber Loss)", "layers": (128, 64, 32), "lr": 0.001, "batch": 64, "dropout": 0.2, "opt": "adam", "loss": "huber"},
        {"id": "Trial-4 (Wide Arch)", "layers": (256, 128, 64), "lr": 0.0005, "batch": 64, "dropout": 0.3, "opt": "adam", "loss": "huber"},
        {"id": "Trial-5 (Small Batch)", "layers": (128, 64, 32), "lr": 0.0005, "batch": 32, "dropout": 0.2, "opt": "adam", "loss": "huber"},
        {"id": "Trial-6 (Large Batch)", "layers": (128, 64, 32), "lr": 0.001, "batch": 128, "dropout": 0.2, "opt": "adam", "loss": "huber"},
        {"id": "Trial-7 (RMSprop Opt)", "layers": (128, 64, 32), "lr": 0.0005, "batch": 64, "dropout": 0.2, "opt": "rmsprop", "loss": "huber"},
        {"id": "Trial-8 (High Reg)", "layers": (128, 64, 32), "lr": 0.001, "batch": 64, "dropout": 0.4, "opt": "adam", "loss": "huber"},
    ]
    
    results = []
    print("\n" + "="*80)
    print("RUNNING SYSTEMATIC HYPERPARAMETER TUNING SEARCH FOR DNN")
    print("="*80)
    
    for t in trials:
        print(f"\nEvaluating {t['id']}: Layers={t['layers']}, LR={t['lr']}, Batch={t['batch']}, Dropout={t['dropout']}, Opt={t['opt']}, Loss={t['loss']}")
        train_loader = DataLoader(TensorDataset(t_X_train, t_y_train), batch_size=t["batch"], shuffle=True)
        
        model = ConfigurableDNN(len(feat_cols), t["layers"], dropout=t["dropout"]).to(device)
        criterion = get_loss_function(t["loss"], delta=1.0)
        optimizer = get_optimizer(model, t["opt"], lr=t["lr"])
        
        start_time = time.time()
        best_val_mae = float("inf")
        
        # Train for 10 epochs for tuning evaluation
        for epoch in range(1, 11):
            model.train()
            for b_x, b_y in train_loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                optimizer.zero_grad()
                out = model(b_x)
                loss = criterion(out, b_y)
                loss.backward()
                optimizer.step()
                
        metrics = evaluate_model_tensor(model, X_val, y_val, device=device)
        elapsed = time.time() - start_time
        print(f"  Result -> Val MAE: ${metrics['MAE']:.2f} | Val RMSE: ${metrics['RMSE']:.2f} | Val R2: {metrics['R2']:.4f} | Time: {elapsed:.1f}s")
        
        results.append({
            "Trial": t["id"],
            "Layers": str(t["layers"]),
            "Learning_Rate": t["lr"],
            "Batch_Size": t["batch"],
            "Dropout": t["dropout"],
            "Optimizer": t["opt"].upper(),
            "Loss_Function": t["loss"].upper(),
            "Val_MAE": metrics["MAE"],
            "Val_RMSE": metrics["RMSE"],
            "Val_R2": metrics["R2"],
            "Duration_Sec": round(elapsed, 1)
        })
        
    df_res = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "hyperparameter_tuning_results.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\n[Tuning] Complete results saved to {out_path}")
    print(df_res.to_string())
    return df_res

if __name__ == "__main__":
    run_tuning_grid()
