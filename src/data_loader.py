"""
Data Loader module for NYC Taxi Fare Prediction.
Downloads and caches an authentic sample (100,000 rows) of the official
Kaggle New York City Taxi Fare Prediction dataset.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import urllib.request
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RAW_CSV_PATH = os.path.join(RAW_DATA_DIR, "train_sample_100k.csv")

DATA_URL = "https://huggingface.co/datasets/JosephFeig/NYC-Taxi/resolve/main/train.csv"

def download_taxi_data(target_rows: int = 100000, force_download: bool = False) -> str:
    """
    Downloads an authentic sample of the Kaggle NYC Taxi Fare training dataset.
    Uses streaming chunked reading to efficiently fetch the exact number of rows needed.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    if os.path.exists(RAW_CSV_PATH) and not force_download:
        file_size = os.path.getsize(RAW_CSV_PATH) / (1024 * 1024)
        print(f"[DataLoader] Raw dataset already exists at: {RAW_CSV_PATH} ({file_size:.2f} MB)")
        return RAW_CSV_PATH
        
    print(f"[DataLoader] Streaming {target_rows:,} authentic rows from {DATA_URL}...")
    req = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    
    rows_written = 0
    buffer = ""
    chunk_size = 64 * 1024  # 64 KB
    
    with urllib.request.urlopen(req, timeout=60) as response, open(RAW_CSV_PATH, "w", encoding="utf-8") as out_file:
        header_written = False
        while rows_written < target_rows:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            text = buffer + chunk.decode("utf-8", errors="ignore")
            lines = text.split("\n")
            buffer = lines[-1]  # Incomplete line
            
            for line in lines[:-1]:
                line_str = line.strip()
                if not line_str:
                    continue
                if not header_written:
                    out_file.write(line_str + "\n")
                    header_written = True
                else:
                    out_file.write(line_str + "\n")
                    rows_written += 1
                    if rows_written % 20000 == 0:
                        print(f"  Downloaded {rows_written:,} / {target_rows:,} rows...")
                    if rows_written >= target_rows:
                        break
                        
    file_size = os.path.getsize(RAW_CSV_PATH) / (1024 * 1024)
    print(f"[DataLoader] Download complete: {rows_written:,} rows saved to {RAW_CSV_PATH} ({file_size:.2f} MB)")
    return RAW_CSV_PATH

def load_raw_dataset(nrows: int = None) -> pd.DataFrame:
    """Loads raw dataset into a Pandas DataFrame."""
    if not os.path.exists(RAW_CSV_PATH):
        download_taxi_data()
    print(f"[DataLoader] Loading raw dataset from {RAW_CSV_PATH}...")
    df = pd.read_csv(RAW_CSV_PATH, nrows=nrows)
    print(f"[DataLoader] Loaded DataFrame with shape: {df.shape}")
    return df

if __name__ == "__main__":
    download_taxi_data(100000)
    df = load_raw_dataset(5)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
