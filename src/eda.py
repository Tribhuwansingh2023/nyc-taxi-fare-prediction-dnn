"""
Exploratory Data Analysis (EDA) module for NYC Taxi Fare Prediction.
Generates comprehensive statistical summaries and high-resolution visualizations
covering univariate distributions, geolocation densities, and temporal ridership patterns.
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
DATASET_CSV_PATH = os.path.join(BASE_DIR, "dataset", "train.csv")
SAMPLE_CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "train_sample_100k.csv")
RAW_CSV_PATH = DATASET_CSV_PATH if os.path.exists(DATASET_CSV_PATH) else SAMPLE_CSV_PATH
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set global plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8

def haversine_np(lon1, lat1, lon2, lat2):
    """Vectorized Haversine distance in kilometers."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    km = 6371.0 * c
    return km

def run_eda():
    print(f"[EDA] Loading raw dataset for EDA from {RAW_CSV_PATH}...")
    if os.path.abspath(RAW_CSV_PATH) == os.path.abspath(DATASET_CSV_PATH):
        df = pd.read_csv(RAW_CSV_PATH, nrows=1000000)
    else:
        df = pd.read_csv(RAW_CSV_PATH)
    total_raw_records = len(df)
    print(f"[EDA] Total raw records: {total_raw_records:,}")
    
    # Save descriptive summary.csv
    try:
        df.describe(include="all").to_csv(os.path.join(BASE_DIR, "summary.csv"))
        print("[EDA] Updated summary.csv saved.")
    except Exception as e:
        print("[EDA] Could not update summary.csv:", e)

    # Parse pickup_datetime
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    
    # Missing values audit
    missing_counts = df.isnull().sum().to_dict()
    print(f"[EDA] Missing values: {missing_counts}")
    
    # Detect raw anomalies
    neg_fares = int((df["fare_amount"] < 0).sum())
    zero_fares = int((df["fare_amount"] == 0).sum())
    low_fares = int(((df["fare_amount"] > 0) & (df["fare_amount"] < 2.5)).sum())
    extreme_fares = int((df["fare_amount"] > 250).sum())
    invalid_passengers = int(((df["passenger_count"] <= 0) | (df["passenger_count"] > 6)).sum())
    
    # NYC Bounding Box: Lat 40.50 to 40.95, Lon -74.25 to -73.70
    coord_outliers = int((
        (df["pickup_latitude"] < 40.50) | (df["pickup_latitude"] > 40.95) |
        (df["pickup_longitude"] < -74.25) | (df["pickup_longitude"] > -73.70) |
        (df["dropoff_latitude"] < 40.50) | (df["dropoff_latitude"] > 40.95) |
        (df["dropoff_longitude"] < -74.25) | (df["dropoff_longitude"] > -73.70)
    ).sum())
    
    # Filter a clean subset for visualization to inspect realistic NYC trips
    valid_mask = (
        (df["fare_amount"] >= 2.5) & (df["fare_amount"] <= 150) &
        (df["passenger_count"] >= 1) & (df["passenger_count"] <= 6) &
        (df["pickup_latitude"].between(40.50, 40.95)) &
        (df["pickup_longitude"].between(-74.25, -73.70)) &
        (df["dropoff_latitude"].between(40.50, 40.95)) &
        (df["dropoff_longitude"].between(-74.25, -73.70))
    )
    df_clean = df[valid_mask].copy()
    
    df_clean["hour"] = df_clean["pickup_datetime"].dt.hour
    df_clean["day_of_week"] = df_clean["pickup_datetime"].dt.day_name()
    df_clean["day_of_week_num"] = df_clean["pickup_datetime"].dt.dayofweek
    df_clean["month"] = df_clean["pickup_datetime"].dt.month
    df_clean["year"] = df_clean["pickup_datetime"].dt.year
    df_clean["distance_km"] = haversine_np(
        df_clean["pickup_longitude"], df_clean["pickup_latitude"],
        df_clean["dropoff_longitude"], df_clean["dropoff_latitude"]
    )
    
    clean_records = len(df_clean)
    retention_pct = (clean_records / total_raw_records) * 100
    print(f"[EDA] Clean records: {clean_records:,} ({retention_pct:.2f}% retention)")

    # -------------------------------------------------------------
    # 1. Fare Amount Distribution
    # -------------------------------------------------------------
    print("[EDA] Plotting 01_fare_distribution.png...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(df_clean["fare_amount"], bins=60, kde=True, color="#2563EB", ax=ax1, edgecolor="white")
    ax1.set_title("Distribution of NYC Taxi Fare Amount ($)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Fare Amount ($)", fontsize=11)
    ax1.set_ylabel("Trip Frequency", fontsize=11)
    ax1.axvline(df_clean["fare_amount"].median(), color="#DC2626", linestyle="--", linewidth=1.5, label=f"Median: ${df_clean['fare_amount'].median():.2f}")
    ax1.axvline(df_clean["fare_amount"].mean(), color="#059669", linestyle="-.", linewidth=1.5, label=f"Mean: ${df_clean['fare_amount'].mean():.2f}")
    ax1.legend(frameon=True, facecolor="white")
    
    sns.boxplot(x=df_clean["fare_amount"], color="#93C5FD", ax=ax2, fliersize=2)
    ax2.set_title("Boxplot of Taxi Fare Amount (Showing Outliers)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Fare Amount ($)", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "01_fare_distribution.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 2. Passenger Count Distribution
    # -------------------------------------------------------------
    print("[EDA] Plotting 02_passenger_count_distribution.png...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    counts = df_clean["passenger_count"].value_counts().sort_index()
    palette = ["#1E40AF", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE", "#DBEAFE"]
    bars = ax1.bar(counts.index, counts.values, color=palette[:len(counts)], edgecolor="#1E3A8A")
    ax1.set_title("Passenger Count Frequency Distribution", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Number of Passengers", fontsize=11)
    ax1.set_ylabel("Count of Trips", fontsize=11)
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f"{height:,}\n({height/len(df_clean)*100:.1f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)
                     
    sns.boxplot(x="passenger_count", y="fare_amount", data=df_clean[df_clean["fare_amount"] <= 50],
                palette="Blues_r", ax=ax2, fliersize=1)
    ax2.set_title("Fare Amount vs Passenger Count (Fare <= $50)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Passenger Count", fontsize=11)
    ax2.set_ylabel("Fare Amount ($)", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "02_passenger_count_distribution.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. NYC Pickup Density Map
    # -------------------------------------------------------------
    print("[EDA] Plotting 03_nyc_pickup_density.png...")
    plt.figure(figsize=(10, 10))
    sample_geo = df_clean.sample(min(30000, len(df_clean)), random_state=42)
    plt.scatter(sample_geo["pickup_longitude"], sample_geo["pickup_latitude"],
                s=0.8, alpha=0.3, color="#1D4ED8")
    
    # Annotate key NYC landmarks
    landmarks = {
        "Midtown Manhattan": (-73.9855, 40.7580),
        "Lower Manhattan / Wall St": (-74.0090, 40.7050),
        "JFK Airport": (-73.7781, 40.6413),
        "LaGuardia Airport (LGA)": (-73.8740, 40.7769),
        "Central Park": (-73.9665, 40.7812)
    }
    for name, (lon, lat) in landmarks.items():
        plt.scatter(lon, lat, color="#DC2626", s=60, marker="o", edgecolors="white", linewidth=1.5, zorder=5)
        plt.text(lon + 0.008, lat + 0.005, name, fontsize=9, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.8, ec="black", lw=0.5))
                 
    plt.title("Spatial Density of NYC Taxi Pickups", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Longitude", fontsize=11)
    plt.ylabel("Latitude", fontsize=11)
    plt.xlim(-74.10, -73.75)
    plt.ylim(40.60, 40.88)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "03_nyc_pickup_density.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. NYC Dropoff Density Map
    # -------------------------------------------------------------
    print("[EDA] Plotting 04_nyc_dropoff_density.png...")
    plt.figure(figsize=(10, 10))
    plt.scatter(sample_geo["dropoff_longitude"], sample_geo["dropoff_latitude"],
                s=0.8, alpha=0.3, color="#047857")
    for name, (lon, lat) in landmarks.items():
        plt.scatter(lon, lat, color="#DC2626", s=60, marker="o", edgecolors="white", linewidth=1.5, zorder=5)
        plt.text(lon + 0.008, lat + 0.005, name, fontsize=9, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.8, ec="black", lw=0.5))
                 
    plt.title("Spatial Density of NYC Taxi Dropoffs", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Longitude", fontsize=11)
    plt.ylabel("Latitude", fontsize=11)
    plt.xlim(-74.10, -73.75)
    plt.ylim(40.60, 40.88)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "04_nyc_dropoff_density.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 5. Pickup vs Dropoff Geolocation Comparison
    # -------------------------------------------------------------
    print("[EDA] Plotting 05_pickup_vs_dropoff_geo.png...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.scatter(sample_geo["pickup_longitude"], sample_geo["pickup_latitude"], s=0.7, alpha=0.25, color="#2563EB")
    ax1.set_title("Pickup Locations Heatmap / Scatter", fontsize=12, fontweight="bold")
    ax1.set_xlim(-74.05, -73.75)
    ax1.set_ylim(40.60, 40.88)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    
    ax2.scatter(sample_geo["dropoff_longitude"], sample_geo["dropoff_latitude"], s=0.7, alpha=0.25, color="#10B981")
    ax2.set_title("Drop-off Locations Heatmap / Scatter", fontsize=12, fontweight="bold")
    ax2.set_xlim(-74.05, -73.75)
    ax2.set_ylim(40.60, 40.88)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "05_pickup_vs_dropoff_geo.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 6. Hourly Ridership Patterns
    # -------------------------------------------------------------
    print("[EDA] Plotting 06_hourly_ridership_patterns.png...")
    hourly = df_clean.groupby("hour").agg(
        trip_count=("fare_amount", "count"),
        mean_fare=("fare_amount", "mean"),
        mean_distance=("distance_km", "mean")
    ).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    
    b1 = ax1.bar(hourly["hour"], hourly["trip_count"], color="#3B82F6", alpha=0.75, width=0.6, label="Trip Volume")
    l1 = ax2.plot(hourly["hour"], hourly["mean_fare"], color="#DC2626", marker="o", linewidth=2.5, label="Mean Fare ($)")
    
    ax1.set_title("NYC Taxi Ridership Volume and Average Fare by Hour of Day", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Hour of Day (24-Hour Format)", fontsize=11)
    ax1.set_ylabel("Trip Volume (Count)", fontsize=11, color="#1E40AF")
    ax2.set_ylabel("Average Fare ($)", fontsize=11, color="#B91C1C")
    ax1.set_xticks(range(24))
    
    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", frameon=True, facecolor="white")
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "06_hourly_ridership_patterns.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 7. Day of Week Ridership
    # -------------------------------------------------------------
    print("[EDA] Plotting 07_day_of_week_ridership.png...")
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = df_clean.groupby("day_of_week").agg(
        trip_count=("fare_amount", "count"),
        mean_fare=("fare_amount", "mean")
    ).reindex(days_order).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()
    
    colors_dow = ["#60A5FA" if d not in ["Friday", "Saturday"] else "#F59E0B" for d in days_order]
    ax1.bar(daily["day_of_week"], daily["trip_count"], color=colors_dow, width=0.55, edgecolor="#1E3A8A")
    ax2.plot(daily["day_of_week"], daily["mean_fare"], color="#DC2626", marker="s", linewidth=2.5)
    
    ax1.set_title("Trip Volume and Mean Fare Across Days of the Week", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Day of Week", fontsize=11)
    ax1.set_ylabel("Total Trip Count", fontsize=11)
    ax2.set_ylabel("Mean Fare ($)", fontsize=11, color="#DC2626")
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "07_day_of_week_ridership.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 8. Heatmap of Day of Week x Hour
    # -------------------------------------------------------------
    print("[EDA] Plotting 08_hourly_day_heatmap.png...")
    pivot_trips = df_clean.pivot_table(
        index="day_of_week",
        columns="hour",
        values="fare_amount",
        aggfunc="count"
    ).reindex(days_order)
    
    plt.figure(figsize=(15, 6))
    sns.heatmap(pivot_trips, cmap="YlGnBu", annot=False, fmt="d", cbar_kws={"label": "Trip Frequency"})
    plt.title("Ridership Heatmap: Day of Week vs. Hour of Day (Peak Demand Matrix)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Hour of Day (0 - 23)", fontsize=11)
    plt.ylabel("Day of Week", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "08_hourly_day_heatmap.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 9. Distance vs Fare Amount
    # -------------------------------------------------------------
    print("[EDA] Plotting 09_distance_vs_fare.png...")
    plt.figure(figsize=(10, 6))
    dist_sample = df_clean[(df_clean["distance_km"] <= 35) & (df_clean["fare_amount"] <= 100)].sample(
        min(15000, len(df_clean)), random_state=42
    )
    sns.scatterplot(x="distance_km", y="fare_amount", data=dist_sample, alpha=0.3, color="#2563EB", s=15)
    
    # Linear fit line
    sns.regplot(x="distance_km", y="fare_amount", data=dist_sample, scatter=False,
                color="#DC2626", line_kws={"linewidth": 2.5, "label": "Linear Fit Trend"})
                
    plt.title("Relationship Between Haversine Distance (km) and Taxi Fare ($)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Haversine Distance (km)", fontsize=11)
    plt.ylabel("Taxi Fare Amount ($)", fontsize=11)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "09_distance_vs_fare.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 10. Correlation Heatmap
    # -------------------------------------------------------------
    print("[EDA] Plotting 10_feature_correlation_heatmap.png...")
    num_cols = ["fare_amount", "passenger_count", "distance_km", "hour", "day_of_week_num", "month", "year"]
    corr_matrix = df_clean[num_cols].corr()
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                linewidths=0.5, cbar_kws={"label": "Pearson Correlation (r)"})
    plt.title("Feature Pearson Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "10_feature_correlation_heatmap.png"), dpi=300)
    plt.close()

    # Save summary stats to JSON
    summary = {
        "total_raw_records": total_raw_records,
        "clean_records": clean_records,
        "retention_percentage": round(retention_pct, 2),
        "missing_counts": missing_counts,
        "anomalies": {
            "negative_fares": neg_fares,
            "zero_fares": zero_fares,
            "low_fares_under_2_50": low_fares,
            "extreme_fares_over_250": extreme_fares,
            "invalid_passengers": invalid_passengers,
            "geographical_outliers": coord_outliers
        },
        "fare_stats": {
            "mean": round(float(df_clean["fare_amount"].mean()), 2),
            "median": round(float(df_clean["fare_amount"].median()), 2),
            "std": round(float(df_clean["fare_amount"].std()), 2),
            "min": round(float(df_clean["fare_amount"].min()), 2),
            "max": round(float(df_clean["fare_amount"].max()), 2),
            "p25": round(float(df_clean["fare_amount"].quantile(0.25)), 2),
            "p75": round(float(df_clean["fare_amount"].quantile(0.75)), 2)
        },
        "distance_stats": {
            "mean_km": round(float(df_clean["distance_km"].mean()), 2),
            "median_km": round(float(df_clean["distance_km"].median()), 2),
            "std_km": round(float(df_clean["distance_km"].std()), 2),
            "max_km": round(float(df_clean["distance_km"].max()), 2)
        },
        "passenger_distribution": {int(k): int(v) for k, v in counts.items()}
    }
    
    with open(os.path.join(RESULTS_DIR, "eda_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"[EDA] EDA execution complete! Summary saved to results/eda_summary.json")
    return summary

if __name__ == "__main__":
    run_eda()
