"""
train_model.py
--------------
Standalone training script.  Run once before launching the Streamlit app:

    python train_model.py

What it does:
  1. Loads and preprocesses energy + weather data.
  2. Trains XGBoost and Random Forest models.
  3. Evaluates both and prints a comparison table.
  4. Saves models to models/ directory.
  5. Trains and saves an Isolation Forest for anomaly detection.
"""

import sys
import os

# Ensure project root is on the path (works when run from any directory)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import time

from modules.preprocessing import build_feature_matrix, get_feature_columns
from modules.forecast import train_models
from modules.anomaly_detection import train_isolation_forest
from modules.preprocessing import load_energy_data


def print_banner():
    print("\n" + "=" * 60)
    print("   Smart Grid AI — Model Training Script")
    print("=" * 60 + "\n")


def print_metrics_table(name: str, metrics: dict):
    print(f"\n  ── {name} ──────────────────────────────────")
    for k, v in metrics.items():
        print(f"    {k:8s}: {v}")


def main():
    print_banner()

    # ── Step 1: Build feature matrix ────────────────────────────────────────
    print("[1/3] Building feature matrix …")
    t0 = time.time()
    df = build_feature_matrix()
    print(f"      → {len(df):,} rows × {df.shape[1]} columns  ({time.time()-t0:.1f}s)")

    # ── Step 2: Train forecasting models ────────────────────────────────────
    print("\n[2/3] Training forecasting models …")
    results = train_models(test_size=0.2)

    print_metrics_table("XGBoost",       results["xgboost"]["metrics"])
    print_metrics_table("Random Forest", results["random_forest"]["metrics"])

    # Highlight winner
    xgb_r2 = results["xgboost"]["metrics"]["R2"]
    rf_r2  = results["random_forest"]["metrics"]["R2"]
    winner = "XGBoost" if xgb_r2 >= rf_r2 else "Random Forest"
    print(f"\n  ✓ Best model: {winner} (R² = {max(xgb_r2, rf_r2):.4f})")

    # ── Step 3: Train anomaly detector ──────────────────────────────────────
    print("\n[3/3] Training Isolation Forest (anomaly detection) …")
    energy_df = load_energy_data()
    energy_df = energy_df.dropna(subset=["consumption_kwh"])

    feature_cols = ["consumption_kwh", "hour", "weekday"]
    feature_cols = [c for c in feature_cols if c in energy_df.columns]

    model, scaler = train_isolation_forest(energy_df, feature_cols, save=True)
    print(f"      → Trained on {len(energy_df):,} rows.")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training complete!  Saved files:")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    for f in os.listdir(models_dir):
        size_kb = os.path.getsize(os.path.join(models_dir, f)) // 1024
        print(f"    models/{f}  ({size_kb} KB)")
    print("\n  Run the dashboard with:")
    print("    streamlit run app.py\n")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
