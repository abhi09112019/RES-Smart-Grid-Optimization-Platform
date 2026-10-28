"""
forecast.py
-----------
Load forecasting using XGBoost (primary) and Random Forest (comparison).
Trains, evaluates, and saves models.  Provides predict helpers used by the
Streamlit dashboard.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import os

from modules.preprocessing import (
    build_feature_matrix,
    get_feature_columns,
    scale_features,
)

# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

XGB_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
RF_PATH  = os.path.join(MODEL_DIR, "rf_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_xgboost(n_estimators: int = 300, max_depth: int = 5,
                  learning_rate: float = 0.05) -> xgb.XGBRegressor:
    """Instantiate XGBoost with paper-inspired hyperparameters."""
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )


def build_random_forest(n_estimators: int = 200) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "R2": round(r2, 4), "WAPE": round(wape, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# Training entry-point
# ─────────────────────────────────────────────────────────────────────────────

def train_models(test_size: float = 0.2) -> dict:
    """
    Full train/test pipeline.
    Returns a dict with both models and their evaluation metrics.
    """
    print("[forecast] Building feature matrix …")
    df = build_feature_matrix()
    target = "consumption_kwh"
    features = get_feature_columns(df, target)

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # ── XGBoost ──────────────────────────────────────────────────────────────
    print("[forecast] Training XGBoost …")
    xgb_model = build_xgboost()
    xgb_model.fit(X_train, y_train)
    xgb_pred  = xgb_model.predict(X_test)
    xgb_metrics = evaluate_model(y_test, xgb_pred)
    joblib.dump({"model": xgb_model, "features": features}, XGB_PATH)
    print(f"  XGBoost  → {xgb_metrics}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("[forecast] Training Random Forest …")
    rf_model  = build_random_forest()
    rf_model.fit(X_train, y_train)
    rf_pred   = rf_model.predict(X_test)
    rf_metrics = evaluate_model(y_test, rf_pred)
    joblib.dump({"model": rf_model, "features": features}, RF_PATH)
    print(f"  RandomForest → {rf_metrics}")

    return {
        "xgboost": {"model": xgb_model, "metrics": xgb_metrics,
                    "y_test": y_test, "y_pred": xgb_pred,
                    "timestamps": df["timestamp"].iloc[-len(y_test):].values},
        "random_forest": {"model": rf_model, "metrics": rf_metrics,
                          "y_test": y_test, "y_pred": rf_pred},
        "feature_names": features,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers (used by dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def load_xgboost_model():
    """Load saved XGBoost bundle {model, features}."""
    if not os.path.exists(XGB_PATH):
        raise FileNotFoundError(
            "XGBoost model not found. Run train_model.py first."
        )
    return joblib.load(XGB_PATH)


def predict_next_hours(n_hours: int = 24) -> pd.DataFrame:
    """
    Generate a rolling n-hour forecast using the trained XGBoost model.
    Uses the last available rows of the dataset as seed.
    """
    bundle = load_xgboost_model()
    model: xgb.XGBRegressor = bundle["model"]
    feature_names: list = bundle["features"]

    df = build_feature_matrix()
    # Use the last available window as seed
    seed = df.tail(200).copy()

    forecasts = []
    last_ts = seed["timestamp"].max()

    for i in range(1, n_hours + 1):
        next_ts = last_ts + pd.Timedelta(hours=i)
        row = seed.tail(1).copy()
        row["timestamp"] = next_ts
        row["hour"]       = next_ts.hour
        row["day"]        = next_ts.day
        row["month"]      = next_ts.month
        row["weekday"]    = next_ts.weekday()
        row["is_weekend"] = int(next_ts.weekday() >= 5)
        row["hour_sin"]   = np.sin(2 * np.pi * next_ts.hour / 24)
        row["hour_cos"]   = np.cos(2 * np.pi * next_ts.hour / 24)
        row["month_sin"]  = np.sin(2 * np.pi * next_ts.month / 12)
        row["month_cos"]  = np.cos(2 * np.pi * next_ts.month / 12)

        X_row = row[feature_names].values.reshape(1, -1)
        pred  = float(model.predict(X_row)[0])
        pred  = max(0.0, pred)

        # Propagate prediction as the lag-1 for next step
        row["consumption_kwh"] = pred
        seed = pd.concat([seed, row], ignore_index=True)

        forecasts.append({"timestamp": next_ts, "forecast_kwh": round(pred, 4)})

    return pd.DataFrame(forecasts)


def get_feature_importances(top_n: int = 15) -> pd.DataFrame:
    """Return top-N feature importances from XGBoost."""
    bundle = load_xgboost_model()
    model: xgb.XGBRegressor = bundle["model"]
    names = bundle["features"]
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"feature": names, "importance": importances})
    fi_df.sort_values("importance", ascending=False, inplace=True)
    return fi_df.head(top_n).reset_index(drop=True)
