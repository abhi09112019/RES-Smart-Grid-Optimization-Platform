# ⚡ AI-Powered Smart Grid Energy Management System

A production-ready intelligent energy management platform that combines **XGBoost load forecasting**, **analytical solar PV modelling**, **battery management**, and **AI-based cost optimisation** — all presented through a professional **Streamlit dashboard**.

Inspired by: _Matos et al. (2024) — "A Machine Learning-Based Electricity Consumption Forecast and Management System for Renewable Energy Communities", Energies, 17, 630._

---

## 📐 Architecture

```
smart_grid_ai/
│
├── app.py                     ← Streamlit dashboard (7 tabs)
├── train_model.py             ← One-time model training script
│
├── data/
│   ├── energy_data.csv        ← Hourly electricity consumption (8760 rows)
│   └── weather_data.csv       ← Temperature, humidity, DNI irradiance
│
├── models/
│   ├── xgboost_model.pkl      ← Trained XGBoost forecaster
│   ├── rf_model.pkl           ← Trained Random Forest (comparison)
│   └── anomaly_model.pkl      ← Isolation Forest anomaly detector
│
├── modules/
│   ├── preprocessing.py       ← Feature engineering pipeline
│   ├── forecast.py            ← XGBoost + RF load forecasting
│   ├── solar_forecast.py      ← Analytical PV generation model
│   ├── battery_manager.py     ← Battery charge/discharge simulation
│   ├── optimization.py        ← ML-based energy management algorithm
│   ├── pricing_engine.py      ← Tariff structures + bill calculator
│   ├── anomaly_detection.py   ← Isolation Forest + Z-score detection
│   └── utils.py               ← Shared helpers
│
├── dashboard/
│   ├── charts.py              ← All Plotly chart builders
│   └── metrics.py             ← KPI card data builders
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Quick Start

### 1. Clone / copy the project
```bash
cd smart_grid_ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the models (one-time, ~30 seconds)
```bash
python train_model.py
```

Expected output:
```
[1/3] Building feature matrix … → 8,592 rows × 29 columns
[2/3] Training XGBoost  → R²: 0.9711, WAPE: 5.89%
      Training RF       → R²: 0.9730, WAPE: 5.72%
[3/3] Training Isolation Forest …
  Training complete! Run: streamlit run app.py
```

### 4. Launch the dashboard
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 📊 Dashboard Pages

| Tab | Description |
|-----|-------------|
| 🏠 **Overview** | KPIs: total demand, solar generation, self-sufficiency, CO₂ saved; energy source pie; configuration summary |
| 📈 **Load Forecast** | XGBoost test predictions vs actual; MAE / RMSE / R² metrics; 24-hour ahead rolling forecast; feature importances |
| ☀️ **Solar Forecast** | Analytical PV model (high/average/low scenarios); DNI vs output chart; daily totals; scenario statistics table |
| 🔋 **Battery Analytics** | SOC over time; grid buy/sell activity; battery scenario comparison table (small / medium / large) |
| ⚙️ **Optimisation** | ML-based algorithm (paper Fig. 6); savings gauge vs baseline; algorithm comparison table |
| 💰 **Pricing** | Tariff price trace; monthly bill bar chart; tariff comparison table (simple, bi-hourly daily/weekly) |
| 🚨 **Anomaly Detection** | Isolation Forest + Z-score flags on consumption series; z-score histogram; top anomaly table |

---

## 🤖 ML Models & Performance

### Load Forecasting (XGBoost)
- **Features:** hour, month, weekday, is_weekend, lag-1/2/3/6/12/24/48/168 h, rolling mean/std (3h/6h/24h), temperature, humidity
- **Training split:** 80% train / 20% test (chronological)
- **Hyperparameters:** 300 estimators, max_depth=5, learning_rate=0.05, subsample=0.8

| Metric | XGBoost | Random Forest |
|--------|---------|---------------|
| MAE    | 0.238 kWh | 0.232 kWh |
| RMSE   | 0.304 kWh | 0.293 kWh |
| R²     | 0.971  | 0.973 |
| WAPE   | 5.89%  | 5.72% |

*Results comparable with REC paper (WAPE ~8.2% on real UK council building data)*

### Solar PV (Analytical Model)
Based on Filik et al. (2017):

```
E_pv = DNI [kWh/m²] × A [m²] × η × PR
```

| Scenario | Panel Area | Efficiency | Mean Output |
|----------|-----------|------------|-------------|
| High     | 28.8 m²   | 16%        | 0.432 kWh/h |
| Average  | 15.8 m²   | 16%        | 0.237 kWh/h |
| Low      | 2.88 m²   | 16%        | 0.043 kWh/h |

### Anomaly Detection (Isolation Forest)
- Contamination: 3%
- Features: consumption_kwh, hour, weekday
- Fallback: IQR fence (Q1 − 2.5×IQR, Q3 + 2.5×IQR)

---

## ⚙️ Energy Management Algorithm

Mirrors the REC paper's ML-based decision algorithm (Figure 6):

```
At midnight:
  forecast day-ahead demand & solar (δ1, δ2 heuristics)
  pre-charge battery from cheap off-peak grid

During day (every hour):
  consume solar production first
  if surplus → charge battery → sell excess to grid
  if deficit (before 10:00) → buy from grid
  if deficit (after 10:00)  → draw from battery first
```

**Cost function** (Equation 1 from paper):
```
cost = consumption_midnight × buy_price_midnight + battery_losses_midnight
     + Σ (consumption_i × buy_price_i − surplus_i × sell_price_i + battery_losses_i)
```

---

## 💰 Tariff Structure

| Tariff | Price |
|--------|-------|
| Simple flat rate | 0.1583 EUR/kWh |
| Bi-hourly peak (09–21h) | 0.1917 EUR/kWh |
| Bi-hourly off-peak | 0.1044 EUR/kWh |
| Grid sell price | 0.03 EUR/kWh (fixed) |

---

## 🔋 Battery Scenarios

| Scenario | Capacity/Member | Total (10 members) |
|----------|----------------|-------------------|
| Small    | 1.6 kWh        | 16 kWh |
| Medium   | 6.4 kWh        | 64 kWh |
| Large    | 12.8 kWh       | 128 kWh |

Inverter efficiency: **97%** · Min SOC: **10%** · Max SOC: **100%**

---

## 🔮 Future Improvements

1. **LSTM forecasting** — sequence-to-sequence model for longer horizons
2. **Real-time MQTT data ingestion** — live smart meter feeds
3. **Multi-building heterogeneous REC** — per-building models aggregated
4. **Dynamic pricing** — intraday spot market integration
5. **EV charging optimisation** — vehicle-to-grid (V2G) scheduling
6. **Hyperparameter optimisation** — Optuna / Bayesian search
7. **PCA / feature selection** — dimensionality reduction for LSTM

---

## 📚 References

- Matos, M. et al. (2024). *A Machine Learning-Based Electricity Consumption Forecast and Management System for Renewable Energy Communities*. Energies, 17, 630.
- Filik, T. et al. (2017). *Solar radiation-to-power generation models for one-axis tracking PV system*. E3S Web Conf., 22.
- Pholboon, S. et al. (2015). *Real-time battery management algorithm for peak demand shaving*. IEEE ISGT LATAM.
- Abbasi, R.A. et al. (2019). *Short Term Load Forecasting Using XGBoost*. WAINA 2019.
