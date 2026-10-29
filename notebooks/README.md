# NetworkPulse: Two-Model Architecture Interactive Notebooks

This directory contains standalone, fully executed Jupyter Notebooks (`.ipynb`) with pre-rendered visual charts, residual diagnostic plots, and metrics tables for the **final two MAE-optimized forecasting models**:

---

## Notebook Catalog

### 1. [`01_NetworkPulse_Best_MAE_ARIMA.ipynb`](file:///c:/Users/lenovo/Downloads/NetworkPulse/notebooks/01_NetworkPulse_Best_MAE_ARIMA.ipynb)
- **Model:** Model A — Log-ARIMA(1,0,0)
- **Mathematical Foundation:**
  - Augmented Dickey-Fuller (ADF) test proves level stationarity ($p < 0.0001$), confirming $d=0$ differencing.
  - Log-transformation with naive exponential inversion $\hat{y} = \exp(\hat{z})$ targets the conditional geometric median (~50 ms).
  - Bounded strictly within physical telemetry limits $[50.0, 500.0]$ ms.
- **Key Performance:**
  - Chronological Validation MAE: **55.83 ms**
  - Primary 24-Hour Horizon MAE: **70.73 ms**
  - 672-Hour Multi-Week Benchmark MAE: **59.95 ms** | RMSE: **113.44 ms**

---

### 2. [`02_NetworkPulse_Best_MAE_XGBoost.ipynb`](file:///c:/Users/lenovo/Downloads/NetworkPulse/notebooks/02_NetworkPulse_Best_MAE_XGBoost.ipynb)
- **Model:** Model B — XGBoost Quantile Regressor ($q=0.46$)
- **Feature & Training Design:**
  - 82 causal features (strictly shifted lags $1, 2, 3, 6, 12, 24$, differences, rolling statistics, EWMAs, and exogenous telemetry).
  - Strictly drops device identifiers and ground-truth attack labels to prevent data leakage.
  - Pre-trained booster loaded from `models/xgboost/best_mae_xgboost_model.json`.
  - Recursive 24-step forecasting with dynamic lag and rolling buffer updates.
- **Key Performance:**
  - Chronological Validation MAE: **47.40 ms** (Selected from $q \in [0.46, 0.50]$)
  - Primary 24-Hour Horizon MAE: **58.06 ms**
  - 672-Hour Multi-Week Benchmark MAE: **47.16 ms** | RMSE: **119.87 ms**
  - **Wins 28 / 28 (100%)** of rolling 24-hour test windows against ARIMA.

---

### 3. [`03_NetworkPulse_Final_Two_Models_Comparison.ipynb`](file:///c:/Users/lenovo/Downloads/NetworkPulse/notebooks/03_NetworkPulse_Final_Two_Models_Comparison.ipynb)
- **Scope:** Direct head-to-head operational benchmark of Model A and Model B.
- **Workflow:**
  - Common interface via `TwoModelForecastingPipeline`.
  - Side-by-side evaluation on identical chronological timestamps and test horizons.
  - Micro-pooled and macro-window metrics tables, residual distributions, and bar charts.
