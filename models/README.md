# NetworkPulse — Models Directory

This directory organizes the forecasting models, serialized weights/configurations, Jupyter notebooks, and driver scripts exclusively for the **Final Two-Model Active Architecture**.

---

## Active Forecasting Architecture: Two-Model Pipeline

To minimize Mean Absolute Error (MAE) without unnecessary complexity, NetworkPulse deploys strictly **two verified models**:

1. **Model A: Best-MAE ARIMA [Log-ARIMA(1,0,0)]**
   - **Type:** Univariate Parametric Time Series
   - **Order:** $(p=1, d=0, q=0)$
   - **Transformation:** $\ln(\text{RTT})$ with naive exponential back-transformation $\hat{y} = \exp(\hat{z})$
   - **Chronological Validation MAE:** **55.83 ms** | Multi-Week Test MAE: **60.79 ms**
   - **Notebook:** [`models/arima/NetworkPulse_Best_MAE_ARIMA.ipynb`](file:///models/arima/NetworkPulse_Best_MAE_ARIMA.ipynb)
   - **Runner Script:** [`models/arima/run_best_mae_arima.py`](file:///models/arima/run_best_mae_arima.py)
   - **Config:** [`models/arima/best_mae_arima_config.json`](file:///models/arima/best_mae_arima_config.json)

2. **Model B: Best-MAE XGBoost [Quantile Regressor $q=0.46$]**
   - **Type:** Feature-Based Gradient Boosting (82 predictive lag, rolling, EWMA & telemetry features)
   - **Objective:** `reg:quantileerror` ($\alpha = 0.46$)
   - **Strategy:** Recursive 24-step forecasting with dynamic lag and rolling statistic updates
   - **Chronological Validation MAE:** **47.40 ms** | Multi-Week Test MAE: **47.16 ms**
   - **Notebook:** [`models/xgboost/NetworkPulse_Best_MAE_XGBoost.ipynb`](file:///models/xgboost/NetworkPulse_Best_MAE_XGBoost.ipynb)
   - **Runner Script:** [`models/xgboost/run_best_mae_xgboost.py`](file:///models/xgboost/run_best_mae_xgboost.py)
   - **Config:** [`models/xgboost/best_mae_xgboost_config.json`](file:///models/xgboost/best_mae_xgboost_config.json)
   - **Artifact:** [`models/xgboost/best_mae_xgboost_model.json`](file:///models/xgboost/best_mae_xgboost_model.json)

All historical research experiments and non-active competing variants (squared-error champions, Pareto ensembles, naive baselines) are preserved in [`artifacts/`](file:///artifacts/).

---

## Directory Structure

```
models/
├── README.md                                  # Directory documentation & model catalog
│
├── arima/                                     # Model A: Best-MAE ARIMA
│   ├── NetworkPulse_Best_MAE_ARIMA.ipynb      # Dedicated interactive notebook for Log-ARIMA(1,0,0)
│   ├── run_best_mae_arima.py                  # Standalone execution & evaluation script
│   └── best_mae_arima_config.json             # Model A configuration & validation metadata
│
└── xgboost/                                   # Model B: Best-MAE XGBoost
    ├── NetworkPulse_Best_MAE_XGBoost.ipynb    # Dedicated interactive notebook for Quantile XGBoost
    ├── run_best_mae_xgboost.py                # Standalone execution & evaluation script
    ├── best_mae_xgboost_config.json           # Model B configuration & validation metadata
    └── best_mae_xgboost_model.json            # Serialized Quantile booster artifact (q=0.46)
```

---

## Execution Quickstart

To run the unified two-model pipeline (both models evaluated head-to-head on identical test origins):

```powershell
python run_final_two_models.py
```

To run individual model driver scripts:

```powershell
# 1. Run Model A (Log-ARIMA)
python models/arima/run_best_mae_arima.py

# 2. Run Model B (Quantile XGBoost)
python models/xgboost/run_best_mae_xgboost.py
```
