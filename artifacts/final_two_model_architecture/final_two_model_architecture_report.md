# NetworkPulse: Final Two-Model Forecasting Architecture Report

**Author:** Senior Machine Learning Engineer & Python Developer  
**Target Variable:** `round_trip_time` (RTT in milliseconds)  
**Optimization Objective:** Minimizing Mean Absolute Error (MAE)  
**Evaluation Scope:** 672 Out-of-Sample Test Hours across 28 Rolling 24-Hour Windows  

---

## 1. Architectural Overview & Model Selection

The NetworkPulse forecasting architecture has been finalized to strictly **two active models**, eliminating all non-essential competing implementations from the operational pipeline:

1. **Model A: Best-MAE ARIMA [Log-ARIMA(1,0,0)]** — High-efficiency univariate parametric forecaster targeting the geometric median under exponential back-transformation.
2. **Model B: Best-MAE XGBoost [Quantile Regressor q=0.46]** — Feature-based gradient boosting forecaster targeting the conditional median with pinball loss and recursive multi-step forecasting.

All competing variants (Squared-Error XGBoost, Balanced Pareto ensembles, quantile ensembles, persistence, seasonal naive, and fixed median) have been decommissioned from active inference while remaining preserved in historical research archives.

---

## 2. Model Selection & Validation Rigor

### Model A: Selection of Best-MAE ARIMA
- **Differencing Decision:** Stationarity analysis via Augmented Dickey-Fuller (ADF) yields test statistic **-136.3510** ($p < 0.0001$), firmly rejecting a unit root. Differencing $d=1$ constitutes harmful over-differencing.
- **Log Transformation Justification:** Untransformed ARIMA models fit Gaussian errors in levels, pinning forecasts to the conditional mean (~100 ms) and yielding validation MAE of **76.45 ms**. Log transformation maps errors multiplicatively, allowing naive exponential inversion to target the conditional median (50 ms).
- **Order Selection:** Validation grid search established **Log-ARIMA(1,0,0)** at **55.83 ms validation MAE**, outperforming all differenced models while matching the 2-parameter Log-ARIMA(1,0,1) (55.83 ms) with maximum parsimony.

### Model B: Selection of Best-MAE XGBoost
- **Quantile Tuning Grid Search (Chronological Validation Only):**
  * $q=0.46$: **47.40 ms MAE** (Selected Champion)
  * $q=0.47$: **47.41 ms MAE**
  * $q=0.48$: **47.42 ms MAE**
  * $q=0.49$: **47.42 ms MAE**
  * $q=0.50$: **47.43 ms MAE**
  * Standard L1 Loss: **51.14 ms MAE**
- **Loss Function Dynamics:** Quantile alpha=0.46 penalizes positive over-predictions slightly more than under-predictions, preventing false spike overshoots during nominal calm periods (80% of data at 50 ms).

---

## 3. Verified Performance Benchmark (672 Test Hours)

| Metric | Model A: Log-ARIMA(1,0,0) | Model B: XGBoost Quantile (q=0.46) | Delta (XGBoost vs ARIMA) |
| :--- | :---: | :---: | :---: |
| **Validation MAE (Chronological)** | **55.83 ms** | **47.40 ms** | **-8.43 ms (-15.1%)** |
| **Validation RMSE (Chronological)** | **104.59 ms** | **113.77 ms** | +9.18 ms |
| **Primary 24-Hour Test MAE** | **70.73 ms** | **58.06 ms** | **-12.67 ms (-17.9%)** |
| **Primary 24-Hour Test RMSE** | **127.95 ms** | **135.69 ms** | +7.74 ms |
| **Multi-Week Micro MAE (Pooled 672h)** | **60.79 ms** | **47.85 ms** | **-12.93 ms (-21.3%)** |
| **Multi-Week Micro RMSE (Pooled 672h)** | **114.70 ms** | **121.21 ms** | +6.51 ms |
| **Multi-Week Micro MSE (Pooled 672h)** | **13,156.35** | **14,691.17** | +1,534.82 |
| **Multi-Week Macro MAE (Window Mean)** | **60.79 ms** $\pm 15.75$ | **47.85 ms** $\pm 18.43$ | **-12.93 ms (-21.3%)** |
| **Windows Won (out of 28)** | **6 / 28 (21.4%)** | **22 / 28 (78.6%)** | XGBoost wins 78.6% of horizons |

---

## 4. Key Engineering & Physical Insights

1. **XGBoost Dominates on MAE:** Across all 28 rolling test windows, XGBoost Quantile ($q=0.46$) achieves **47.85 ms pooled MAE**, reducing error by **12.93 ms (-21.3%)** compared to Log-ARIMA (**60.79 ms**).
2. **ARIMA Retains Superior RMSE/MSE:** Because Log-ARIMA has a smoother, mean-reverting tail behavior, it achieves **114.70 ms RMSE** vs **121.21 ms** for XGBoost, demonstrating the fundamental trade-off between L1 median-targeting and L2 squared-error penalties.
3. **Strict Information Availability:** Both models rely strictly on historical information available up to the forecast origin $t_0$. In recursive 24-step forecasting, predictions $\hat{y}_{t+h}$ dynamically update lags and rolling stats without looking ahead.

---

## 5. Artifact Directory & Verification Deliverables

All deliverables have been saved to [`artifacts/final_two_model_architecture/`](file:///C:/Users/lenovo/Downloads/NetworkPulse/artifacts/final_two_model_architecture/):
- [`two_model_comparison_predictions.csv`](file:///C:/Users/lenovo/Downloads/NetworkPulse/artifacts/final_two_model_architecture/two_model_comparison_predictions.csv)
- [`two_model_daily_metrics.csv`](file:///C:/Users/lenovo/Downloads/NetworkPulse/artifacts/final_two_model_architecture/two_model_daily_metrics.csv)
- [`two_model_summary_metrics.csv`](file:///C:/Users/lenovo/Downloads/NetworkPulse/artifacts/final_two_model_architecture/two_model_summary_metrics.csv)
- Plots: `two_model_actual_vs_pred_24h.png`, `two_model_multi_window_mae_comparison.png`, `two_model_residual_diagnostics.png`
