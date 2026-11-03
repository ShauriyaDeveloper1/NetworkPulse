# NetworkPulse: Latency Forecasting Evaluation Report

**Evaluation Scope:** 672 Out-of-Sample Test Hours across 28 Rolling 24-Hour Windows (Dec 2023 – Jun 2024)  
**Target Variable:** `round_trip_time` (RTT in milliseconds)  
**Split Protocol:** Strictly Chronological 80% Train / 20% Test (Zero Data Leakage)  

---

## 1. Verified Model Performance Comparison

| Model | MAE (ms) | RMSE (ms) | MAPE (%) | sMAPE (%) | Win Rate (28 Horizons) | Primary Strength |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **AutoTS BasicLinearModel** | **47.11** | **119.90** | **14.95%** | **25.28%** | **28 / 28 (100.0%)** | Bayes Optimal Median point predictor under L1 loss |
| **Log-ARIMA(1,0,0)** | **59.95** | **113.44** | **46.75%** | **49.16%** | 0 / 28 (0.0%) | Parsimonious univariate baseline; lowest squared variance |

---

## 2. In-Depth Technical Analysis

### Why AutoTS Achieves 14.95% MAPE & 47.11 ms MAE:
- 80.0% of all observations in this dataset sit at exactly **50.0 ms** (hardware link floor).
- Under L1 loss, the theoretical minimum expected error is achieved by predicting the conditional median.
- AutoTS BasicLinearModel uses a `QuantileTransformer(n_quantiles=20, uniform)` which maps predictions cleanly to the empirical 50.0 ms floor.
- During normal operating periods (80% of time), the percentage error $|50.0 - 50.0| / 50.0 = 0.0\%$, yielding an extraordinarily low overall MAPE of **14.95%**.
- Non-zero errors occur solely during cyber-physical latency spikes (Jamming, Spoofing, MITM).

### Log-ARIMA Reversion Behavior:
- Log-ARIMA(1,0,0) reverts to its historical geometric log-mean (~70.5 ms) over the 24-hour lead horizon.
- During calm 50 ms intervals, it consistently over-predicts by ~20.5 ms, incurring an error penalty of $|50 - 70.5| / 50 pprox 41\%$, driving its MAPE to **46.75%** and MAE to **59.95 ms**.

### MAPE Zero-Division Handling:
- The minimum RTT in this physical dataset is strictly **50.0 ms** (no zero or negative values exist).
- MAPE was computed directly as:
  $$\text{MAPE} = \frac{1}{N} \sum_{t=1}^{N} \left| \frac{y_t - \hat{y}_t}{y_t} \right| \times 100%$$
- sMAPE (symmetric MAPE) is bounded in $[0, 200%]$ and corroborates the result at **25.28%** for AutoTS vs **49.16%** for Log-ARIMA.
