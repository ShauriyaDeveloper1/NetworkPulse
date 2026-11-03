# NetworkPulse: Interactive Research Notebooks

This directory contains standalone Jupyter Notebooks (`.ipynb`) with visual charts, residual diagnostic plots, and metrics tables for NetworkPulse forecasting:

---

## Notebook Catalog

### 1. [`01_NetworkPulse_Best_MAE_ARIMA.ipynb`](file:///c:/Users/lenovo/Downloads/NetworkPulse/notebooks/01_NetworkPulse_Best_MAE_ARIMA.ipynb)
- **Model:** Baseline Model — Log-ARIMA(1,0,0)
- **Mathematical Foundation:**
  - Augmented Dickey-Fuller (ADF) test proves level stationarity ($p < 0.0001$), confirming $d=0$ differencing.
  - Log-transformation with naive exponential inversion $\hat{y} = \exp(\hat{z})$ targets the conditional geometric median (~50 ms).
  - Bounded strictly within physical telemetry limits $[50.0, 500.0]$ ms.
- **Key Performance:**
  - Chronological Validation MAE: **55.83 ms**
  - Primary 24-Hour Horizon MAE: **70.73 ms**
  - 672-Hour Multi-Week Benchmark MAE: **59.95 ms** | RMSE: **113.44 ms**
