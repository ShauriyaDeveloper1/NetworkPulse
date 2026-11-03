# NetworkPulse — Models Directory

This directory organizes the model configurations, serialized weights, and metadata for the **NetworkPulse Active Production Architecture**.

---

## Active Model Architecture

NetworkPulse deploys three complementary models:

1. **Baseline Forecaster: Best-MAE ARIMA [Log-ARIMA(1,0,0)]**
   - **Type:** Univariate Parametric Time Series
   - **Order:** $(p=1, d=0, q=0)$
   - **Transformation:** $\ln(\text{RTT})$ with naive exponential back-transformation $\hat{y} = \exp(\hat{z})$
   - **Test Performance:** MAE: **59.95 ms** | RMSE: **113.44 ms** | MAPE: **46.75%**
   - **Config:** `models/arima/best_mae_arima_config.json`
   - **Artifacts:** `models/arima/arima_model.pkl`

2. **Advanced Forecaster: AutoTS Champion [BasicLinearModel]**
   - **Type:** Multi-Scale Automated Time Series Regressor
   - **Transformation:** `QuantileTransformer(20, uniform)` + `CointegrationTransformer(2)` + Wavelet harmonics
   - **Test Performance:** MAE: **47.11 ms** | RMSE: **119.90 ms** | MAPE: **14.95%** (Bayes Optimal Median Floor)
   - **Config:** `models/autots/best_autots_model.json`

3. **Anomaly Detector: Isolation Forest [Tree Ensemble]**
   - **Type:** Unsupervised Latency & Telemetry Anomaly Detection
   - **Configuration:** 200 Isolation Trees, contamination = 0.08, StandardScaler
   - **Features:** 24 multi-scale causal features & coupled stress interactions
   - **Spike Precision:** **90.51%** (Full Dataset) | **89.88% – 91.51%** (Out-of-Sample Test)
   - **Spike Accuracy:** **86.49% – 89.50%**
   - **Artifacts:** `models/isolation_forest/isolation_forest.pkl`, `scaler.pkl`, `feature_config.json`

---

## Directory Structure

```
models/
├── README.md                                  # Directory documentation & model catalog
├── arima/                                     # Baseline: Log-ARIMA(1,0,0)
│   ├── best_mae_arima_config.json             # ARIMA configuration & validation metadata
│   ├── arima_model.pkl                        # Serialized fitted ARIMA model
│   └── scaler.pkl                             # Transformer artifact
├── autots/                                    # Advanced Champion: AutoTS BasicLinearModel
│   └── best_autots_model.json                 # Selected model parameters & transformation pipeline
└── isolation_forest/                          # Anomaly Detector: Isolation Forest
    ├── isolation_forest.pkl                   # Trained Isolation Forest model
    ├── scaler.pkl                             # RobustStandardScaler artifact
    └── feature_config.json                    # Feature definitions & hyperparameters
```
