# NetworkPulse: Network Performance Forecasting, Anomaly Detection & Observability Dashboard

[![Python Version](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/Tests-37%20Passed-10B981?logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-06B6D4)](#)

**NetworkPulse** is a full-stack time-series observability, forecasting, and unsupervised anomaly detection platform engineered for mission-critical unmanned aerial vehicle (UAV / drone) communication networks. It monitors high-frequency telemetry, forecasts round-trip latency (`round_trip_time`) across a 24-hour lead horizon, isolates cyber-physical anomalies with **Isolation Forest**, computes an explainable **Network Health Score**, and serves interactive analytics via a dark-mode SaaS dashboard.

---

## Architecture Overview

```
                          ┌──────────────────────────────────────────────┐
                          │    drone_communication_dataset.csv           │
                          │   (45,289 Hourly Drone Telemetry Readings)   │
                          └──────────────────────┬───────────────────────┘
                                                 │
                                80/20 Chronological Split (No Leakage)
                                                 │
                     ┌───────────────────────────┴───────────────────────────┐
                     ▼                                                       ▼
        Train Partition (36,231 hrs)                            Test Partition (9,058 hrs)
     [2019-11-01 ➔ 2023-12-19]                               [2023-12-19 ➔ 2024-12-31]
                     │                                                       │
         Multi-Model Optimization                                28 Rolling 24h Test Horizons
                     │                                                (672 Test Hours)
     ┌───────────────┴───────────────┐                                       │
     ▼                               ▼                                       ▼
┌──────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
│    Forecasting Models    │   │  Isolation Forest (IF)   │◄──┤  Evaluation & Artifacts  │
│                          │   │                          │   │                          │
│ 1. AutoTS (47.11 ms MAE) │   │ • 24 Causal Features     │   │ • 10 Visualizations      │
│ 2. ARIMA  (59.95 ms MAE) │   │ • 200 Isolation Trees    │   │ • Forecast Metrics JSON  │
│                          │   │ • Contam 0.08 (>90% Prec)│   │ • Anomaly Evaluation Rep │
└────────────┬─────────────┘   └─────────────┬────────────┘   └──────────────────────────┘
             │                               │
             └───────────────┬───────────────┘
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 Flask REST API & Web Dashboard              │
  │        (http://127.0.0.1:5000 • Dark-Mode SaaS UI)          │
  │                                                             │
  │  • Real-Time Telemetry & Rolling ±1σ Bounds                 │
  │  • 24-Hour Horizon Multi-Model Interactive Comparison       │
  │  • Unsupervised Isolation Forest Anomaly Detection (0..1)   │
  │  • Explainable Network Health Score (0..100)                │
  │  • Historical Telemetry & Protocol Distribution Analytics   │
  │  • Raw Dataset Explorer & RFC-4180 CSV Data Export          │
  └─────────────────────────────────────────────────────────────┘
```

---

## Machine Learning Models & Leaderboard

All forecasting models are evaluated on **672 contiguous held-out test hours across 28 rolling 24-hour evaluation horizons** with zero data leakage:

| Model Architecture | Role | Optimization Objective | Test MAE (672h) | Test RMSE | Test MAPE | Win Rate | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **AutoTS BasicLinearModel** | **Advanced Champion** | Wavelet Daubechies/Morlet + QuantileTransformer | **47.11 ms** | **119.90 ms** | **14.95%** | **28/28 (100%)** | Completed, subject to validation |
| **Log-ARIMA(1,0,0)** | **Fixed Baseline** | Log Gaussian autoregression with naive $\exp(\hat{z})$ | **59.95 ms** | **113.44 ms** | **46.75%** | 0/28 (0%) | Completed |
| **Isolation Forest Detector** | **Anomaly Engine** | Multi-Scale Ensemble (200 trees, $\nu = 0.08$) | — | — | — | — | Completed (>90.5% Precision) |

### Key Theoretical & Empirical Insights:
1. **The Bayes Error Floor of 47.11 ms:** 80.0% of all network observations sit at exactly 50.0 ms (nominal hardware link floor). Under $L_1$ loss (MAE), the theoretical Bayes optimal point predictor is the conditional median. AutoTS collapsed predictions to the 50.0 ms nominal floor via quantile inversion, achieving the lowest test MAE (47.11 ms) and lowest MAPE (14.95%).
2. **Memoryless Attack Spikes:** Rigorous Markov transition testing proved that cyber-attack excursions are memoryless Bernoulli events ($P(\text{Spike}_t \mid \text{Spike}_{t-1}) = 0.1990 \approx P(\text{Spike}) = 0.2000$), occurring without deterministic precursor signals.
3. **Log-ARIMA Reversion Penalty:** Log-ARIMA(1,0,0) reverts to the unconditional geometric mean (~70.5 ms), causing a ~20.5 ms over-prediction penalty during nominal calm periods.

---

## Isolation Forest Anomaly Detection

Trained via `train_isolation_forest.py` on 80% chronological train partition:
- **Architecture:** `sklearn.ensemble.IsolationForest(n_estimators=200, max_samples=256, contamination=0.08, random_state=42)`
- **24 Causal Features:** RTT, physical excess above 50 ms baseline, multi-scale rolling means & standard deviations (3h, 6h, 12h, 24h), rolling Z-scores, multi-lag signals (1, 2, 3, 6, 24), rate of change, and coupled latency-exogenous stress interaction features.
- **Continuous Scoring:** Normalizes decision function into continuous $[0, 1]$ severity score.
- **High-Precision Performance on Latency Spikes ($RTT > 50\text{ ms}$):**
  * **Precision:** **90.51%** (Full Dataset) | **89.88% – 91.51%** (Out-of-Sample Test)
  * **Classification Accuracy:** **86.49% – 89.50%**
  * **Specificity:** **98.35%** (Calm false alarm rate below $1.65\%$)
  * **Mean Anomaly RTT:** **331.5 ms** (vs Normal Mean RTT: **81.4 ms** — over **4.0×** latency contrast)
- **Contamination & Operating Threshold Grid:**
  * $\text{Contamination} = 0.05$ *(Ultra High Precision)*: Precision = **91.51%**, Accuracy = 84.07%, Recall = 23.50%, Anomaly Mean RTT = **350.8 ms**
  * $\text{Contamination} = 0.08$ *(Production Default)*: Precision = **90.51%**, Accuracy = 86.49%, Recall = 36.31%, Anomaly Mean RTT = **331.5 ms**
  * $\text{Contamination} = 0.12$ *(High Accuracy)*: Precision = **88.11%**, Accuracy = **89.03%**, Recall = 52.94%, Anomaly Mean RTT = **309.2 ms**
  * $\text{Contamination} = 0.20$ *(Balanced Maximum F1)*: Precision = **81.64%**, Accuracy = **92.69%**, Recall = **82.44%**, F1 = **82.04%**

---

## Explainable Network Health Score

Calculates bounded $[0, 100]$ operational score:

$$\text{Health Score} = \max\Big(0, \min\big(100, 100 - \text{LatencyPenalty} - \text{AnomalyPenalty} - \text{LossPenalty}\big)\Big)$$

- **Latency Penalty:** $\max(0, \text{RTT} - 50.0) \times 0.15$ (deduction per ms above 50 ms nominal floor)
- **Anomaly Penalty:** $20.0$ point instant deduction when flagged by Isolation Forest
- **Loss Penalty:** $\text{PacketLossPct} \times 50.0$ (deducts 50 points per 1.0% packet drop)
- **Status Classification:**
  * $\ge 80.0$: **EXCELLENT** (nominal 50 ms baseline, stable RF links)
  * $\ge 60.0$: **GOOD** (minor jitter or slight RF fluctuations)
  * $\ge 40.0$: **DEGRADED** (elevated latency, congestion, or intermittent packet loss)
  * $< 40.0$: **CRITICAL** (severe link breakdown or active cyber attack)

---

## REST API Endpoints

The backend provides certified REST API endpoints:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/overview` | `GET` | Summary statistics (avg, min, max, current RTT, detected anomalies, network health score) |
| `/api/network-performance` | `GET` | Telemetry time-series, rolling 6h mean/std bands, and Isolation Forest anomaly markers |
| `/api/forecast` | `GET` | 24h actual vs AutoTS vs ARIMA predictions for selected `?window=N` (1..28) |
| `/api/anomalies` | `GET` | Detected anomalies with continuous scores, severity tiers, and pagination |
| `/api/model-performance` | `GET` | Full multi-model metrics (MAE, RMSE, MAPE, sMAPE) and Bayes floor analysis |
| `/api/historical-analysis` | `GET` | Empirical distributions (p50..p99), calm vs burst breakdown, and protocol stats |
| `/api/health` | `GET` | System health check and dataset records count |
| `/api/data/explorer` | `GET` | Searchable, paginated raw dataset rows with `?anomalies_only=true` |
| `/api/export/csv` | `GET` | RFC-4180 CSV download streaming for anomalies, forecasts, and telemetry |

---

## Publication-Quality Visualizations

All 10 required visualization artifacts are rendered and stored in `artifacts/visualizations/`:

1. `actual_vs_forecast_rtt.png` — 24-hour horizon actual RTT vs AutoTS vs ARIMA
2. `arima_vs_actual.png` — Baseline Log-ARIMA(1,0,0) forecast against actual ground-truth
3. `autots_vs_actual.png` — AutoTS BasicLinearModel 50 ms floor forecast against actual ground-truth
4. `isolation_forest_anomaly_timeline.png` — Multi-day latency telemetry stream with anomaly overlay
5. `rtt_distribution_normal_vs_anomaly.png` — KDE comparison of normal vs anomalous observations
6. `anomaly_score_distribution.png` — Frequency distribution of continuous anomaly scores
7. `rolling_mean_and_std.png` — Telemetry with 6-hour rolling mean and $\pm 1\sigma$ volatility bands
8. `stationarity_analysis.png` — Differencing and ADF stationarity diagnostic ($p < 10^{-15}$)
9. `acf_plot.png` — Autocorrelation function ($k=1 \dots 48$) confirming non-seasonal structure
10. `pacf_plot.png` — Partial autocorrelation function isolating $p=1$ lag significance

---

## Repository Structure

```text
NetworkPulse/
├── app.py                             # Production Flask REST API and dashboard server
├── config.py                          # Central configuration (paths, features, health formula)
├── drone_communication_dataset.csv    # Benchmark telemetry dataset (45,289 hourly rows × 35 cols)
├── train_isolation_forest.py          # Isolation Forest training and sensitivity pipeline
├── generate_visualizations.py         # Generates all 10 visual artifacts
├── requirements.txt                   # Environment package dependencies
├── pytest.ini                         # Pytest configuration
├── README.md                          # Full system documentation
│
├── .agents/                           # Stitch MCP integration config
│   └── mcp_config.json                # Stitch endpoint & API authentication header
│
├── stitch_screens/                    # Stitch-generated UI screens & high-res designs
│   ├── networkpulse_overview.html     # Overview dashboard screen HTML
│   ├── networkpulse_overview.png      # Overview screen 2560x3564 screenshot
│   ├── networkpulse_rtt_forecasting.html # RTT forecasting screen HTML
│   ├── networkpulse_rtt_forecasting.png  # RTT forecasting screen 2560x4338 screenshot
│   ├── networkpulse_anomaly_detection.html # Anomaly detection screen HTML
│   └── networkpulse_anomaly_detection.png  # Anomaly detection screen 2560x4094 screenshot
│
├── src/                               # Modular application package
│   ├── anomaly_detection/             # Anomaly detection sub-package
│   │   ├── feature_engineering.py     # 14 causal features extraction
│   │   └── isolation_forest_detector.py # Isolation Forest detector with scoring [0, 1]
│   ├── data_loader.py                 # Telemetry ingestion, validation & timestamp parsing
│   ├── split.py                       # Strictly chronological 80/20 train/test splitter
│   ├── evaluation.py                  # Standardized metrics calculation (MAE, RMSE, MAPE, sMAPE)
│   └── forecasting_engine.py          # Unified inference engine: Log-ARIMA & AutoTS
│
├── models/                            # Serialized model artifacts
│   ├── arima/                         # Best-MAE Log-ARIMA(1,0,0) artifacts
│   ├── autots/                        # AutoTS champion model configuration
│   └── isolation_forest/              # Isolation Forest model (.pkl, scaler.pkl, config.json)
│
├── predictions/                       # Unified prediction files
│   ├── anomaly_predictions.csv        # 45,289 rows with continuous anomaly scores
│   ├── arima_predictions.csv          # 672h Log-ARIMA predictions
│   ├── autots_predictions.csv         # 672h AutoTS predictions
│   └── forecast_comparison_24h.csv    # 672h comparative multi-model evaluation file
│
├── artifacts/                         # Evaluation artifacts & reports
│   ├── visualizations/                # 10 publication-quality charts (.png)
│   ├── evaluation/                    # JSON metrics & comprehensive markdown reports
│   ├── autots_advanced/               # AutoTS advanced model metrics
│   └── final_prediction_audit/        # Prediction audit & L1 Bayes optimality proof
│
├── templates/
│   └── index.html                     # Responsive single-page application dashboard
│
├── static/
│   ├── css/dashboard.css              # Custom SaaS design system
│   └── js/dashboard.js                # Chart.js instances, tab routing & API client
│
└── tests/                             # Automated test suite (37 passing tests)
    ├── test_isolation_forest.py       # Anomaly detector & feature tests
    ├── test_dashboard_api.py          # REST API & route integration tests
    ├── test_arima_forecaster.py       # ARIMA stationarity & log-inversion tests
    ├── test_autots_advanced.py        # AutoTS artifact integrity & schema tests
    ├── test_final_prediction_audit.py # Constant-median & horizon integrity tests
    └── test_data_and_eval.py          # Chronological split & metric computation tests
```

---

## Quickstart Guide

### 1. Environment Setup

Clone the repository and install dependencies in a Python 3.12 virtual environment:

```powershell
# Clone repository
git clone https://github.com/ShauriyaDeveloper1/NetworkPulse.git
cd NetworkPulse

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard

Start the local Flask observability server:

```powershell
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 3. Run Anomaly Detection & Visualization Pipelines

```powershell
# Train Isolation Forest & Generate Predictions
python train_isolation_forest.py

# Generate All 10 Publication Visualizations
python generate_visualizations.py
```

### 4. Run Automated Test Suite

Verify all 37 unit and integration tests across data loading, models, anomaly detection, and API routes:

```powershell
.\venv\Scripts\python.exe -m pytest -v tests
```
*Expected result:* `37 passed in ~14s`

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
