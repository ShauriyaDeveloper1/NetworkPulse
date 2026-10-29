# NetworkPulse: Network Performance Forecasting, Anomaly Detection & Observability Dashboard

[![Python Version](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/Tests-36%20Passed-10B981?logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-06B6D4)](#)

**NetworkPulse** is a full-stack time-series observability, forecasting, and anomaly detection platform engineered for mission-critical unmanned aerial vehicle (UAV / drone) communication networks. It monitors high-frequency telemetry, forecasts round-trip latency (`round_trip_time`) across a 24-hour lead horizon, classifies cyber-physical anomalies, and serves interactive analytics via a dark-mode SaaS dashboard.

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
                     ▼                                                       │
  ┌─────────────────────────────────────────────────────────────┐            │
  │                     Forecasting Models                      │◄───────────┘
  │                                                             │
  │  1. AutoTS BasicLinearModel    (Champion:  47.11 ms MAE)    │
  │  2. XGBoost Quantile (q=0.46)  (Benchmark: 47.16 ms MAE)    │
  │  3. Log-ARIMA(1,0,0) Baseline  (Baseline:  59.95 ms MAE)    │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 Flask REST API & Web Dashboard              │
  │        (http://127.0.0.1:5000 • Dark-Mode SaaS UI)          │
  │                                                             │
  │  • Real-Time Telemetry & Rolling ±1σ Bounds                 │
  │  • 24-Hour Horizon Multi-Model Interactive Comparison       │
  │  • Anomaly Detection & Cyber-Attack Attribution (Jamming...)│
  │  • Step-by-Step Horizon Error Profile (h=1..24)             │
  │  • Raw Dataset Explorer & RFC-4180 CSV Data Export          │
  └─────────────────────────────────────────────────────────────┘
```

---

## Forecasting Benchmark Leaderboard

All models are evaluated on **672 contiguous held-out test hours across 28 rolling 24-hour evaluation horizons** (Winter, Spring, and Summer test seasons) with zero data leakage:

| Model Architecture | Role | Optimization Objective | Test MAE (672h) | Test RMSE | Win Rate (vs Baseline) | Inference Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **AutoTS BasicLinearModel** | **Advanced Champion** | Wavelet Multiscale + QuantileTransformer ($n=20$) | **47.11 ms** | **119.90 ms** | **28 / 28 (100.0%)** | 755.6 ms |
| **XGBoost Quantile ($q=0.46$)** | **Feature Benchmark** | Pinball Loss targeting conditional median with 82 causal lags | **47.16 ms** | **119.87 ms** | **27 / 28 (96.4%)** | 82.5 ms |
| **Log-ARIMA(1,0,0)** | **Fixed Baseline** | Log-space Gaussian with naive exponential inversion $\exp(\hat{z})$ | **59.95 ms** | **113.44 ms** | 0 / 28 (0.0%) | 44.4 ms |

### Key Theoretical & Empirical Insights:
1. **The Bayes Error Floor of 47.11 ms:** 80.0% of all network observations sit at exactly 50.0 ms (nominal hardware link floor). Under $L_1$ loss (MAE), the optimal point predictor is the conditional median. AutoTS collapsed predictions to the 50.0 ms nominal floor, achieving the empirical Bayes error bound on the held-out test set.
2. **Memoryless Attack Spikes:** Rigorous Markov transition testing proved that cyber-attack excursions are memoryless Bernoulli events ($P(\text{Spike}_t \mid \text{Spike}_{t-1}) = 0.1990 \approx P(\text{Spike}) = 0.2000$). Because attacks occur without precursor notice, sub-2.0 ms MAE cannot be achieved without future lookahead leakage.
3. **Log-ARIMA Reversion Penalty:** Log-ARIMA(1,0,0) reverts to the unconditional geometric mean (~70.5 ms), causing a ~20.5 ms over-prediction penalty during nominal calm periods.

---

## Features & Capabilities

### 1. Web Observability Dashboard (`http://127.0.0.1:5000`)
- **Theme:** High-contrast dark-mode SaaS UI inspired by Datadog and Grafana (deep navy `#080c14`, cyan `#06b6d4`, electric blue `#3b82f6`, violet `#8b5cf6`).
- **Sidebar Navigation:** Collapsible sidebar with 8 dedicated interactive views:
  1. **Overview:** System KPI cards, primary telemetry & rolling variance chart, recent anomaly stream, and model scorecard.
  2. **Live Monitoring:** High-frequency telemetry stream with rolling 6h mean, dynamic anomaly threshold, packet loss rate (%), and Signal-to-Noise Ratio (SNR dB).
  3. **Latency Forecast:** Window selector (Windows 1–28) comparing Actual RTT vs. AutoTS vs. Log-ARIMA vs. XGBoost across all 24 hours ($h=1 \dots 24$) with step-by-step error profiles.
  4. **Anomaly Detection:** Tabular anomaly log classified into 4 severity tiers (Critical, High, Medium, Low) and attributed to cyber-attack signatures.
  5. **Model Comparison:** Comparative leaderboard, parameter specs, and architectural analysis.
  6. **Historical Analytics:** RTT empirical percentile breakdown, cyber-attack frequency chart, transport protocol split (UDP vs TCP), and encryption overhead.
  7. **Data Explorer:** Searchable, filterable tabular dataset browser with server-side pagination.
  8. **Reports & Export:** Executive summary report with printable PDF styling and one-click RFC-4180 CSV downloads.

### 2. Anomaly Detection & Cyber-Attack Attribution
Dynamic threshold monitoring identifies latency excursions ($> 50.0\text{ ms}$) and maps them against network threat signatures:
- **Jamming:** RF noise flooding causing extreme RTT and SNR collapse.
- **Spoofing / GPS Spoofing:** Packet desynchronization excursions.
- **Man-in-the-Middle (MITM):** Routing delay insertions.
- **DDoS:** Base station buffer exhaustion.
- **Malware & Protocol Exploits:** Uncharacteristic latency bursts.

### 3. REST API Endpoints
The backend provides a comprehensive, documented REST API:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/health` | `GET` | Service status, dataset record count, timestamps, and model availability |
| `/api/metrics/summary` | `GET` | Current RTT, average RTT, anomaly count/rate, forecasted max RTT, and health status |
| `/api/telemetry/history` | `GET` | Sliced time-series with 6h rolling mean, rolling standard deviation bands, and anomalies |
| `/api/forecast/windows` | `GET` | Index of all 28 rolling 24-hour test horizons with dates and summary MAEs |
| `/api/forecast/comparison` | `GET` | 24-hour actual vs AutoTS vs ARIMA vs XGBoost predictions for selected `?window=N` |
| `/api/forecast/horizon_profile` | `GET` | Step-by-step ($h=1 \dots 24$) average MAE error profile |
| `/api/models/comparison` | `GET` | Structured metrics, parameters, and theoretical Bayes floor analysis |
| `/api/anomalies` | `GET` | Paginated anomaly events with severity filters (`?severity=critical`) and attack labels |
| `/api/data/explorer` | `GET` | Searchable, paginated raw dataset rows with `?anomalies_only=true` |
| `/api/export/csv` | `GET` | RFC-4180 CSV download streaming for anomalies, forecasts, and telemetry |
| `/api/reports/summary` | `GET` | Executive report metadata and certified evaluation findings |

---

## Repository Structure

```text
NetworkPulse/
├── app.py                             # Production Flask REST API and dashboard server
├── drone_communication_dataset.csv    # Benchmark telemetry dataset (45,289 hourly rows × 35 cols)
├── run_final_two_models.py            # Master evaluation script for active two-model architecture
├── requirements.txt                   # Environment package dependencies
├── pytest.ini                         # Pytest configuration
├── .gitignore                         # Standard Python, virtualenv & cache exclusions
├── README.md                          # Full system documentation
│
├── src/                               # Core modular package
│   ├── __init__.py                    # Package exports
│   ├── data_loader.py                 # Telemetry ingestion, validation & timestamp parsing
│   ├── split.py                       # Strictly chronological 80/20 train/test splitter
│   ├── evaluation.py                  # Standardized metrics calculation (MAE, RMSE, sMAPE)
│   └── forecasting_engine.py          # Unified inference engine: Log-ARIMA & Quantile XGBoost
│
├── models/                            # Model-wise production artifacts
│   ├── README.md                      # Catalog of model files and execution instructions
│   ├── arima/                         # Model A: Best-MAE ARIMA
│   │   ├── best_mae_arima_config.json # Hyperparameters & validation metrics
│   │   ├── run_best_mae_arima.py      # Standalone ARIMA execution script
│   │   └── NetworkPulse_Best_MAE_ARIMA.ipynb
│   └── xgboost/                       # Model B: Best-MAE XGBoost
│       ├── best_mae_xgboost_config.json # Hyperparameters & feature configuration
│       ├── best_mae_xgboost_model.json  # Serialized XGBoost booster artifact
│       ├── run_best_mae_xgboost.py    # Standalone XGBoost execution script
│       └── NetworkPulse_Best_MAE_XGBoost.ipynb
│
├── notebooks/                         # Clean, ordered Jupyter Notebooks
│   ├── README.md                      # Notebook catalog & quickstart guide
│   ├── 01_NetworkPulse_Best_MAE_ARIMA.ipynb
│   ├── 02_NetworkPulse_Best_MAE_XGBoost.ipynb
│   └── 03_NetworkPulse_Final_Two_Models_Comparison.ipynb
│
├── artifacts/                         # Evaluation artifacts & reports
│   ├── autots_advanced/               # AutoTS champion model config, predictions & metrics
│   ├── final_two_model_architecture/  # Aligned 672h predictions, comparison CSVs & charts
│   ├── final_prediction_audit/        # Mathematical audit, Bayes proof & window metrics
│   └── research_experiments_archive/  # Safely archived intermediate experiments
│
├── templates/
│   └── index.html                     # Responsive single-page application dashboard
│
├── static/
│   ├── css/
│   │   └── dashboard.css              # Custom SaaS design system & print styles
│   └── js/
│       └── dashboard.js               # Chart.js instances, tab routing & API client
│
└── tests/                             # Automated test suite (36 passing tests)
    ├── conftest.py                    # Shared test fixtures
    ├── test_dashboard_api.py          # REST API & frontend route integration tests
    ├── test_arima_forecaster.py       # ARIMA stationarity & log-inversion unit tests
    ├── test_xgboost_forecaster.py     # XGBoost feature engineering & recursive forecast tests
    ├── test_final_two_models.py       # Two-model multi-window benchmark tests
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
git clone https://github.com/your-username/NetworkPulse.git
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

### 3. Run Forecasting Pipelines

Execute the master evaluation pipeline or individual models:

```powershell
# Run the Master Two-Model Evaluation Pipeline (Both Models)
python run_final_two_models.py

# Run Model A: Best-MAE Log-ARIMA(1,0,0)
python models/arima/run_best_mae_arima.py

# Run Model B: Best-MAE XGBoost Quantile (q=0.46)
python models/xgboost/run_best_mae_xgboost.py
```

### 4. Run Automated Test Suite

Verify all 36 unit and integration tests across data loading, models, and API routes:

```powershell
pytest -v tests
```
*Expected result:* `36 passed in ~24s`

---

## Citation & Dataset

The telemetry data originates from high-frequency drone communication logs tracking radio frequency parameters, communication protocols (UDP/TCP), encryption schemes (AES-128/256), and labeled cyber-attack vectors:
- **Sampling Interval:** Contiguous hourly intervals (1 hour)
- **Time Span:** November 1, 2019 to December 31, 2024 (45,289 records)
- **Target Variable:** `round_trip_time` (Nominal floor: 50.0 ms; Anomaly range: 51.0–500.0 ms)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
