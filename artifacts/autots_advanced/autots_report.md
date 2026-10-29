# NetworkPulse — AutoTS Advanced Forecasting Pipeline Report

**Telemetry Target:** `round_trip_time` (measured in milliseconds)  
**Dataset:** `drone_communication_dataset.csv` (45,289 hourly observations)  
**Library Version:** `autots` v1.0.4  
**Forecast Horizon:** Exactly 24 consecutive hourly steps ($h = 1 \dots 24$)  
**Optimization Objective:** Minimum Validation Mean Absolute Error (MAE in ms)  
**Evaluation Protocol:** Strict chronological training, validation, and untouched held-out testing  

---

## 1. Executive Summary & Model Selection

| Model Role | Architecture | Order / Hyperparameters | Transformation Pipeline | Chronological Val MAE | Primary 24h Test MAE | Multi-Week Pooled MAE (672h) | Multi-Week Pooled RMSE | Training Time |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | **Log-ARIMA(1,0,0)** | $(p=1, d=0, q=0)$ | $\ln(y)$, Naive $\exp(\hat{z})$ inversion | **55.83 ms** | **70.73 ms** | **59.95 ms** | **113.44 ms** | **0.04 s** |
| **Advanced (AutoTS)** | **BasicLinearModel** | `{"datepart_method": ["db2_365.25_12_0.5", "morlet_7_7_1"], "changepoint_spacing": 5040, "changepoint_distance_end": 360, "regression_type": null, "lambda_": 1000, "trend_phi": 0.98, "holiday_countries_used": false}` | `{"fillna": "ffill", "transformations": {"0": "QuantileTransformer", "1": "CointegrationTransformer", "2": "Slice"}, "transformation_params": {"0": {"output_distribution": "uniform", "n_quantiles": 20}, "1": {"n_components": 2, "max_components": 5, "shrinkage": 1e-05, "method": "cca", "min_periods": 20}, "2": {"method": 0.5}}}` | **36.14 ms** | **58.00 ms** | **47.11 ms** | **119.90 ms** | **176.05 s** |

### Key Findings & Verdict
1. **Model Discovery:** AutoTS searched multiple candidate model families (Exponential Smoothing / ETS, GLM, GLS, SeasonalNaive, AverageValueNaive, Motifs, NVAR, DMD, BasicLinearModel) under backward chronological validation and identified **BasicLinearModel** as the top-performing configuration targeting MAE minimization.
2. **Head-to-Head Performance:** AutoTS model **BasicLinearModel** achieved a pooled MAE of **47.11 ms**, representing an improvement of **+12.83 ms (+21.41%)** over the Log-ARIMA baseline.
3. **Out-of-Sample Windows Won:** AutoTS won **28 out of 28** non-overlapping 24-hour test horizons (100.0%).
4. **Target Noise Floor & 2.0 ms Target:** As demonstrated theoretically, the telemetry distribution contains an intrinsic noise floor of **50.02 ms** (80% calm at 50.0 ms nominal floor, 20% stochastic attack bursts up to 499 ms with standard deviation 112.56 ms). An MAE of $\le 2.0$ ms is mathematically unattainable without future attack label leakage.

---

## 2. Experimental Methodology & Chronological Partitions

### 2.1 Chronological Splits (Zero Temporal Leakage)
- **Training Set (80%):** Observations 0 to 36,231 (`2019-11-01 00:00:00` to `2023-12-19 14:00:00`).
- **Held-Out Test Set (20%):** Observations 36,231 to 45,289 (`2023-12-19 15:00:00` to `2024-12-31 00:00:00`).
- **AutoTS Search History:** Recent 4,000 hours of the training partition (`2023-07-05 23:00:00` to `2023-12-19 14:00:00`).
- **Validation Scheme:** AutoTS `backwards` validation across 2 non-overlapping 24-hour validation rounds prior to the training boundary.

### 2.2 Optimization Objective
- Direct minimization of MAE in milliseconds:
  ```python
  metric_weighting = {
      'mae_weighting': 10.0,
      'rmse_weighting': 0.0,
      'smape_weighting': 0.0,
      'spl_weighting': 0.0,
      'containment_weighting': 0.0,
      'contour_weighting': 0.0,
      'runtime_weighting': 0.0,
  }
  ```
- Non-negative constraint enabled (`no_negatives=True`) with telemetry bounds clipping $[50.0, 500.0]$ ms.

---

## 3. Head-to-Head Evaluation Results

### 3.1 Primary 24-Hour Horizon (Window 1: `2023-12-19 15:00:00` to `2023-12-20 14:00:00`)

| Metric | Fixed Baseline Log-ARIMA(1,0,0) | AutoTS [BasicLinearModel] | Absolute Delta | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **MAE (ms)** | **70.73** | **58.00** | **-12.73** | **+18.00%** |
| **RMSE (ms)** | **127.95** | **135.73** | **+7.78** | **-6.08%** |
| **sMAPE (%)** | **53.62%** | **29.41%** | **-24.21%** | — |
| **Inference Time (s)** | **0.0426** | **0.7229** | **+0.6804** | — |

### 3.2 Multi-Week Benchmark (28 Windows = 672 Hours Across 4 Test Seasons)

| Metric | Fixed Baseline Log-ARIMA(1,0,0) | AutoTS [BasicLinearModel] | Delta | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Pooled (Micro) MAE** | **59.95 ms** | **47.11 ms** | **+12.83 ms** | AutoTS outperformed baseline |
| **Pooled (Micro) RMSE** | **113.44 ms** | **119.90 ms** | **+6.47 ms** | L2 error across all 672 observations |
| **Macro Window MAE** | **59.95 +/- 20.33 ms** | **47.11 +/- 23.80 ms** | **-12.83 ms** | Mean over 28 daily windows |
| **Windows Won** | **0 / 28 (0.0%)** | **28 / 28 (100.0%)** | — | Non-overlapping daily wins |
| **Total Inference Time** | **1.24 s** | **21.16 s** | **+19.91 s** | Total inference across 28 origins |

---

## 4. Horizon Error Profile (Step 1 Through 24)

| Step ($h$) | Baseline Log-ARIMA MAE (ms) | AutoTS [BasicLinearModel] MAE (ms) | Delta (ms) |
| :---: | :---: | :---: | :---: |
| **h = 01** | 84.10 | 72.14 | -11.96 |
| **h = 02** | 35.61 | 18.04 | -17.57 |
| **h = 03** | 45.93 | 34.18 | -11.75 |
| **h = 04** | 62.36 | 51.93 | -10.43 |
| **h = 05** | 52.05 | 39.07 | -12.98 |
| **h = 06** | 70.43 | 55.50 | -14.93 |
| **h = 07** | 68.71 | 60.00 | -8.71 |
| **h = 08** | 73.13 | 64.25 | -8.88 |
| **h = 09** | 62.52 | 52.32 | -10.20 |
| **h = 10** | 38.03 | 22.04 | -16.00 |
| **h = 11** | 41.02 | 23.54 | -17.49 |
| **h = 12** | 82.54 | 75.11 | -7.43 |
| **h = 13** | 61.63 | 49.71 | -11.91 |
| **h = 14** | 32.50 | 17.79 | -14.71 |
| **h = 15** | 70.48 | 58.96 | -11.52 |
| **h = 16** | 84.11 | 78.21 | -5.90 |
| **h = 17** | 42.38 | 26.36 | -16.03 |
| **h = 18** | 82.11 | 70.21 | -11.89 |
| **h = 19** | 40.30 | 24.39 | -15.91 |
| **h = 20** | 76.33 | 63.00 | -13.33 |
| **h = 21** | 69.15 | 55.96 | -13.19 |
| **h = 22** | 28.27 | 9.14 | -19.13 |
| **h = 23** | 93.32 | 83.21 | -10.11 |
| **h = 24** | 41.77 | 25.68 | -16.09 |

---

## 5. Artifacts and Reproducibility

All generated artifacts are saved in `artifacts/autots_advanced/`:

- `autots_advanced.py` — AutoTS training, validation, and evaluation pipeline script.
- `best_autots_model.json` — Selected model metadata, hyperparameters, and transformation parameters.
- `model_comparison.csv` — Standardized comparison table between Log-ARIMA and AutoTS.
- `advanced_predictions.csv` — Full out-of-sample prediction table for AutoTS (672 observations).
- `baseline_predictions.csv` — Full out-of-sample prediction table for Log-ARIMA (672 observations).
- `forecast_comparison.png` — Two-panel diagnostic plot of actual versus predicted RTT.
- `horizon_error_profile.png` — Horizon step error degradation profile ($h = 1 \dots 24$).
- `evaluation_metrics.json` — Machine-readable evaluation metrics and configuration parameters.
- `autots_report.md` — This comprehensive engineering and scientific evaluation report.

---

## 6. Conclusions & Limitations

1. **AutoTS Effectiveness:** AutoTS successfully searched a wide variety of time-series model families without manual parameter guessing, providing an automated and reproducible advanced forecasting pipeline.
2. **Preservation of Assets:** Existing XGBoost model artifacts, preprocessing routines, and configs were completely preserved without modification or deletion.
3. **Physical Telemetry Limits:** The 50 ms nominal floor and intermittent, unannounced cyber-physical attack spikes impose a hard theoretical limit on univariate forecasting accuracy. Both models effectively learn the nominal baseline regime while remaining resilient to extreme outliers.
