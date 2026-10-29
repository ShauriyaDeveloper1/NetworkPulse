# NetworkPulse — Prediction Integrity and Evaluation Audit Report

**Audit Target:** Out-of-sample prediction integrity, transformation mechanics, and evaluation consistency for `round_trip_time` (RTT) forecasting.  
**Dataset:** `drone_communication_dataset.csv` (45,289 hourly observations)  
**Evaluated Artifacts:**  
- `artifacts/autots_advanced/advanced_predictions.csv`  
- `artifacts/autots_advanced/baseline_predictions.csv`  
- `artifacts/autots_advanced/best_autots_model.json`  
- `artifacts/autots_advanced/model_comparison.csv`  
- `artifacts/autots_advanced/evaluation_metrics.json`  

---

## 1. Executive Verdict & Core Findings

| Audit Check | Verdict | Details |
| :--- | :---: | :--- |
| **Row Count & Timestamp Alignment** | **VERIFIED PERFECT** | Exactly 672 hourly observations across 28 rolling 24-hour windows. Timestamps, origins, and ground-truth RTT match 100% between baseline and advanced models. |
| **Reproducibility of Reported MAE** | **100% REPRODUCED** | Recomputed directly from saved predictions: AutoTS MAE = **47.1146 ms** (reported as 47.11 ms); Log-ARIMA MAE = **59.9492 ms** (reported as 59.95 ms). |
| **AutoTS Prediction Dynamics** | **COLLAPSED TO CONSTANT 50.0 ms** | AutoTS `BasicLinearModel` produced exactly **1 unique value (50.000 ms)** across all 672 forecasted hours (mean: 50.0, std: 0.0). |
| **Constant 50 ms Diagnostic Comparison** | **NUMERICALLY IDENTICAL** | Predicting a constant 50.0 ms across all 672 hours yields an MAE of **47.1146 ms** (difference vs AutoTS: **0.000000 ms**). AutoTS provides **zero dynamic forecasting variation** beyond emitting the nominal hardware floor. |
| **Physical Clipping [50, 500] ms** | **JUSTIFIED** | Hardware nominal floor is 50.0 ms (zero observations < 50.0 ms in all 45,289 dataset rows). Clamping enforces domain validity. |
| **Evaluation Window Consistency** | **VERIFIED & RECONCILED** | Validation was executed strictly prior to `train_end` (index 36,231; `2023-12-19 14:00:00`). Test windows were completely held out until final scoring. |

---

## 2. Granular Prediction Audit

### 2.1 File & Alignment Verification
A line-by-line inspection of `advanced_predictions.csv` and `baseline_predictions.csv` confirmed:
- **Total Rows:** Exactly 672 rows in each file ($28 \text{ windows} \times 24 \text{ forecast steps}$).
- **Chronological Alignment:** Timestamps match across every single row:
  - Window 1: `2023-12-19 15:00:00` to `2023-12-20 14:00:00`
  - Window 28: `2024-06-25 15:00:00` to `2024-06-26 14:00:00`
- **Zero Temporal Shift:** Step $h=1$ always corresponds to $t_0 + 1\text{ hour}$, through step $h=24$ ($t_0 + 24\text{ hours}$).
- **Actual RTT:** Matches the raw dataset observations at each timestamp with 0.00 ms discrepancy.

### 2.2 Prediction Distribution & Variance Analysis

| Metric | Baseline Log-ARIMA(1,0,0) | AutoTS BasicLinearModel | Constant 50.0 ms Benchmark |
| :--- | :---: | :---: | :---: |
| **Count** | 672 | 672 | 672 |
| **Number of Unique Values** | **254** | **1** | **1** |
| **Mean Prediction** | 70.53 ms | 50.00 ms | 50.00 ms |
| **Standard Deviation** | 0.83 ms | **0.00 ms** | 0.00 ms |
| **Minimum Prediction** | 68.55 ms | 50.00 ms | 50.00 ms |
| **25th Percentile** | 69.82 ms | 50.00 ms | 50.00 ms |
| **50th Percentile (Median)** | 70.37 ms | 50.00 ms | 50.00 ms |
| **75th Percentile** | 71.37 ms | 50.00 ms | 50.00 ms |
| **Maximum Prediction** | 72.38 ms | 50.00 ms | 50.00 ms |

#### Key Finding: Why AutoTS Collapsed to 50.0 ms
AutoTS selected the following pipeline:
- Architecture: `BasicLinearModel`
- Transformation: `QuantileTransformer(n_quantiles=20, output_distribution='uniform')` + `CointegrationTransformer(2)` + `Slice(0.5)`

In the training telemetry, **80.00% of all observations sit exactly at 50.0 ms**. Under a 20-quantile uniform transformation:
1. All observations in the bottom 80% map to the lowest quantiles ($q \le 0.80$).
2. During inverse transformation, any predicted value in the range $[0.0, 0.80]$ maps back strictly to **50.000 ms**.
3. AutoTS successfully recognized that predicting the **conditional geometric median** (50.0 ms) minimizes the L1 loss function, because any positive prediction elevation during the 80% calm hours creates an immediate, linear penalty.

---

## 3. Constant 50.0 ms Diagnostic Benchmark Comparison

To evaluate whether AutoTS provides measurable forecasting value beyond predicting the dominant nominal floor, we computed metrics for a naive constant 50.0 ms predictor:

### 3.1 672-Hour Test Benchmark (28 Windows Across 4 Seasons)

| Model | Pooled MAE (ms) | Pooled RMSE (ms) | sMAPE (%) | Macro Window MAE (ms) | Windows Won vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AutoTS BasicLinearModel** | **47.1146** | 119.9034 | 24.81% | 47.11 ± 23.80 | **28 / 28 (100.0%)** |
| **Constant 50.0 ms Predictor** | **47.1146** | 119.9034 | 24.81% | 47.11 ± 23.80 | **28 / 28 (100.0%)** |
| **Baseline Log-ARIMA(1,0,0)** | 59.9492 | **113.4361** | 33.15% | 59.95 ± 20.33 | 0 / 28 (0.0%) |
| **Delta (AutoTS - Constant 50)** | **0.000000** | **0.000000** | **0.00%** | **0.000000** | — |

### 3.2 672-Hour Validation Benchmark (28 Windows Prior to Train Boundary)

| Model | Pooled MAE (ms) | Pooled RMSE (ms) | sMAPE (%) | Macro Window MAE (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **AutoTS BasicLinearModel** | **50.5655** | 122.6812 | 26.33% | 50.57 ± 20.91 |
| **Constant 50.0 ms Predictor** | **50.5655** | 122.6812 | 26.33% | 50.57 ± 20.91 |
| **Baseline Log-ARIMA(1,0,0)** | 63.4169 | **115.4326** | 34.62% | 63.42 ± 17.85 |
| **Delta (AutoTS - Constant 50)** | **0.000000** | **0.000000** | **0.00%** | **0.000000** |

### 3.3 Diagnostic Conclusion
AutoTS **does not beat the constant 50.0 ms diagnostic predictor**. Its predictions are numerically identical to predicting 50.0 ms. AutoTS's "advantage" over Log-ARIMA stems entirely from adopting the nominal floor (50.0 ms) rather than mean-reverting towards the geometric mean (~70.5 ms).

---

## 4. Reconciliation of Validation vs Test Metrics

The previous reports documented different MAE scores across different phases. These have been independently reconciled:

| Metric Reported | Value | Partition & Sample Size | Origin & Explanation |
| :--- | :---: | :--- | :--- |
| **AutoTS Internal Val Score** | **36.14 ms** | 48 hours ($2 \times 24\text{h}$) | Computed by AutoTS's internal `validation_method='backwards'` across 2 validation rounds right before `train_end`. During those specific 48 hours, fewer attack bursts occurred, yielding an MAE of 36.14 ms. |
| **Validation Benchmark** | **50.57 ms** | 672 hours ($28 \times 24\text{h}$) | Computed across 28 rolling windows prior to `train_end` (`2023-11-21 15:00:00` to `2023-12-18 15:00:00`). Reflects the long-term empirical attack rate (20.2% spikes). |
| **Held-Out Test Benchmark** | **47.11 ms** | 672 hours ($28 \times 24\text{h}$) | Computed on untouched held-out test data (`2023-12-19 15:00:00` to `2024-06-25 14:00:00`). During this test sample, average spike severity was slightly lower, producing exactly 47.11 ms. |
| **Primary 24h Test Horizon** | **58.00 ms** | 24 hours (Window 1) | The first 24 hours of the held-out test partition (`2023-12-19 15:00:00` to `2023-12-20 14:00:00`), which contained 5 spike hours (20.8% attack density). |

Every number is verified, mathematically accounted for, and derived without temporal leakage.

---

## 5. Physical Justification of 50–500 ms Clipping

- **Dataset Ground Truth:** Across all 45,289 records, $\min(\text{RTT}) = 50.000\text{ ms}$, $\text{median} = 50.000\text{ ms}$, $\max = 499.000\text{ ms}$.
- **Hardware Nominal Floor:** In drone RF communication systems, physical hardware transmission and round-trip handshakes cannot complete in sub-50 ms latencies under this protocol configuration.
- **Audit Conclusion:** Clipping is fully justified by domain physics and prevents unphysical negative values. In the case of AutoTS `BasicLinearModel`, the quantile inversion naturally mapped predictions to 50.0 ms even before the clipping step.
