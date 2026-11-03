# NetworkPulse: Isolation Forest Anomaly Detection Report

**Author:** Senior Machine Learning Engineer  
**Model:** `sklearn.ensemble.IsolationForest`  
**Dataset:** `drone_communication_dataset.csv` (45,289 hourly records)  
**Selected Contamination:** 0.08 (8.0%)  
**Features:** 14 Causal Time-Series & Exogenous Telemetry Features  

---

## 1. Executive Summary & Objective

In high-reliability drone communication networks, sudden packet round-trip time (RTT) degradation signals either physical RF channel fading or hostile cyber-physical attacks (jamming, spoofing, MITM).

An unsupervised **Isolation Forest** was trained on the chronological training partition (first 80% = 36,231 records, 2019 to 2023) and validated on the chronological holdout test partition (20% = 9,058 records, 2023 to 2024) to avoid data leakage.

### Key Performance Numbers (Chronological Test Partition):
- **Detected Anomaly Rate:** **8.18%** (741 anomalous hours out of 9,058)
- **Mean RTT of Anomalous Observations:** **331.5 ms**
- **Mean RTT of Normal Observations:** **80.1 ms**
- **Median RTT of Normal Observations:** **50.0 ms** (matches nominal 50.0 ms link floor)

---

## 2. Contamination Sensitivity Analysis

| Contamination Parameter | Precision (RTT Spikes) | Recall (RTT Spikes) | Accuracy | F1-Score | Anomaly Mean RTT | Normal Mean RTT |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.05** | **91.51%** | 23.50% | 84.07% | 37.40% | 350.8 ms | 86.9 ms |
| **0.08** | **89.88%** | 36.31% | 86.28% | 51.73% | 331.5 ms | 80.1 ms |
| **0.10** | **88.95%** | 45.20% | 87.77% | 59.94% | 319.0 ms | 75.6 ms |
| **0.12** | **88.11%** | 52.94% | 89.03% | 66.14% | 309.2 ms | 71.8 ms |
| **0.15** | **85.62%** | 65.27% | 90.75% | 74.07% | 292.1 ms | 65.7 ms |
| **0.20** | **81.64%** | 82.44% | 92.69% | 82.04% | 268.0 ms | 57.6 ms |

---

## 3. Important Methodological Disclaimers

> [!NOTE]
> **Unsupervised Nature of Isolation Forest:**  
> Isolation Forest isolates anomalies based solely on tree path length in the feature space without reference to ground-truth labels. The dataset's external cyber-attack flags (`label_jamming`, `label_spoofing`, etc.) are provided for contextual verification only. High-RTT excursions caused by natural RF degradation are genuine operational anomalies even if not caused by a malicious cyber attack. Therefore, standard supervised metrics (accuracy/F1) must be interpreted strictly in this context.

---

## 4. Deliverables Produced
- Trained Booster Model: `models/isolation_forest/isolation_forest.pkl`
- Feature Scaler: `models/isolation_forest/scaler.pkl`
- Feature Configuration: `models/isolation_forest/feature_config.json`
- Full Dataset Anomaly Predictions: `predictions/anomaly_predictions.csv`
- Metrics JSON: `artifacts/evaluation/isolation_forest_metrics.json`
