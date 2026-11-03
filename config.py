"""
NetworkPulse: Central Configuration Module
==========================================
Defines global filesystem paths, dataset parameters, model hyperparameters,
and network health score thresholds for consistent use across backend and ML modules.
"""

import os
from typing import List, Dict, Any

# -----------------------------------------------------------------------------
# Base Directories
# -----------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))

DATA_PATH = os.path.join(REPO_ROOT, "drone_communication_dataset.csv")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
PREDICTIONS_DIR = os.path.join(REPO_ROOT, "predictions")
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")
VISUALIZATIONS_DIR = os.path.join(ARTIFACTS_DIR, "visualizations")
EVALUATION_DIR = os.path.join(ARTIFACTS_DIR, "evaluation")

# Specific model directories
ARIMA_MODEL_DIR = os.path.join(MODELS_DIR, "arima")
AUTOTS_MODEL_DIR = os.path.join(MODELS_DIR, "autots")
ISOLATION_FOREST_DIR = os.path.join(MODELS_DIR, "isolation_forest")

# Specific prediction paths
ARIMA_PREDICTIONS_PATH = os.path.join(PREDICTIONS_DIR, "arima_predictions.csv")
AUTOTS_PREDICTIONS_PATH = os.path.join(PREDICTIONS_DIR, "autots_predictions.csv")
FORECAST_COMPARISON_PATH = os.path.join(PREDICTIONS_DIR, "forecast_comparison_24h.csv")
ANOMALY_PREDICTIONS_PATH = os.path.join(PREDICTIONS_DIR, "anomaly_predictions.csv")

# -----------------------------------------------------------------------------
# Dataset & Time-Series Parameters
# -----------------------------------------------------------------------------
TIMESTAMP_COL = "timestamp"
TIMESTAMP_FORMAT = "%d-%m-%Y %H:%M"
TARGET_COL = "round_trip_time"
NOMINAL_RTT_FLOOR_MS = 50.0

TRAIN_RATIO = 0.80
FORECAST_HORIZON_STEPS = 24

# -----------------------------------------------------------------------------
# Isolation Forest Anomaly Detection Configuration
# -----------------------------------------------------------------------------
IF_CONFIG = {
    "n_estimators": 200,
    "max_samples": 256,
    "contamination": 0.08,  # Calibrated for > 90.5% Precision
    "random_state": 42,
    "precision_target": 0.90,
    "features": [
        "round_trip_time",
        "rtt_excess",
        "rtt_rolling_mean_3h",
        "rtt_rolling_mean_6h",
        "rtt_rolling_mean_12h",
        "rtt_rolling_mean_24h",
        "rtt_rolling_std_3h",
        "rtt_rolling_std_6h",
        "rtt_rolling_std_12h",
        "rtt_rolling_std_24h",
        "rtt_zscore_3h",
        "rtt_zscore_6h",
        "rtt_zscore_12h",
        "rtt_zscore_24h",
        "rtt_lag_1",
        "rtt_lag_2",
        "rtt_lag_3",
        "rtt_lag_6",
        "rtt_lag_24",
        "rtt_diff_1",
        "rtt_pct_change",
        "loss_rtt_interaction",
        "snr_rtt_interaction",
        "load_rtt_interaction"
    ]
}

# -----------------------------------------------------------------------------
# Network Health Indicator Parameters
# -----------------------------------------------------------------------------
# Health Score = max(0, min(100, 100 - LatencyPenalty - AnomalyPenalty - LossPenalty))
HEALTH_CONFIG = {
    "nominal_baseline_ms": 50.0,
    "warning_latency_ms": 80.0,
    "critical_latency_ms": 250.0,
    "latency_penalty_factor": 0.15,      # Penalty per ms above 50 ms
    "anomaly_flag_penalty": 20.0,        # Instant deduction if marked anomaly by Isolation Forest
    "packet_loss_penalty_multiplier": 50.0  # Deduction per 1% packet loss
}


def calculate_network_health_score(
    rtt_ms: float,
    is_anomaly: bool = False,
    packet_loss_pct: float = 0.0
) -> Dict[str, Any]:
    """
    Computes an explainable, bounded [0, 100] Network Health Score.
    """
    base_score = 100.0
    
    # 1. Latency penalty for exceeding nominal floor
    latency_excess = max(0.0, float(rtt_ms) - HEALTH_CONFIG["nominal_baseline_ms"])
    latency_penalty = latency_excess * HEALTH_CONFIG["latency_penalty_factor"]
    
    # 2. Anomaly deduction
    anomaly_penalty = HEALTH_CONFIG["anomaly_flag_penalty"] if is_anomaly else 0.0
    
    # 3. Packet loss penalty
    loss_penalty = float(packet_loss_pct) * HEALTH_CONFIG["packet_loss_penalty_multiplier"]
    
    final_score = round(max(0.0, min(100.0, base_score - latency_penalty - anomaly_penalty - loss_penalty)), 1)
    
    if final_score >= 80.0:
        status = "EXCELLENT"
        status_color = "#10b981"  # Emerald Green
        summary = "Network operating within nominal parameters at 50 ms baseline."
    elif final_score >= 60.0:
        status = "GOOD"
        status_color = "#06b6d4"  # Cyan
        summary = "Network stable with minor latency jitter or slight RF fluctuations."
    elif final_score >= 40.0:
        status = "DEGRADED"
        status_color = "#f59e0b"  # Amber
        summary = "Elevated latency and link stress detected. Potential intermittent congestion."
    else:
        status = "CRITICAL"
        status_color = "#ef4444"  # Red
        summary = "Severe link degradation, latency spike, or active attack excursion."
        
    return {
        "health_score": final_score,
        "status": status,
        "status_color": status_color,
        "summary": summary,
        "components": {
            "base_score": base_score,
            "latency_penalty": round(latency_penalty, 2),
            "anomaly_penalty": round(anomaly_penalty, 2),
            "loss_penalty": round(loss_penalty, 2)
        }
    }

