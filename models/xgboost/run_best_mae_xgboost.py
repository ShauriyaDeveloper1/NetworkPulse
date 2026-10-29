"""
NetworkPulse: Model B (Best-MAE XGBoost) Execution & Evaluation Script
Model: XGBoost Quantile Regressor (q=0.46, max_depth=3, lr=0.03) with recursive multi-step forecasting.
Executes 24-hour horizon forecasting, evaluates on the primary test window and multi-window benchmark,
and produces diagnostic reports and metrics.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is available
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data_loader import load_and_validate_dataset
from src.evaluation import calculate_metrics
from src.forecasting_engine import BestMaeXGBoostForecaster


def run_best_mae_xgboost():
    print("=" * 75)
    print("NETWORKPULSE: MODEL B — BEST-MAE XGBOOST [Quantile Regressor q=0.46]")
    print("=" * 75)

    model_dir = os.path.dirname(__file__)
    csv_path = os.path.join(REPO_ROOT, "drone_communication_dataset.csv")

    # 1. Ingestion
    print("\n>>> 1. Loading Telemetry Dataset...")
    df, meta = load_and_validate_dataset(csv_path)
    total_len = len(df)
    train_end = int(total_len * 0.80)
    print(f"  Total records: {total_len:,}")
    print(f"  Chronological train end: {df.index[train_end - 1]} ({train_end:,} rows)")

    # 2. Forecaster Initialization
    print("\n>>> 2. Initializing XGBoost Quantile (q=0.46) Forecaster...")
    model_path = os.path.join(model_dir, "best_mae_xgboost_model.json")
    forecaster = BestMaeXGBoostForecaster(model_path=model_path, quantile_alpha=0.46)
    print(f"  Model Family:    {forecaster.model_family}")
    print(f"  Model Name:      {forecaster.model_name}")
    print(f"  Quantile Alpha:  {forecaster.quantile_alpha}")
    print(f"  Loaded Artifact: {forecaster.model_path}")
    print(f"  Strategy:        Recursive 24-step forecasting with dynamic lag and rolling statistic updates")

    # 3. Primary 24-Hour Evaluation
    print("\n>>> 3. Evaluating Primary 24-Hour Horizon (2023-12-19 15:00:00 to 2023-12-20 14:00:00)...")
    history = df.iloc[:train_end]
    horizon = df.iloc[train_end : train_end + 24]

    result = forecaster.forecast_24h(history_df=history, horizon_df=horizon, steps=24)
    metrics_p24 = result.metrics

    print("  Primary 24-Hour Forecast Metrics:")
    print(f"    MAE:   {metrics_p24['MAE']:.2f} ms")
    print(f"    RMSE:  {metrics_p24['RMSE']:.2f} ms")
    print(f"    sMAPE: {metrics_p24['sMAPE']:.2f}%")
    print(f"    MAPE:  {metrics_p24['MAPE']:.2f}%")

    # 4. Multi-Week Rolling Benchmark (28 windows = 672 hours)
    print("\n>>> 4. Evaluating Multi-Week Benchmark (28 Rolling Horizons = 672 Hours)...")
    seasons_starts = [train_end, train_end + 24*7*9, train_end + 24*7*19, train_end + 24*7*26]
    all_maes = []
    all_rmses = []
    all_y_true = []
    all_y_pred = []

    for s_start in seasons_starts:
        for d in range(7):
            origin_idx = s_start + d * 24
            hist_d = df.iloc[:origin_idx]
            horiz_d = df.iloc[origin_idx : origin_idx + 24]
            res_d = forecaster.forecast_24h(history_df=hist_d, horizon_df=horiz_d, steps=24)
            y_t = horiz_d["round_trip_time"].values
            y_p = res_d.predictions_ms
            all_y_true.extend(y_t)
            all_y_pred.extend(y_p)
            all_maes.append(float(np.mean(np.abs(y_t - y_p))))
            all_rmses.append(float(np.sqrt(np.mean((y_t - y_p) ** 2))))

    y_true_all = np.array(all_y_true)
    y_pred_all = np.array(all_y_pred)
    micro_mae = float(np.mean(np.abs(y_true_all - y_pred_all)))
    micro_rmse = float(np.sqrt(np.mean((y_true_all - y_pred_all) ** 2)))
    macro_mae = float(np.mean(all_maes))
    macro_std = float(np.std(all_maes))

    print("\n  Multi-Week Benchmark Summary (672 Hours Across 4 Seasons):")
    print(f"    Micro (Pooled) MAE:  {micro_mae:.2f} ms")
    print(f"    Micro (Pooled) RMSE: {micro_rmse:.2f} ms")
    print(f"    Macro (Window Mean): {macro_mae:.2f} ms +/- {macro_std:.2f} ms")

    # 5. Save Configuration Metadata
    config_dict = {
        "model_family": "XGBoost",
        "model_name": "XGBoost Quantile (q=0.46)",
        "objective": "reg:quantileerror",
        "quantile_alpha": 0.46,
        "feature_count": 82,
        "validation_mae_ms": 47.4009,
        "primary_24h_metrics": metrics_p24,
        "multi_week_benchmark_metrics": {
            "micro_mae_ms": micro_mae,
            "micro_rmse_ms": micro_rmse,
            "macro_mae_ms": macro_mae,
            "macro_std_ms": macro_std
        },
        "status": "ACTIVE_PRODUCTION_MODEL_B"
    }

    config_path = os.path.join(model_dir, "best_mae_xgboost_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"\nSaved updated configuration to {config_path}")
    print("=" * 75)
    print("MODEL B EXECUTION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    run_best_mae_xgboost()
