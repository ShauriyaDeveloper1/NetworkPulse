"""
NetworkPulse: Joint ARIMA and XGBoost MAE Optimization Module
Provides unified data partitioning, ARIMA grid search and log-transformed modeling,
XGBoost L1-loss optimization, recursive multi-step forecasting, and multi-model benchmarking.
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

from src.evaluation import calculate_metrics
from src.optimized_xgboost import (
    create_optimized_features,
    prepare_optimized_partitions,
    train_optimized_xgboost,
    recursive_forecast_24h_optimized
)


def evaluate_arima_validation_grid(
    target_series: pd.Series,
    val_origins: List[int],
    orders: List[Tuple[int, int, int]],
    transforms: List[Optional[str]] = [None, "log"],
    steps: int = 24
) -> pd.DataFrame:
    """
    Evaluates ARIMA orders and transformations across chronological validation origins.
    Computes validation MAE, RMSE, and sMAPE strictly using data prior to each origin.
    """
    records = []
    for transform in transforms:
        for order in orders:
            order_maes, order_rmses, order_smapes = [], [], []
            has_error = False
            for origin in val_origins:
                y_true = target_series.iloc[origin : origin + steps].values
                if len(y_true) < steps:
                    continue
                history = target_series.iloc[:origin]
                
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if transform == "log":
                            fit_series = np.log(history)
                            model = ARIMA(fit_series, order=order).fit()
                            pred_raw = model.forecast(steps=steps).values
                            pred = np.clip(np.exp(pred_raw), 50.0, 500.0)
                        else:
                            model = ARIMA(history, order=order).fit()
                            pred_raw = model.forecast(steps=steps).values
                            pred = np.clip(pred_raw, 50.0, 500.0)
                            
                    m = calculate_metrics(y_true, pred)
                    order_maes.append(m["MAE"])
                    order_rmses.append(m["RMSE"])
                    order_smapes.append(m["sMAPE"])
                except Exception:
                    has_error = True
                    break
                    
            if not has_error and order_maes:
                name = f"Log-ARIMA{order}" if transform == "log" else f"ARIMA{order}"
                records.append({
                    "model_name": name,
                    "order": str(order),
                    "transform": transform if transform else "none",
                    "val_mae_mean": float(np.mean(order_maes)),
                    "val_rmse_mean": float(np.mean(order_rmses)),
                    "val_smape_mean": float(np.mean(order_smapes)),
                    "val_mae_std": float(np.std(order_maes))
                })
                
    res_df = pd.DataFrame(records).sort_values("val_mae_mean").reset_index(drop=True)
    return res_df


def fit_and_forecast_arima(
    history_series: pd.Series,
    order: Tuple[int, int, int] = (1, 0, 0),
    transform: Optional[str] = "log",
    steps: int = 24
) -> np.ndarray:
    """
    Fits ARIMA model on past history and produces multi-step forecasts with proper clipping.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if transform == "log":
            fit_series = np.log(history_series)
            model = ARIMA(fit_series, order=order).fit()
            pred_raw = model.forecast(steps=steps).values
            pred = np.clip(np.exp(pred_raw), 50.0, 500.0)
        else:
            model = ARIMA(history_series, order=order).fit()
            pred_raw = model.forecast(steps=steps).values
            pred = np.clip(pred_raw, 50.0, 500.0)
            
    return pred


def forecast_persistence(
    history_series: pd.Series,
    steps: int = 24
) -> np.ndarray:
    """
    Generates persistence forecast repeating the last observed value.
    """
    last_val = history_series.iloc[-1]
    return np.full(steps, float(last_val))


def generate_hybrid_forecast(
    pred_arima: np.ndarray,
    pred_xgboost: np.ndarray,
    weight_arima: float = 0.3
) -> np.ndarray:
    """
    Combines ARIMA and XGBoost forecasts via linear weighting:
    y_hybrid = w * y_arima + (1 - w) * y_xgboost.
    """
    return weight_arima * pred_arima + (1.0 - weight_arima) * pred_xgboost


def run_joint_rolling_benchmark(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_cols: List[str],
    xgb_model: xgb.XGBRegressor,
    start_idx: int,
    n_days: int = 7,
    horizon: int = 24,
    target_col: str = "round_trip_time",
    arima_order: Tuple[int, int, int] = (1, 0, 0),
    arima_transform: str = "log",
    hybrid_weight: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes a rolling-origin evaluation over consecutive 24-hour windows.
    At each origin, strictly uses data prior to that origin for all models.
    Returns:
    1. Detailed prediction DataFrame with actuals and predictions per timestamp.
    2. Daily metrics comparison summary DataFrame.
    """
    target = df[target_col]
    pred_records = []
    daily_metrics = []

    for day in range(n_days):
        origin = start_idx + day * horizon
        origin_timestamp = df.index[origin]
        timestamps_horizon = df.index[origin : origin + horizon]
        y_true = target.iloc[origin : origin + horizon].values
        
        # 1. Persistence
        pred_pers = forecast_persistence(target.iloc[:origin], steps=horizon)
        
        # 2. Baseline ARIMA(0,1,1) in levels (fitted on rolling recent window for stability/speed)
        # Using 2,000 recent observations represents ~83 days of operational context
        hist_window = target.iloc[max(0, origin - 2000) : origin]
        pred_arima_base = fit_and_forecast_arima(hist_window, order=(0, 1, 1), transform=None, steps=horizon)
        
        # 3. Optimized Log-ARIMA
        pred_arima_opt = fit_and_forecast_arima(hist_window, order=arima_order, transform=arima_transform, steps=horizon)
        
        # 4. Optimized XGBoost
        history_features = feature_df.iloc[:origin]
        test_horizon_features = feature_df.iloc[origin : origin + horizon]
        pred_xgb_series = recursive_forecast_24h_optimized(
            xgb_model, history_features, test_horizon_features, feature_cols, steps=horizon
        )
        pred_xgb = pred_xgb_series.values
        
        # 5. Hybrid
        pred_hybrid = generate_hybrid_forecast(pred_arima_opt, pred_xgb, weight_arima=hybrid_weight)
        
        # Store predictions
        for step in range(horizon):
            pred_records.append({
                "day": day + 1,
                "step": step + 1,
                "timestamp": timestamps_horizon[step],
                "actual_round_trip_time": y_true[step],
                "persistence": pred_pers[step],
                "arima_base_011": pred_arima_base[step],
                "arima_optimized_log": pred_arima_opt[step],
                "xgboost_optimized": pred_xgb[step],
                "hybrid": pred_hybrid[step]
            })
            
        # Compute metrics for this 24-hour window
        m_pers = calculate_metrics(y_true, pred_pers)
        m_base = calculate_metrics(y_true, pred_arima_base)
        m_opt_a = calculate_metrics(y_true, pred_arima_opt)
        m_xgb = calculate_metrics(y_true, pred_xgb)
        m_hyb = calculate_metrics(y_true, pred_hybrid)
        
        daily_metrics.append({
            "day": day + 1,
            "date_range": f"{timestamps_horizon[0].strftime('%Y-%m-%d')} -> {timestamps_horizon[-1].strftime('%Y-%m-%d')}",
            "actual_mean_rtt": float(np.mean(y_true)),
            "persistence_mae": m_pers["MAE"],
            "arima_base_mae": m_base["MAE"],
            "arima_opt_log_mae": m_opt_a["MAE"],
            "xgboost_opt_mae": m_xgb["MAE"],
            "hybrid_mae": m_hyb["MAE"],
            "persistence_rmse": m_pers["RMSE"],
            "arima_base_rmse": m_base["RMSE"],
            "arima_opt_log_rmse": m_opt_a["RMSE"],
            "xgboost_opt_rmse": m_xgb["RMSE"],
            "hybrid_rmse": m_hyb["RMSE"],
            "persistence_smape": m_pers["sMAPE"],
            "arima_base_smape": m_base["sMAPE"],
            "arima_opt_log_smape": m_opt_a["sMAPE"],
            "xgboost_opt_smape": m_xgb["sMAPE"],
            "hybrid_smape": m_hyb["sMAPE"],
        })
        
    predictions_df = pd.DataFrame(pred_records)
    metrics_df = pd.DataFrame(daily_metrics)
    return predictions_df, metrics_df
