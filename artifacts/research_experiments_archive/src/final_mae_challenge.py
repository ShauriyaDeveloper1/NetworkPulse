"""
NetworkPulse: Stage 2 Final MAE Minimization Challenge Module
Provides rigorous multi-baseline benchmarking, mathematical median targeting,
quantile-tuned gradient boosting, multi-week out-of-sample evaluation,
and micro/macro error calculation without data leakage.
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
    recursive_forecast_24h_optimized
)


def compute_macro_and_micro_metrics(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    window_length: int = 24
) -> Dict[str, float]:
    """
    Computes both Micro (overall pooled) and Macro (average of window-level) metrics.
    Explicitly distinguishes between pooled hourly error and window-average error.
    """
    n_total = len(y_true_all)
    n_windows = n_total // window_length
    
    # 1. Micro (Pooled over all hourly observations)
    micro_mae = float(np.mean(np.abs(y_true_all - y_pred_all)))
    micro_rmse = float(np.sqrt(np.mean((y_true_all - y_pred_all) ** 2)))
    denom = np.abs(y_true_all) + np.abs(y_pred_all)
    micro_smape = float(np.mean(200.0 * np.abs(y_true_all - y_pred_all) / np.where(denom == 0, 1e-8, denom)))
    
    # 2. Macro (Mean across window-level MAEs)
    window_maes = []
    window_rmses = []
    for w in range(n_windows):
        start = w * window_length
        end = start + window_length
        yt = y_true_all[start:end]
        yp = y_pred_all[start:end]
        window_maes.append(float(np.mean(np.abs(yt - yp))))
        window_rmses.append(float(np.sqrt(np.mean((yt - yp) ** 2))))
        
    macro_mae = float(np.mean(window_maes)) if window_maes else micro_mae
    macro_rmse = float(np.mean(window_rmses)) if window_rmses else micro_rmse
    macro_mae_std = float(np.std(window_maes)) if window_maes else 0.0

    return {
        "micro_mae": micro_mae,
        "micro_rmse": micro_rmse,
        "micro_smape": micro_smape,
        "macro_mae": macro_mae,
        "macro_rmse": macro_rmse,
        "macro_mae_std": macro_mae_std,
        "window_maes": window_maes
    }


def generate_baseline_forecasts(
    target_history: pd.Series,
    train_median: float = 50.0,
    steps: int = 24
) -> Dict[str, np.ndarray]:
    """
    Generates three distinct, leakage-free baselines:
    1. Last-Observation Persistence: y(t+h) = y(origin - 1)
    2. Fixed Train Median: y(t+h) = Median(y_train) = 50.0 ms
    3. Seasonal Naive: y(t+h) = y(origin - 24 + h)
    """
    last_obs = float(target_history.iloc[-1])
    pred_persistence = np.full(steps, last_obs)
    pred_fixed_median = np.full(steps, train_median)
    
    # Seasonal naive (lag 24)
    if len(target_history) >= steps:
        pred_seasonal_naive = target_history.iloc[-steps:].values.astype(float)
    else:
        pred_seasonal_naive = pred_persistence.copy()
        
    return {
        "persistence": pred_persistence,
        "fixed_median": pred_fixed_median,
        "seasonal_naive": pred_seasonal_naive
    }


def fit_and_forecast_log_arima(
    history_series: pd.Series,
    order: Tuple[int, int, int] = (1, 0, 0),
    steps: int = 24,
    recent_window: int = 2000
) -> np.ndarray:
    """
    Fits Log-ARIMA model on historical RTT and back-transforms using naive exp(y)
    to target the conditional median.
    """
    hist = history_series.iloc[max(0, len(history_series) - recent_window):]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_series = np.log(hist)
        model = ARIMA(log_series, order=order).fit()
        raw_pred = model.forecast(steps=steps).values
        pred = np.clip(np.exp(raw_pred), 50.0, 500.0)
    return pred


def train_quantile_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 0.48,
    random_state: int = 42
) -> xgb.XGBRegressor:
    """
    Fits an XGBRegressor using Quantile loss (reg:quantileerror) tuned on validation.
    Quantile alpha slightly below 0.50 (e.g. 0.48) penalizes over-prediction,
    pinning the nominal prediction closer to 50 ms while retaining sensitivity to spikes.
    """
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        eval_metric="mae",
        max_depth=3,
        learning_rate=0.03,
        n_estimators=300,
        reg_alpha=1.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def run_comprehensive_multi_week_benchmark(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_cols: List[str],
    xgb_l1: xgb.XGBRegressor,
    xgb_quantile: xgb.XGBRegressor,
    train_end_idx: int,
    target_col: str = "round_trip_time",
    horizon: int = 24
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Executes a comprehensive benchmark across:
    - Primary Week (Week 1: Dec 2023, Days 1-7)
    - Winter Week (Week 2: Jan 2024, Days 30-36)
    - Spring Week (Week 3: Mar 2024, Days 90-96)
    - Summer Week (Week 4: Jun 2024, Days 180-186)
    Total of 28 consecutive 24-hour horizons (672 continuous hours).
    
    Returns:
    1. Hourly prediction DataFrame
    2. Daily metrics DataFrame
    3. Multi-week overall summary metrics (Macro and Micro)
    """
    target = df[target_col]
    train_median = float(np.median(target.iloc[:train_end_idx]))

    test_week_definitions = [
        ("Week 1 (Launch: Dec 2023)", train_end_idx),
        ("Week 2 (Winter: Jan 2024)", train_end_idx + 30 * 24),
        ("Week 3 (Spring: Mar 2024)", train_end_idx + 90 * 24),
        ("Week 4 (Summer: Jun 2024)", train_end_idx + 180 * 24),
    ]

    hourly_records = []
    daily_records = []
    
    # Track overall arrays for micro MAE
    all_actuals = []
    model_predictions_all = {
        "persistence": [],
        "fixed_median": [],
        "seasonal_naive": [],
        "arima_base_011": [],
        "arima_opt_log": [],
        "xgboost_opt_l1": [],
        "xgboost_opt_quantile": []
    }

    global_day_counter = 0

    for week_label, week_start_idx in test_week_definitions:
        for day in range(7):
            global_day_counter += 1
            origin = week_start_idx + day * horizon
            timestamps_horizon = df.index[origin : origin + horizon]
            y_true = target.iloc[origin : origin + horizon].values
            
            # 1. Simple Baselines
            hist_series = target.iloc[:origin]
            baselines = generate_baseline_forecasts(hist_series, train_median=train_median, steps=horizon)
            p_pers = baselines["persistence"]
            p_fmed = baselines["fixed_median"]
            p_snaive = baselines["seasonal_naive"]
            
            # 2. Base ARIMA(0,1,1) in levels
            hist_window = target.iloc[max(0, origin - 2000) : origin]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m_base = ARIMA(hist_window, order=(0, 1, 1)).fit()
                p_arima_base = np.clip(m_base.forecast(steps=horizon).values, 50.0, 500.0)
                
            # 3. Optimized Log-ARIMA(1,0,0)
            p_arima_opt = fit_and_forecast_log_arima(hist_series, order=(1, 0, 0), steps=horizon)
            
            # 4. Optimized XGBoost L1
            p_xgb_l1 = recursive_forecast_24h_optimized(
                xgb_l1, feature_df.iloc[:origin], feature_df.iloc[origin : origin + horizon], feature_cols, steps=horizon
            ).values
            
            # 5. Optimized XGBoost Quantile
            p_xgb_quant = recursive_forecast_24h_optimized(
                xgb_quantile, feature_df.iloc[:origin], feature_df.iloc[origin : origin + horizon], feature_cols, steps=horizon
            ).values
            
            # Store hourly predictions
            for step in range(horizon):
                hourly_records.append({
                    "week_label": week_label,
                    "global_day": global_day_counter,
                    "day_in_week": day + 1,
                    "step": step + 1,
                    "timestamp": timestamps_horizon[step],
                    "actual_rtt": y_true[step],
                    "persistence": p_pers[step],
                    "fixed_median": p_fmed[step],
                    "seasonal_naive": p_snaive[step],
                    "arima_base_011": p_arima_base[step],
                    "arima_opt_log": p_arima_opt[step],
                    "xgboost_opt_l1": p_xgb_l1[step],
                    "xgboost_opt_quantile": p_xgb_quant[step]
                })

            # Append to overall
            all_actuals.extend(y_true)
            model_predictions_all["persistence"].extend(p_pers)
            model_predictions_all["fixed_median"].extend(p_fmed)
            model_predictions_all["seasonal_naive"].extend(p_snaive)
            model_predictions_all["arima_base_011"].extend(p_arima_base)
            model_predictions_all["arima_opt_log"].extend(p_arima_opt)
            model_predictions_all["xgboost_opt_l1"].extend(p_xgb_l1)
            model_predictions_all["xgboost_opt_quantile"].extend(p_xgb_quant)

            # Daily MAEs
            daily_records.append({
                "week_label": week_label,
                "global_day": global_day_counter,
                "day_in_week": day + 1,
                "date_range": f"{timestamps_horizon[0].strftime('%Y-%m-%d')} to {timestamps_horizon[-1].strftime('%Y-%m-%d')}",
                "actual_mean_rtt": float(np.mean(y_true)),
                "persistence_mae": calculate_metrics(y_true, p_pers)["MAE"],
                "fixed_median_mae": calculate_metrics(y_true, p_fmed)["MAE"],
                "seasonal_naive_mae": calculate_metrics(y_true, p_snaive)["MAE"],
                "arima_base_mae": calculate_metrics(y_true, p_arima_base)["MAE"],
                "arima_opt_log_mae": calculate_metrics(y_true, p_arima_opt)["MAE"],
                "xgboost_l1_mae": calculate_metrics(y_true, p_xgb_l1)["MAE"],
                "xgboost_quantile_mae": calculate_metrics(y_true, p_xgb_quant)["MAE"],
                "persistence_rmse": calculate_metrics(y_true, p_pers)["RMSE"],
                "fixed_median_rmse": calculate_metrics(y_true, p_fmed)["RMSE"],
                "arima_base_rmse": calculate_metrics(y_true, p_arima_base)["RMSE"],
                "arima_opt_log_rmse": calculate_metrics(y_true, p_arima_opt)["RMSE"],
                "xgboost_l1_rmse": calculate_metrics(y_true, p_xgb_l1)["RMSE"],
                "xgboost_quantile_rmse": calculate_metrics(y_true, p_xgb_quant)["RMSE"],
            })

    hourly_df = pd.DataFrame(hourly_records)
    daily_df = pd.DataFrame(daily_records)
    
    # Calculate Overall Multi-Week Summary
    y_true_arr = np.array(all_actuals)
    overall_summary = {}
    for m_name, preds_list in model_predictions_all.items():
        p_arr = np.array(preds_list)
        overall_summary[m_name] = compute_macro_and_micro_metrics(y_true_arr, p_arr, window_length=horizon)

    return hourly_df, daily_df, overall_summary
