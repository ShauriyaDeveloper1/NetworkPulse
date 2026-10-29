"""
NetworkPulse: Dual-Objective Forecast Optimization Module
Implements dedicated MAE-minimization (quantile pinball loss),
dedicated MSE/RMSE-minimization (squared-error regression),
and Pareto-efficient trade-off ensembling.
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
from src.optimized_xgboost import recursive_forecast_24h_optimized
from src.final_mae_challenge import generate_baseline_forecasts, fit_and_forecast_log_arima


def train_best_mae_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 0.48,
    random_state: int = 42
) -> xgb.XGBRegressor:
    """
    Trains the Champion MAE model using Quantile Pinball loss (q=0.48).
    Pins nominal predictions tightly to the 50 ms floor while maintaining spike sensitivity.
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


def train_best_rmse_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int = 4,
    learning_rate: float = 0.03,
    n_estimators: int = 300,
    reg_lambda: float = 1.0,
    random_state: int = 42
) -> xgb.XGBRegressor:
    """
    Trains the Champion MSE/RMSE model using Squared Error loss (reg:squarederror).
    Predicts the conditional mean, minimizing squared deviations on high-latency spikes.
    """
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        reg_alpha=0.0,
        reg_lambda=reg_lambda,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def generate_pareto_forecast(
    pred_mae: np.ndarray,
    pred_rmse: np.ndarray,
    weight_mae: float = 0.50
) -> np.ndarray:
    """
    Constructs a convex combination between the best MAE model and best RMSE model:
    y_pareto = w * y_mae + (1 - w) * y_rmse.
    """
    return weight_mae * pred_mae + (1.0 - weight_mae) * pred_rmse


def evaluate_pareto_validation_frontier(
    df: pd.DataFrame,
    df_feat: pd.DataFrame,
    feature_cols: List[str],
    model_mae: xgb.XGBRegressor,
    model_rmse: xgb.XGBRegressor,
    val_origins: List[int],
    weights: List[float] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    horizon: int = 24,
    target_col: str = "round_trip_time"
) -> pd.DataFrame:
    """
    Computes validation MAE, RMSE, and MSE across convex weights w in [0, 1]
    to map the empirical Pareto frontier.
    """
    target = df[target_col]
    records = []

    for w in weights:
        w_maes, w_rmses, w_mses = [], [], []
        for origin in val_origins:
            y_true = target.iloc[origin : origin + horizon].values
            h_df = df_feat.iloc[:origin]
            t_df = df_feat.iloc[origin : origin + horizon]
            
            p_m = recursive_forecast_24h_optimized(model_mae, h_df, t_df, feature_cols, steps=horizon).values
            p_r = recursive_forecast_24h_optimized(model_rmse, h_df, t_df, feature_cols, steps=horizon).values
            
            p_blend = w * p_m + (1.0 - w) * p_r
            m = calculate_metrics(y_true, p_blend)
            w_maes.append(m["MAE"])
            w_rmses.append(m["RMSE"])
            w_mses.append(m["RMSE"] ** 2)

        records.append({
            "weight_mae": float(w),
            "weight_rmse": float(round(1.0 - w, 2)),
            "val_mae_mean": float(np.mean(w_maes)),
            "val_rmse_mean": float(np.mean(w_rmses)),
            "val_mse_mean": float(np.mean(w_mses)),
            "val_mae_std": float(np.std(w_maes)),
            "val_rmse_std": float(np.std(w_rmses))
        })

    return pd.DataFrame(records)


def run_comprehensive_dual_benchmark(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_cols: List[str],
    model_mae: xgb.XGBRegressor,
    model_rmse: xgb.XGBRegressor,
    train_end_idx: int,
    balanced_weight: float = 0.50,
    target_col: str = "round_trip_time",
    horizon: int = 24
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the dual-objective multi-week benchmark across:
    - Primary Week (Week 1: Dec 2023)
    - Winter Week (Week 2: Jan 2024)
    - Spring Week (Week 3: Mar 2024)
    - Summer Week (Week 4: Jun 2024)
    Total 28 windows = 672 continuous test hours.
    
    Returns:
    1. Hourly predictions DataFrame
    2. Daily metrics DataFrame
    3. Multi-week overall pooled summary DataFrame
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

    all_actuals = []
    model_preds_pooled = {
        "persistence": [],
        "fixed_median": [],
        "seasonal_naive": [],
        "arima_base_011": [],
        "arima_opt_log": [],
        "best_mae_model": [],
        "best_rmse_model": [],
        "balanced_pareto_model": []
    }

    global_day_counter = 0

    for week_label, week_start in test_week_definitions:
        for day in range(7):
            global_day_counter += 1
            origin = week_start + day * horizon
            timestamps_horizon = df.index[origin : origin + horizon]
            y_true = target.iloc[origin : origin + horizon].values
            
            # 1. Baselines
            hist_series = target.iloc[:origin]
            baselines = generate_baseline_forecasts(hist_series, train_median=train_median, steps=horizon)
            p_pers = baselines["persistence"]
            p_fmed = baselines["fixed_median"]
            p_snaive = baselines["seasonal_naive"]
            
            # 2. Baseline ARIMA(0,1,1) in levels
            hist_window = target.iloc[max(0, origin - 2000) : origin]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m_base = ARIMA(hist_window, order=(0, 1, 1)).fit()
                p_arima_base = np.clip(m_base.forecast(steps=horizon).values, 50.0, 500.0)
                
            # 3. Optimized Log-ARIMA(1,0,0)
            p_arima_opt = fit_and_forecast_log_arima(hist_series, order=(1, 0, 0), steps=horizon)
            
            # 4. Best MAE Model
            p_mae = recursive_forecast_24h_optimized(
                model_mae, feature_df.iloc[:origin], feature_df.iloc[origin : origin + horizon], feature_cols, steps=horizon
            ).values
            
            # 5. Best RMSE Model
            p_rmse = recursive_forecast_24h_optimized(
                model_rmse, feature_df.iloc[:origin], feature_df.iloc[origin : origin + horizon], feature_cols, steps=horizon
            ).values
            
            # 6. Balanced Pareto Model
            p_pareto = generate_pareto_forecast(p_mae, p_rmse, weight_mae=balanced_weight)
            
            # Store hourly
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
                    "best_mae_model": p_mae[step],
                    "best_rmse_model": p_rmse[step],
                    "balanced_pareto_model": p_pareto[step]
                })

            # Append pooled
            all_actuals.extend(y_true)
            model_preds_pooled["persistence"].extend(p_pers)
            model_preds_pooled["fixed_median"].extend(p_fmed)
            model_preds_pooled["seasonal_naive"].extend(p_snaive)
            model_preds_pooled["arima_base_011"].extend(p_arima_base)
            model_preds_pooled["arima_opt_log"].extend(p_arima_opt)
            model_preds_pooled["best_mae_model"].extend(p_mae)
            model_preds_pooled["best_rmse_model"].extend(p_rmse)
            model_preds_pooled["balanced_pareto_model"].extend(p_pareto)

            # Store daily
            m_pers = calculate_metrics(y_true, p_pers)
            m_fmed = calculate_metrics(y_true, p_fmed)
            m_snaive = calculate_metrics(y_true, p_snaive)
            m_a_base = calculate_metrics(y_true, p_arima_base)
            m_a_opt = calculate_metrics(y_true, p_arima_opt)
            m_mae = calculate_metrics(y_true, p_mae)
            m_rmse = calculate_metrics(y_true, p_rmse)
            m_par = calculate_metrics(y_true, p_pareto)

            daily_records.append({
                "week_label": week_label,
                "global_day": global_day_counter,
                "day_in_week": day + 1,
                "date_range": f"{timestamps_horizon[0].strftime('%Y-%m-%d')} to {timestamps_horizon[-1].strftime('%Y-%m-%d')}",
                "actual_mean_rtt": float(np.mean(y_true)),
                "persistence_mae": m_pers["MAE"],
                "persistence_rmse": m_pers["RMSE"],
                "persistence_mse": m_pers["RMSE"]**2,
                "fixed_median_mae": m_fmed["MAE"],
                "fixed_median_rmse": m_fmed["RMSE"],
                "fixed_median_mse": m_fmed["RMSE"]**2,
                "seasonal_naive_mae": m_snaive["MAE"],
                "seasonal_naive_rmse": m_snaive["RMSE"],
                "seasonal_naive_mse": m_snaive["RMSE"]**2,
                "arima_base_mae": m_a_base["MAE"],
                "arima_base_rmse": m_a_base["RMSE"],
                "arima_base_mse": m_a_base["RMSE"]**2,
                "arima_opt_log_mae": m_a_opt["MAE"],
                "arima_opt_log_rmse": m_a_opt["RMSE"],
                "arima_opt_log_mse": m_a_opt["RMSE"]**2,
                "best_mae_model_mae": m_mae["MAE"],
                "best_mae_model_rmse": m_mae["RMSE"],
                "best_mae_model_mse": m_mae["RMSE"]**2,
                "best_rmse_model_mae": m_rmse["MAE"],
                "best_rmse_model_rmse": m_rmse["RMSE"],
                "best_rmse_model_mse": m_rmse["RMSE"]**2,
                "balanced_pareto_mae": m_par["MAE"],
                "balanced_pareto_rmse": m_par["RMSE"],
                "balanced_pareto_mse": m_par["RMSE"]**2,
            })

    hourly_df = pd.DataFrame(hourly_records)
    daily_df = pd.DataFrame(daily_records)

    # Compute overall multi-week pooled metrics
    y_true_all = np.array(all_actuals)
    summary_records = []
    for model_name, preds_list in model_preds_pooled.items():
        p_all = np.array(preds_list)
        m = calculate_metrics(y_true_all, p_all)
        mse = float(np.mean((y_true_all - p_all) ** 2))
        
        # Calculate macro window metrics
        win_maes = daily_df[f"{model_name}_mae"].values if f"{model_name}_mae" in daily_df.columns else []
        win_rmses = daily_df[f"{model_name}_rmse"].values if f"{model_name}_rmse" in daily_df.columns else []
        
        summary_records.append({
            "model_name": model_name,
            "Pooled_MAE": m["MAE"],
            "Pooled_RMSE": m["RMSE"],
            "Pooled_MSE": mse,
            "Pooled_sMAPE": m["sMAPE"],
            "Pooled_MAPE": m["MAPE"],
            "Macro_MAE_Mean": float(np.mean(win_maes)) if len(win_maes) else m["MAE"],
            "Macro_MAE_StdDev": float(np.std(win_maes)) if len(win_maes) else 0.0,
            "Macro_RMSE_Mean": float(np.mean(win_rmses)) if len(win_rmses) else m["RMSE"]
        })

    summary_df = pd.DataFrame(summary_records).sort_values("Pooled_MAE").reset_index(drop=True)
    return hourly_df, daily_df, summary_df
