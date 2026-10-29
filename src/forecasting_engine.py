"""
NetworkPulse: Unified Two-Model Forecasting Engine
Defines the final active production forecasting architecture with exactly two models:
  1. Model A: Best-MAE ARIMA (Log-ARIMA(1,0,0) with naive exponential inversion)
  2. Model B: Best-MAE XGBoost (Quantile Regressor q=0.46 with recursive multi-step forecasting)

Provides standardized inference interfaces, common prediction format, and fair evaluation tools.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

# Ensure project root is available
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluation import calculate_metrics


def create_optimized_features(
    df: pd.DataFrame,
    target_col: str = "round_trip_time"
) -> pd.DataFrame:
    """
    Constructs an enhanced set of past-only time-series features:
    - Target lags: 1, 2, 3, 6, 12, 24
    - Target differences: lag1 - lag2, lag1 - lag25
    - Multi-window rolling statistics: 3h, 6h, 12h, 24h, 48h mean; 6h, 24h, 48h std
    - Exponentially weighted moving averages (EWMA): spans 6, 24, 48
    - Rolling min, max, and range over 24h
    - Exogenous network telemetry lags and rolling means
    - Cyclical calendar encodings
    - Strict exclusion of device identifiers and retrospective attack labels
    """
    feat_df = df.copy()

    # 1. Deterministic calendar and cyclical features
    feat_df["hour"] = feat_df.index.hour
    feat_df["dayofweek"] = feat_df.index.dayofweek
    feat_df["is_weekend"] = feat_df["dayofweek"].isin([5, 6]).astype(int)
    feat_df["month"] = feat_df.index.month

    feat_df["hour_sin"] = np.sin(2 * np.pi * feat_df["hour"] / 24.0)
    feat_df["hour_cos"] = np.cos(2 * np.pi * feat_df["hour"] / 24.0)
    feat_df["dayofweek_sin"] = np.sin(2 * np.pi * feat_df["dayofweek"] / 7.0)
    feat_df["dayofweek_cos"] = np.cos(2 * np.pi * feat_df["dayofweek"] / 7.0)

    # 2. Target lags (strictly shifted)
    target = feat_df[target_col]
    t_shift = target.shift(1)

    for lag in [1, 2, 3, 6, 12, 24]:
        feat_df[f"rtt_lag_{lag}"] = target.shift(lag)

    # 3. Target differences
    feat_df["rtt_diff_1"] = target.shift(1) - target.shift(2)
    feat_df["rtt_diff_24"] = target.shift(1) - target.shift(25)

    # 4. Multi-window rolling statistics
    for w in [3, 6, 12, 24, 48]:
        feat_df[f"rtt_rolling_mean_{w}h"] = t_shift.rolling(w, min_periods=1).mean()
        if w in [6, 24, 48]:
            feat_df[f"rtt_rolling_std_{w}h"] = t_shift.rolling(w, min_periods=1).std().fillna(0.0)

    # 5. Exponentially Weighted Moving Averages (EWMA)
    for span in [6, 24, 48]:
        feat_df[f"rtt_ewma_{span}"] = t_shift.ewm(span=span).mean()

    # 6. Rolling extremes & range
    feat_df["rtt_rolling_min_24h"] = t_shift.rolling(24, min_periods=1).min()
    feat_df["rtt_rolling_max_24h"] = t_shift.rolling(24, min_periods=1).max()
    feat_df["rtt_rolling_range_24h"] = feat_df["rtt_rolling_max_24h"] - feat_df["rtt_rolling_min_24h"]

    # 7. Exogenous network telemetry lags & rolling means
    exog_cols = [
        "packet_loss_rate", "signal_strength", "signal_noise_ratio",
        "network_traffic_volume", "base_station_load", "uplink_downlink_quality",
        "sequence_number_gap", "data_rate", "transmission_power"
    ]
    for c in exog_cols:
        if c in feat_df.columns:
            c_shift = feat_df[c].shift(1)
            feat_df[f"{c}_lag_1"] = c_shift
            feat_df[f"{c}_rolling_mean_6h"] = c_shift.rolling(6, min_periods=1).mean()
            feat_df[f"{c}_rolling_mean_24h"] = c_shift.rolling(24, min_periods=1).mean()

    # 8. Categorical one-hot encodings
    for c in ["communication_protocol", "encryption_type"]:
        if c in feat_df.columns:
            dummies = pd.get_dummies(feat_df[c], prefix=c, drop_first=True, dtype=int)
            feat_df = pd.concat([feat_df, dummies], axis=1)

    # 9. Drop non-predictors, identifiers, and retrospective ground-truth attack labels
    drop_cols = [
        "timestamp", "drone_identification", "drone_gps_coordinates",
        "communication_protocol", "encryption_type", "label_normal",
        "label_spoofing", "label_mitm", "label_ddos", "label_gps_spoofing",
        "label_malware", "label_jamming", "label_protocol_exploit"
    ]
    existing_drops = [c for c in drop_cols if c in feat_df.columns]
    feat_df = feat_df.drop(columns=existing_drops)

    return feat_df


def recursive_forecast_24h_optimized(
    model: xgb.XGBRegressor,
    history_df: pd.DataFrame,
    test_horizon_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "round_trip_time",
    steps: int = 24
) -> pd.Series:
    """
    Executes a 24-step recursive forecast with dynamically updated lags,
    rolling statistics, EWMA, and differences. Zero target leakage.
    """
    recent_history = history_df[[target_col]].copy().iloc[-48:]
    forecast_timestamps = test_horizon_df.index[:steps]
    working_rtt = list(recent_history[target_col].values)
    predictions = []

    for step_idx in range(steps):
        cur_time = forecast_timestamps[step_idx]
        row_dict = {}

        # Calendar features
        hr = cur_time.hour
        dow = cur_time.dayofweek
        row_dict["hour"] = hr
        row_dict["dayofweek"] = dow
        row_dict["is_weekend"] = int(dow in [5, 6])
        row_dict["month"] = cur_time.month
        row_dict["hour_sin"] = np.sin(2 * np.pi * hr / 24.0)
        row_dict["hour_cos"] = np.cos(2 * np.pi * hr / 24.0)
        row_dict["dayofweek_sin"] = np.sin(2 * np.pi * dow / 7.0)
        row_dict["dayofweek_cos"] = np.cos(2 * np.pi * dow / 7.0)

        # Target lags
        for lag in [1, 2, 3, 6, 12, 24]:
            row_dict[f"rtt_lag_{lag}"] = working_rtt[-lag]

        # Target differences
        row_dict["rtt_diff_1"] = working_rtt[-1] - working_rtt[-2]
        row_dict["rtt_diff_24"] = working_rtt[-1] - working_rtt[-25]

        # Multi-window rolling statistics
        for w in [3, 6, 12, 24, 48]:
            row_dict[f"rtt_rolling_mean_{w}h"] = float(np.mean(working_rtt[-w:]))
            if w in [6, 24, 48]:
                row_dict[f"rtt_rolling_std_{w}h"] = float(np.std(working_rtt[-w:]))

        # EWMA
        for span in [6, 24, 48]:
            s_series = pd.Series(working_rtt)
            row_dict[f"rtt_ewma_{span}"] = float(s_series.ewm(span=span).mean().iloc[-1])

        # Rolling min, max, range
        past_24 = working_rtt[-24:]
        row_dict["rtt_rolling_min_24h"] = float(np.min(past_24))
        row_dict["rtt_rolling_max_24h"] = float(np.max(past_24))
        row_dict["rtt_rolling_range_24h"] = float(np.max(past_24) - np.min(past_24))

        # Exogenous telemetry from test horizon
        for col in feature_cols:
            if col not in row_dict:
                row_dict[col] = test_horizon_df[col].iloc[step_idx]

        step_df = pd.DataFrame([row_dict])[feature_cols]
        pred_val = float(model.predict(step_df)[0])
        pred_val = max(50.0, pred_val)

        predictions.append(pred_val)
        working_rtt.append(pred_val)

    return pd.Series(predictions, index=forecast_timestamps, name="optimized_xgboost_forecast")



@dataclass
class ForecastResult:
    """Standardized forecast container for all NetworkPulse forecasting models."""
    model_name: str
    model_family: str
    origin_timestamp: pd.Timestamp
    timestamps: pd.DatetimeIndex
    predictions_ms: np.ndarray
    ground_truth_ms: Optional[np.ndarray] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts predictions into a standardized tabular format."""
        df_out = pd.DataFrame({
            "timestamp": self.timestamps,
            "predicted_rtt_ms": self.predictions_ms
        })
        if self.ground_truth_ms is not None:
            df_out["actual_rtt_ms"] = self.ground_truth_ms
            df_out["absolute_error_ms"] = np.abs(self.ground_truth_ms - self.predictions_ms)
            df_out["squared_error_ms2"] = (self.ground_truth_ms - self.predictions_ms) ** 2
        return df_out


class BaseForecaster(ABC):
    """Abstract Base Class for NetworkPulse active forecasting models."""

    def __init__(self, model_name: str, model_family: str):
        self.model_name = model_name
        self.model_family = model_family

    @abstractmethod
    def forecast_24h(
        self,
        history_df: pd.DataFrame,
        horizon_df: Optional[pd.DataFrame] = None,
        target_col: str = "round_trip_time",
        steps: int = 24
    ) -> ForecastResult:
        """Generates a 24-hour horizon forecast given historical telemetry prior to origin."""
        pass


class BestMaeARIMAForecaster(BaseForecaster):
    """
    Model A: Best-MAE ARIMA Forecaster
    Architecture: Log-ARIMA(1,0,0) with differencing d=0 and naive exponential inversion.
    Theoretical Principle:
      Fitting ln(RTT) maps additive Gaussian noise to multiplicative log-normal deviations.
      Naive exponential inversion exp(z) produces the conditional median (~50 ms),
      strictly outperforming standard levels ARIMA(0,1,1) on MAE.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 0, 0),
        transformation: str = "log",
        recent_window: int = 2000,
        clip_bounds: Tuple[float, float] = (50.0, 500.0)
    ):
        super().__init__(model_name="Log-ARIMA(1,0,0)", model_family="ARIMA")
        self.order = order
        self.transformation = transformation
        self.recent_window = recent_window
        self.clip_bounds = clip_bounds
        self.config_metadata = {
            "model_type": "Log-ARIMA(1,0,0)",
            "order": list(self.order),
            "transformation": self.transformation,
            "differencing": f"d={self.order[1]} (ADF p < 0.0001 proves level stationarity)",
            "inversion_strategy": "naive_exp_targeting_conditional_geometric_median",
            "inversion_math": "Naive exp(y) targets conditional median (50 ms); Jensen's correction targets mean (100 ms)",
            "validation_mae_ms": 55.82846,
            "multi_week_test_mae_ms": 60.78736
        }

    def forecast_24h(
        self,
        history_df: pd.DataFrame,
        horizon_df: Optional[pd.DataFrame] = None,
        target_col: str = "round_trip_time",
        steps: int = 24
    ) -> ForecastResult:
        if target_col not in history_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in history_df.")

        origin_timestamp = history_df.index[-1]
        hist_target = history_df[target_col]

        # Use recent training window for efficiency and local regime stability
        if self.recent_window is not None and len(hist_target) > self.recent_window:
            hist_target = hist_target.iloc[-self.recent_window:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.transformation == "log":
                log_series = np.log(hist_target.astype(float))
                model = ARIMA(log_series, order=self.order).fit()
                raw_pred = model.forecast(steps=steps).values
                # Naive exponential inversion strictly optimal for MAE
                preds = np.clip(np.exp(raw_pred), self.clip_bounds[0], self.clip_bounds[1])
            else:
                model = ARIMA(hist_target.astype(float), order=self.order).fit()
                raw_pred = model.forecast(steps=steps).values
                preds = np.clip(raw_pred, self.clip_bounds[0], self.clip_bounds[1])

        # Generate forecast timestamps (1-hour frequency)
        if horizon_df is not None and len(horizon_df) >= steps:
            timestamps = horizon_df.index[:steps]
            ground_truth = horizon_df[target_col].iloc[:steps].values if target_col in horizon_df.columns else None
        else:
            timestamps = pd.date_range(origin_timestamp + pd.Timedelta(hours=1), periods=steps, freq="h")
            ground_truth = None

        metrics = {}
        if ground_truth is not None:
            metrics = calculate_metrics(ground_truth, preds)

        return ForecastResult(
            model_name=self.model_name,
            model_family=self.model_family,
            origin_timestamp=origin_timestamp,
            timestamps=timestamps,
            predictions_ms=preds,
            ground_truth_ms=ground_truth,
            metrics=metrics,
            metadata=self.config_metadata
        )


class BestMaeXGBoostForecaster(BaseForecaster):
    """
    Model B: Best-MAE XGBoost Forecaster
    Architecture: XGBoost Quantile Regressor (q=0.46, max_depth=3, lr=0.03) with recursive forecasting.
    Theoretical Principle:
      Pinball loss with alpha=0.46 penalizes positive over-predictions slightly more than
      under-predictions, pinning baseline predictions firmly to the 50 ms nominal floor
      while preserving predictive sensitivity to acute bursts.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        quantile_alpha: float = 0.46,
        clip_bounds: Tuple[float, float] = (50.0, 500.0)
    ):
        super().__init__(model_name="XGBoost Quantile (q=0.46)", model_family="XGBoost")
        self.quantile_alpha = quantile_alpha
        self.clip_bounds = clip_bounds
        self.model = xgb.XGBRegressor()

        # Resolve model path
        resolved_path = None
        candidate_paths = [
            model_path,
            os.path.join(REPO_ROOT, "models", "xgboost", "best_mae_xgboost_model.json"),
            os.path.join(REPO_ROOT, "models", "xgboost", "best_mae_model.json"),
            os.path.join(REPO_ROOT, "artifacts", "final_two_model_architecture", "best_mae_xgboost_model.json"),
            os.path.join(REPO_ROOT, "artifacts", "dual_objective_optimization", "best_mae_model.json")
        ]
        for p in candidate_paths:
            if p and os.path.exists(p):
                resolved_path = p
                break

        if resolved_path:
            self.model.load_model(resolved_path)
            self.model_path = resolved_path
        else:
            self.model_path = None

        self.config_metadata = {
            "model_type": "XGBoost Quantile Regressor",
            "objective": "reg:quantileerror",
            "quantile_alpha": self.quantile_alpha,
            "max_depth": 3,
            "learning_rate": 0.03,
            "n_estimators": 300,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "validation_mae_ms": 47.40085,
            "multi_week_test_mae_ms": 47.85421,
            "forecasting_strategy": "Recursive 24-step forecasting with dynamic lag and rolling statistic updates"
        }

    def forecast_24h(
        self,
        history_df: pd.DataFrame,
        horizon_df: Optional[pd.DataFrame] = None,
        target_col: str = "round_trip_time",
        steps: int = 24
    ) -> ForecastResult:
        if self.model is None:
            raise RuntimeError("XGBoost model is not loaded. Provide a valid model_path.")

        origin_timestamp = history_df.index[-1]

        # Combine history and horizon for feature engineering
        if horizon_df is not None:
            combined_df = pd.concat([history_df, horizon_df])
            test_horizon_features = horizon_df.iloc[:steps]
            timestamps = horizon_df.index[:steps]
            ground_truth = horizon_df[target_col].iloc[:steps].values if target_col in horizon_df.columns else None
        else:
            timestamps = pd.date_range(origin_timestamp + pd.Timedelta(hours=1), periods=steps, freq="h")
            # Build dummy horizon frame with known calendar features
            dummy_horizon = pd.DataFrame(index=timestamps)
            for c in history_df.columns:
                if c != target_col:
                    dummy_horizon[c] = history_df[c].iloc[-1]
                else:
                    dummy_horizon[c] = np.nan
            combined_df = pd.concat([history_df, dummy_horizon])
            test_horizon_features = dummy_horizon
            ground_truth = None

        # Build feature matrix
        df_feat = create_optimized_features(combined_df)
        feature_cols = [c for c in df_feat.columns if c != target_col]
        test_features_sub = df_feat.loc[timestamps, feature_cols]

        # Execute multi-step recursive forecast
        preds_series = recursive_forecast_24h_optimized(
            model=self.model,
            history_df=history_df,
            test_horizon_df=test_features_sub,
            feature_cols=feature_cols,
            target_col=target_col,
            steps=steps
        )
        preds = np.clip(preds_series.values, self.clip_bounds[0], self.clip_bounds[1])

        metrics = {}
        if ground_truth is not None:
            metrics = calculate_metrics(ground_truth, preds)

        return ForecastResult(
            model_name=self.model_name,
            model_family=self.model_family,
            origin_timestamp=origin_timestamp,
            timestamps=timestamps,
            predictions_ms=preds,
            ground_truth_ms=ground_truth,
            metrics=metrics,
            metadata=self.config_metadata
        )


class TwoModelForecastingPipeline:
    """
    Final NetworkPulse Active Forecasting Architecture.
    Encapsulates strictly the two best-MAE models:
      1. Model A: Best-MAE ARIMA (Log-ARIMA(1,0,0))
      2. Model B: Best-MAE XGBoost (Quantile Regressor q=0.46)
    All competing models (MSE champions, ensembles, naive baselines) are excluded.
    """

    def __init__(
        self,
        arima_forecaster: Optional[BestMaeARIMAForecaster] = None,
        xgboost_forecaster: Optional[BestMaeXGBoostForecaster] = None
    ):
        self.arima = arima_forecaster or BestMaeARIMAForecaster()
        self.xgboost = xgboost_forecaster or BestMaeXGBoostForecaster()

    def forecast_24h(
        self,
        df: pd.DataFrame,
        origin_idx: int,
        steps: int = 24,
        target_col: str = "round_trip_time"
    ) -> pd.DataFrame:
        """
        Generates aligned 24-hour forecasts for both Model A and Model B from a single forecast origin.
        Ensures identical origin timestamps, horizon steps, and evaluation conditions.
        """
        history_df = df.iloc[:origin_idx]
        horizon_df = df.iloc[origin_idx : origin_idx + steps]
        origin_timestamp = history_df.index[-1]

        # Model A: ARIMA
        res_arima = self.arima.forecast_24h(
            history_df=history_df,
            horizon_df=horizon_df,
            target_col=target_col,
            steps=steps
        )

        # Model B: XGBoost
        res_xgb = self.xgboost.forecast_24h(
            history_df=history_df,
            horizon_df=horizon_df,
            target_col=target_col,
            steps=steps
        )

        # Build standardized comparison DataFrame
        actuals = horizon_df[target_col].values if target_col in horizon_df.columns else np.full(steps, np.nan)
        timestamps = horizon_df.index[:steps]

        df_comparison = pd.DataFrame({
            "step": np.arange(1, steps + 1),
            "timestamp": timestamps,
            "origin_timestamp": origin_timestamp,
            "actual_rtt_ms": actuals,
            "arima_pred_ms": res_arima.predictions_ms,
            "xgboost_pred_ms": res_xgb.predictions_ms,
            "arima_abs_err_ms": np.abs(actuals - res_arima.predictions_ms),
            "xgboost_abs_err_ms": np.abs(actuals - res_xgb.predictions_ms)
        })

        return df_comparison

    def evaluate_multi_window(
        self,
        df: pd.DataFrame,
        origin_indices: List[int],
        steps: int = 24,
        target_col: str = "round_trip_time"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Executes rolling multi-window benchmark across multiple 24-hour evaluation horizons.
        Returns:
          1. Detailed hourly prediction DataFrame
          2. Per-window daily summary metrics DataFrame
          3. Pooled Micro and Macro overall summary metrics
        """
        hourly_records = []
        window_records = []

        all_y_true = []
        all_y_arima = []
        all_y_xgb = []

        for w_idx, o_idx in enumerate(origin_indices, 1):
            df_window = self.forecast_24h(df, origin_idx=o_idx, steps=steps, target_col=target_col)
            df_window["window_id"] = w_idx
            hourly_records.append(df_window)

            y_true = df_window["actual_rtt_ms"].values
            y_ar = df_window["arima_pred_ms"].values
            y_xg = df_window["xgboost_pred_ms"].values

            all_y_true.extend(y_true)
            all_y_arima.extend(y_ar)
            all_y_xgb.extend(y_xg)

            # Window metrics
            m_ar = calculate_metrics(y_true, y_ar)
            m_xg = calculate_metrics(y_true, y_xg)

            window_records.append({
                "window_id": w_idx,
                "origin_timestamp": str(df_window["origin_timestamp"].iloc[0]),
                "start_timestamp": str(df_window["timestamp"].iloc[0]),
                "end_timestamp": str(df_window["timestamp"].iloc[-1]),
                "actual_mean_rtt": float(np.mean(y_true)),
                "actual_median_rtt": float(np.median(y_true)),
                "arima_mae_ms": m_ar["MAE"],
                "arima_rmse_ms": m_ar["RMSE"],
                "xgboost_mae_ms": m_xg["MAE"],
                "xgboost_rmse_ms": m_xg["RMSE"],
                "mae_advantage_xgboost_ms": m_ar["MAE"] - m_xg["MAE"]
            })

        df_all_hourly = pd.concat(hourly_records, ignore_index=True)
        df_daily_metrics = pd.DataFrame(window_records)

        # Pooled Micro & Macro Calculations
        y_true_arr = np.array(all_y_true)
        y_ar_arr = np.array(all_y_arima)
        y_xg_arr = np.array(all_y_xgb)

        summary_metrics = {
            "total_evaluated_hours": len(y_true_arr),
            "total_windows": len(origin_indices),
            "arima": {
                "micro_mae_ms": float(np.mean(np.abs(y_true_arr - y_ar_arr))),
                "micro_rmse_ms": float(np.sqrt(np.mean((y_true_arr - y_ar_arr) ** 2))),
                "micro_mse_ms2": float(np.mean((y_true_arr - y_ar_arr) ** 2)),
                "macro_mae_ms": float(df_daily_metrics["arima_mae_ms"].mean()),
                "macro_mae_std_ms": float(df_daily_metrics["arima_mae_ms"].std()),
                "macro_rmse_ms": float(df_daily_metrics["arima_rmse_ms"].mean())
            },
            "xgboost": {
                "micro_mae_ms": float(np.mean(np.abs(y_true_arr - y_xg_arr))),
                "micro_rmse_ms": float(np.sqrt(np.mean((y_true_arr - y_xg_arr) ** 2))),
                "micro_mse_ms2": float(np.mean((y_true_arr - y_xg_arr) ** 2)),
                "macro_mae_ms": float(df_daily_metrics["xgboost_mae_ms"].mean()),
                "macro_mae_std_ms": float(df_daily_metrics["xgboost_mae_ms"].std()),
                "macro_rmse_ms": float(df_daily_metrics["xgboost_rmse_ms"].mean())
            },
            "relative_performance": {
                "micro_mae_difference_ms": float(np.mean(np.abs(y_true_arr - y_ar_arr)) - np.mean(np.abs(y_true_arr - y_xg_arr))),
                "relative_mae_improvement_pct": float(
                    (np.mean(np.abs(y_true_arr - y_ar_arr)) - np.mean(np.abs(y_true_arr - y_xg_arr))) /
                    np.mean(np.abs(y_true_arr - y_ar_arr)) * 100.0
                ),
                "windows_won_by_xgboost": int((df_daily_metrics["xgboost_mae_ms"] < df_daily_metrics["arima_mae_ms"]).sum()),
                "windows_won_by_arima": int((df_daily_metrics["arima_mae_ms"] < df_daily_metrics["xgboost_mae_ms"]).sum())
            }
        }

        return df_all_hourly, df_daily_metrics, summary_metrics
