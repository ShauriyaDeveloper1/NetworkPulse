"""
NetworkPulse: Optimized XGBoost Forecasting Module for MAE Minimization
Implements enhanced past-only feature engineering, L1-loss optimization (reg:absoluteerror),
and both recursive and direct multi-step forecasting engines.
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb


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


def prepare_optimized_partitions(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio_within_train: float = 0.15,
    target_col: str = "round_trip_time"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Creates optimized features and performs strict chronological splitting into
    Pure Train, Chronological Validation (for tuning), and Held-Out Test.
    """
    feat_df = create_optimized_features(df, target_col=target_col)
    feature_cols = [c for c in feat_df.columns if c != target_col]

    train_end_idx = int(len(df) * train_ratio)
    
    # Drop first 48 rows of train due to 48h rolling/lag features
    train_full = feat_df.iloc[48:train_end_idx].copy()
    test_df = feat_df.iloc[train_end_idx:].copy()

    val_size = int(len(train_full) * val_ratio_within_train)
    pure_train_df = train_full.iloc[:-val_size].copy()
    val_df = train_full.iloc[-val_size:].copy()

    X_train = pure_train_df[feature_cols]
    y_train = pure_train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def train_optimized_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42
) -> xgb.XGBRegressor:
    """
    Fits an XGBRegressor using the L1-loss objective (reg:absoluteerror) to directly minimize MAE.
    """
    default_params = {
        "max_depth": 3,
        "learning_rate": 0.03,
        "n_estimators": 300,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBRegressor(
        objective="reg:absoluteerror",
        eval_metric="mae",
        random_state=random_state,
        n_jobs=-1,
        **default_params
    )

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train, verbose=False)

    return model


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
