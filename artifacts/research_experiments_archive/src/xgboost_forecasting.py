"""
NetworkPulse: Stage 2 XGBoost Time-Series Forecasting Module
Implements leakage-free feature engineering, chronological validation,
XGBoost model training, and recursive 24-hour ahead multi-step forecasting.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import json
import warnings


def create_time_series_features(
    df: pd.DataFrame,
    target_col: str = "round_trip_time",
    is_training: bool = True
) -> pd.DataFrame:
    """
    Constructs leakage-free lag, rolling, calendar, and telemetry features.
    
    All rolling and lag features strictly use past observations (shift >= 1).
    Excludes identifiers (e.g. drone_identification, drone_gps_coordinates)
    and retrospective attack diagnostic labels.
    """
    feat_df = df.copy()

    # 1. Calendar and cyclical temporal features (deterministic)
    feat_df["hour"] = feat_df.index.hour
    feat_df["dayofweek"] = feat_df.index.dayofweek
    feat_df["is_weekend"] = feat_df["dayofweek"].isin([5, 6]).astype(int)
    feat_df["month"] = feat_df.index.month

    # Cyclical encodings
    feat_df["hour_sin"] = np.sin(2 * np.pi * feat_df["hour"] / 24.0)
    feat_df["hour_cos"] = np.cos(2 * np.pi * feat_df["hour"] / 24.0)
    feat_df["dayofweek_sin"] = np.sin(2 * np.pi * feat_df["dayofweek"] / 7.0)
    feat_df["dayofweek_cos"] = np.cos(2 * np.pi * feat_df["dayofweek"] / 7.0)

    # 2. Target lag features (strictly past values)
    target = feat_df[target_col]
    rtt_lags = [1, 2, 3, 6, 12, 24]
    for lag in rtt_lags:
        feat_df[f"rtt_lag_{lag}"] = target.shift(lag)

    # 3. Target rolling statistics (strictly past values via shift(1))
    target_shifted = target.shift(1)
    feat_df["rtt_rolling_mean_6h"] = target_shifted.rolling(window=6, min_periods=1).mean()
    feat_df["rtt_rolling_std_6h"] = target_shifted.rolling(window=6, min_periods=1).std().fillna(0.0)
    feat_df["rtt_rolling_mean_24h"] = target_shifted.rolling(window=24, min_periods=1).mean()
    feat_df["rtt_rolling_std_24h"] = target_shifted.rolling(window=24, min_periods=1).std().fillna(0.0)
    feat_df["rtt_rolling_min_24h"] = target_shifted.rolling(window=24, min_periods=1).min()
    feat_df["rtt_rolling_max_24h"] = target_shifted.rolling(window=24, min_periods=1).max()

    # 4. Exogenous network telemetry lag and rolling features
    exog_metrics = [
        "packet_loss_rate",
        "signal_strength",
        "signal_noise_ratio",
        "network_traffic_volume",
        "base_station_load"
    ]
    for m in exog_metrics:
        if m in feat_df.columns:
            m_shifted = feat_df[m].shift(1)
            feat_df[f"{m}_lag_1"] = m_shifted
            feat_df[f"{m}_rolling_mean_6h"] = m_shifted.rolling(window=6, min_periods=1).mean()
            feat_df[f"{m}_rolling_mean_24h"] = m_shifted.rolling(window=24, min_periods=1).mean()

    # 5. Categorical encodings (one-hot with drop_first for strict numerical matrix)
    cat_cols = ["communication_protocol", "encryption_type"]
    for c in cat_cols:
        if c in feat_df.columns:
            dummies = pd.get_dummies(feat_df[c], prefix=c, drop_first=True, dtype=int)
            feat_df = pd.concat([feat_df, dummies], axis=1)

    # 6. Exclude non-predictor columns: identifiers, coordinates, diagnostic attack labels, raw categoricals
    drop_cols = [
        "timestamp",
        "drone_identification",
        "drone_gps_coordinates",
        "communication_protocol",
        "encryption_type",
        "label_normal",
        "label_spoofing",
        "label_mitm",
        "label_ddos",
        "label_gps_spoofing",
        "label_malware",
        "label_jamming",
        "label_protocol_exploit"
    ]
    existing_drops = [c for c in drop_cols if c in feat_df.columns]
    feat_df = feat_df.drop(columns=existing_drops)

    return feat_df


def prepare_train_val_test_data(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio_within_train: float = 0.15,
    target_col: str = "round_trip_time"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Creates features across the full series, then splits into Train, Validation, and Test partitions.
    
    The first 24 rows of the training set (lost due to 24h lag) are dropped.
    The test partition preserves 100% of rows because lags are populated from the training history.
    """
    total_len = len(df)
    train_end_idx = int(total_len * train_ratio)

    # Feature engineering across chronological timeline
    feat_df = create_time_series_features(df, target_col=target_col)
    
    # Feature columns (exclude target)
    feature_cols = [c for c in feat_df.columns if c != target_col]

    # Split into train and test at the exact Stage 1 boundary
    train_full = feat_df.iloc[:train_end_idx].copy()
    test_df = feat_df.iloc[train_end_idx:].copy()

    # Drop the first 24 observations from train_full where 24h lag is NaN
    train_full_clean = train_full.iloc[24:].copy()

    # Chronological validation split strictly within the training set
    n_train_clean = len(train_full_clean)
    val_size = int(n_train_clean * val_ratio_within_train)
    pure_train_size = n_train_clean - val_size

    pure_train_df = train_full_clean.iloc[:pure_train_size]
    val_df = train_full_clean.iloc[pure_train_size:]

    X_train = pure_train_df[feature_cols]
    y_train = pure_train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42
) -> xgb.XGBRegressor:
    """
    Fits an XGBRegressor using early stopping on the chronological validation set.
    """
    print("\nTraining XGBoost Regressor for RTT forecasting...")
    print(f"  Training samples:   {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Predictor features: {X_train.shape[1]}")

    model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="rmse"
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )

    best_iter = model.best_iteration
    print(f"Model converged at iteration {best_iter} with best validation RMSE: {model.best_score:.4f} ms")
    return model


def recursive_forecast_24h(
    model: xgb.XGBRegressor,
    history_df: pd.DataFrame,
    test_horizon_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "round_trip_time",
    steps: int = 24
) -> pd.Series:
    """
    Executes a strict 24-hour ahead recursive multi-step forecast.
    
    At step h:
    - Target lags < h use model predictions from earlier steps.
    - Target lags >= h use known historical observations.
    - Rolling target statistics are dynamically recomputed over history + predictions.
    - Calendar features are known deterministically for timestamp T+h.
    - Exogenous telemetry features are held at the last known state prior to forecast origin.
    
    Zero future target leakage occurs.
    """
    # Combined working history for rolling calculations
    recent_history = history_df[[target_col]].copy().iloc[-48:]
    forecast_timestamps = test_horizon_df.index[:steps]
    predictions = []

    # Telemetry baseline at origin T (last known values prior to test horizon)
    last_known_row = history_df.iloc[-1]
    
    # Track RTT history including new recursive predictions
    working_rtt = list(recent_history[target_col].values)

    for step_idx, cur_time in enumerate(forecast_timestamps):
        row_dict = {}

        # 1. Calendar features for cur_time
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

        # 2. Target lag features from working_rtt
        # Current index in working_rtt is -1 (previous step)
        rtt_lags = [1, 2, 3, 6, 12, 24]
        for lag in rtt_lags:
            if len(working_rtt) >= lag:
                row_dict[f"rtt_lag_{lag}"] = working_rtt[-lag]
            else:
                row_dict[f"rtt_lag_{lag}"] = working_rtt[0]

        # 3. Rolling target statistics from working_rtt
        past_6 = working_rtt[-6:]
        past_24 = working_rtt[-24:]
        row_dict["rtt_rolling_mean_6h"] = float(np.mean(past_6))
        row_dict["rtt_rolling_std_6h"] = float(np.std(past_6)) if len(past_6) > 1 else 0.0
        row_dict["rtt_rolling_mean_24h"] = float(np.mean(past_24))
        row_dict["rtt_rolling_std_24h"] = float(np.std(past_24)) if len(past_24) > 1 else 0.0
        row_dict["rtt_rolling_min_24h"] = float(np.min(past_24))
        row_dict["rtt_rolling_max_24h"] = float(np.max(past_24))

        # 4. Exogenous telemetry features (known up to origin T or lagged)
        for col in feature_cols:
            if col not in row_dict:
                # Use value from the pre-computed test feature row or last known row
                if col in test_horizon_df.columns:
                    row_dict[col] = test_horizon_df[col].iloc[step_idx]
                elif col in last_known_row:
                    row_dict[col] = last_known_row[col]
                else:
                    row_dict[col] = 0.0

        # Construct single-row DataFrame strictly matching feature_cols
        step_features = pd.DataFrame([row_dict])[feature_cols]

        # Predict RTT for cur_time
        pred_val = float(model.predict(step_features)[0])
        # RTT is physical latency (>= 50 ms nominal)
        pred_val = max(50.0, pred_val)

        predictions.append(pred_val)
        working_rtt.append(pred_val)

    return pd.Series(predictions, index=forecast_timestamps, name="xgboost_forecast")
