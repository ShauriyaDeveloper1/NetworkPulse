import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_and_validate_dataset
from src.xgboost_forecasting import (
    create_time_series_features,
    prepare_train_val_test_data,
    recursive_forecast_24h
)
import xgboost as xgb


def test_feature_engineering_integrity():
    df, _ = load_and_validate_dataset("drone_communication_dataset.csv")
    sample_df = df.iloc[:100].copy()
    feat_df = create_time_series_features(sample_df, target_col="round_trip_time")
    
    # Target column must remain
    assert "round_trip_time" in feat_df.columns
    # Critical lag features
    assert "rtt_lag_1" in feat_df.columns
    assert "rtt_lag_24" in feat_df.columns
    assert "rtt_rolling_mean_6h" in feat_df.columns
    assert "rtt_rolling_std_24h" in feat_df.columns
    # Calendar features
    assert "hour_sin" in feat_df.columns
    assert "dayofweek_cos" in feat_df.columns
    # Excluded diagnostic labels and identifiers
    assert "label_normal" not in feat_df.columns
    assert "label_spoofing" not in feat_df.columns
    assert "drone_identification" not in feat_df.columns
    assert "drone_gps_coordinates" not in feat_df.columns


def test_prepare_train_val_test_split():
    df, _ = load_and_validate_dataset("drone_communication_dataset.csv")
    X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_train_val_test_data(
        df, train_ratio=0.80, val_ratio_within_train=0.15
    )
    
    # Chronological checks
    assert X_train.index.max() < X_val.index.min()
    assert X_val.index.max() < X_test.index.min()
    assert len(features) == X_train.shape[1]
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert len(X_test) == len(df) - int(len(df) * 0.80)


def test_recursive_forecast_shape():
    df, _ = load_and_validate_dataset("drone_communication_dataset.csv")
    X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_train_val_test_data(
        df.iloc[:500], train_ratio=0.80, val_ratio_within_train=0.20
    )
    # Fit tiny model for test
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X_train, y_train)
    
    forecast = recursive_forecast_24h(
        model=model,
        history_df=df.iloc[:400],
        test_horizon_df=X_test.iloc[:24],
        feature_cols=features,
        target_col="round_trip_time",
        steps=24
    )
    assert len(forecast) == 24
    assert (forecast >= 50.0).all()
