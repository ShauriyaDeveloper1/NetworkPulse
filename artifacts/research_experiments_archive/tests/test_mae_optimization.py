import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_and_validate_dataset
from src.optimized_xgboost import (
    create_optimized_features,
    prepare_optimized_partitions,
    recursive_forecast_24h_optimized
)
import xgboost as xgb


def test_optimized_features_integrity():
    df, _ = load_and_validate_dataset("drone_communication_dataset.csv")
    sample_df = df.iloc[:120].copy()
    feat_df = create_optimized_features(sample_df, target_col="round_trip_time")

    assert "round_trip_time" in feat_df.columns
    assert "rtt_diff_1" in feat_df.columns
    assert "rtt_diff_24" in feat_df.columns
    assert "rtt_rolling_mean_48h" in feat_df.columns
    assert "rtt_ewma_24" in feat_df.columns
    assert "rtt_rolling_range_24h" in feat_df.columns
    assert "hour_sin" in feat_df.columns
    assert "label_normal" not in feat_df.columns
    assert "drone_identification" not in feat_df.columns


def test_optimized_partitions_leakage():
    df, _ = load_and_validate_dataset("drone_communication_dataset.csv")
    X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_optimized_partitions(
        df, train_ratio=0.80, val_ratio_within_train=0.15, target_col="round_trip_time"
    )

    assert X_train.index.max() < X_val.index.min()
    assert X_val.index.max() < X_test.index.min()
    assert len(features) == X_train.shape[1]
    assert len(X_test) == len(df) - int(len(df) * 0.80)


def test_optimized_forecast_shape():
    df, _ = load_and_validate_dataset("drone_communication_dataset.csv")
    X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_optimized_partitions(
        df.iloc[:500], train_ratio=0.80, val_ratio_within_train=0.20
    )
    model = xgb.XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=10,
        max_depth=2,
        random_state=42
    )
    model.fit(X_train, y_train)

    forecast = recursive_forecast_24h_optimized(
        model=model,
        history_df=df.iloc[:400],
        test_horizon_df=X_test.iloc[:24],
        feature_cols=features,
        target_col="round_trip_time",
        steps=24
    )
    assert len(forecast) == 24
    assert (forecast >= 50.0).all()
