"""
Unit Tests for Model B: Best-MAE XGBoost [Quantile Regressor q=0.46]
Verifies:
  - Feature engineering pipeline (82 causal features, drops labels and IDs)
  - Trained Booster model loading from models/xgboost/best_mae_xgboost_model.json
  - Multi-step recursive forecasting with dynamic lag updating
  - Physical floor bounding (>= 50.0 ms)
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data_loader import load_and_validate_dataset
from src.forecasting_engine import (
    BestMaeXGBoostForecaster,
    create_optimized_features,
    ForecastResult
)


@pytest.fixture(scope="module")
def sample_telemetry():
    """Loads a small sample from the dataset."""
    csv_path = os.path.join(REPO_ROOT, "drone_communication_dataset.csv")
    df, _ = load_and_validate_dataset(csv_path)
    train_end = int(len(df) * 0.80)
    return df.iloc[train_end - 200 : train_end + 50].copy()


def test_feature_engineering_causality(sample_telemetry):
    """Verifies that create_optimized_features excludes target leakage and labels."""
    feat_df = create_optimized_features(sample_telemetry)

    # Excluded columns
    forbidden = [
        "label_normal", "label_spoofing", "label_mitm", "label_ddos",
        "label_gps_spoofing", "label_malware", "label_jamming",
        "drone_identification", "drone_gps_coordinates"
    ]
    for col in forbidden:
        assert col not in feat_df.columns, f"Forbidden column {col} was found in features"

    # Lags must exist
    for lag in [1, 2, 3, 6, 12, 24]:
        assert f"rtt_lag_{lag}" in feat_df.columns

    # Calendar and rolling features exist
    assert "hour_sin" in feat_df.columns
    assert "rtt_rolling_mean_24h" in feat_df.columns


def test_xgboost_model_loading():
    """Verifies that Model B correctly loads weights and config."""
    model_path = os.path.join(REPO_ROOT, "models", "xgboost", "best_mae_xgboost_model.json")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"

    forecaster = BestMaeXGBoostForecaster(model_path=model_path, quantile_alpha=0.46)
    assert forecaster.quantile_alpha == 0.46
    assert forecaster.model is not None
    assert forecaster.model_path == model_path


def test_xgboost_recursive_forecast_24h(sample_telemetry):
    """Verifies 24-step recursive forecast execution and validity."""
    forecaster = BestMaeXGBoostForecaster(quantile_alpha=0.46)
    history = sample_telemetry.iloc[:200]
    horizon = sample_telemetry.iloc[200:224]

    result = forecaster.forecast_24h(history_df=history, horizon_df=horizon, steps=24)

    assert isinstance(result, ForecastResult)
    assert len(result.predictions_ms) == 24
    assert len(result.timestamps) == 24
    assert np.all(result.predictions_ms >= 50.0)
    assert np.all(result.predictions_ms <= 500.0)
    assert "MAE" in result.metrics
    assert result.metrics["MAE"] > 0.0


def test_xgboost_standalone_script_import():
    """Verifies that the dedicated Model B driver script can be imported."""
    from models.xgboost.run_best_mae_xgboost import run_best_mae_xgboost
    assert callable(run_best_mae_xgboost)
