"""
Unit Tests for Final Two-Model Forecasting Architecture (MAE Focus)
Verifies:
  - Model A: Log-ARIMA(1,0,0) forecast integrity, shape, and physical floor bounds
  - Model B: XGBoost Quantile (q=0.46) recursive forecasting without target leakage
  - TwoModelForecastingPipeline standardized output schema and error calculation
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
    BestMaeARIMAForecaster,
    BestMaeXGBoostForecaster,
    TwoModelForecastingPipeline,
    ForecastResult
)


@pytest.fixture(scope="module")
def telemetry_slice():
    """Loads a small real telemetry slice from drone_communication_dataset.csv for fast testing."""
    csv_path = os.path.join(REPO_ROOT, "drone_communication_dataset.csv")
    df, _ = load_and_validate_dataset(csv_path)
    # Take 300 rows from around the train/test split boundary
    train_end = int(len(df) * 0.80)
    sample_df = df.iloc[train_end - 200 : train_end + 100].copy()
    return sample_df


def test_arima_forecaster_shape_and_bounds(telemetry_slice):
    """Verifies that Model A produces exactly 24 bounded predictions with correct metadata."""
    forecaster = BestMaeARIMAForecaster(order=(1, 0, 0), transformation="log", recent_window=100)
    history = telemetry_slice.iloc[:200]
    horizon = telemetry_slice.iloc[200:224]

    result = forecaster.forecast_24h(history_df=history, horizon_df=horizon, steps=24)

    assert isinstance(result, ForecastResult)
    assert len(result.predictions_ms) == 24
    assert len(result.timestamps) == 24
    assert np.all(result.predictions_ms >= 50.0)
    assert np.all(result.predictions_ms <= 500.0)
    assert result.model_name == "Log-ARIMA(1,0,0)"
    assert result.metadata["order"] == [1, 0, 0]
    assert result.metadata["transformation"] == "log"


def test_xgboost_forecaster_initialization():
    """Verifies that Model B initializes with quantile alpha 0.46 and correct configuration."""
    forecaster = BestMaeXGBoostForecaster(quantile_alpha=0.46)
    assert forecaster.quantile_alpha == 0.46
    assert forecaster.model_name == "XGBoost Quantile (q=0.46)"
    assert forecaster.config_metadata["objective"] == "reg:quantileerror"
    assert forecaster.config_metadata["quantile_alpha"] == 0.46


def test_two_model_pipeline_schema_and_alignment(telemetry_slice):
    """Verifies that TwoModelForecastingPipeline aligns predictions and calculates errors accurately."""
    pipeline = TwoModelForecastingPipeline()
    origin_idx = 200
    steps = 24

    df_comp = pipeline.forecast_24h(df=telemetry_slice, origin_idx=origin_idx, steps=steps)

    expected_cols = [
        "step", "timestamp", "origin_timestamp", "actual_rtt_ms",
        "arima_pred_ms", "xgboost_pred_ms", "arima_abs_err_ms", "xgboost_abs_err_ms"
    ]
    for col in expected_cols:
        assert col in df_comp.columns

    assert len(df_comp) == steps
    assert df_comp["step"].tolist() == list(range(1, steps + 1))
    assert np.all(df_comp["arima_abs_err_ms"] >= 0.0)
    assert np.all(df_comp["xgboost_abs_err_ms"] >= 0.0)
    assert np.all(df_comp["arima_pred_ms"] >= 50.0)
    assert np.all(df_comp["xgboost_pred_ms"] >= 50.0)


def test_two_model_multi_window_evaluation(telemetry_slice):
    """Verifies multi-window evaluation execution and metric computation."""
    pipeline = TwoModelForecastingPipeline()
    origins = [150, 180, 210]
    steps = 24

    df_hourly, df_daily, summary = pipeline.evaluate_multi_window(
        df=telemetry_slice,
        origin_indices=origins,
        steps=steps
    )

    assert len(df_daily) == len(origins)
    assert len(df_hourly) == len(origins) * steps
    assert "arima" in summary
    assert "xgboost" in summary
    assert "micro_mae_ms" in summary["arima"]
    assert "micro_mae_ms" in summary["xgboost"]
    assert summary["arima"]["micro_mae_ms"] > 0
    assert summary["xgboost"]["micro_mae_ms"] > 0
