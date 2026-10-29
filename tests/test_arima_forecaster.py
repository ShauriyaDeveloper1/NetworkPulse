"""
Unit Tests for Model A: Best-MAE ARIMA [Log-ARIMA(1,0,0)]
Verifies:
  - Stationarity properties and differencing requirement (d=0)
  - Log-transformation and geometric median inversion (naive exp(y))
  - Horizon forecast shape (24 steps), non-negativity, and reasonable bounds
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data_loader import load_and_validate_dataset
from src.forecasting_engine import BestMaeARIMAForecaster, ForecastResult


@pytest.fixture(scope="module")
def sample_rtt_history():
    """Loads a small sample from the dataset."""
    csv_path = os.path.join(REPO_ROOT, "drone_communication_dataset.csv")
    df, _ = load_and_validate_dataset(csv_path)
    train_end = int(len(df) * 0.80)
    return df.iloc[train_end - 500 : train_end + 50].copy()


def test_series_stationarity_d0(sample_rtt_history):
    """Verifies that RTT series is strongly stationary in levels (d=0)."""
    train_slice = sample_rtt_history.iloc[:400]["round_trip_time"].dropna()
    adf_result = adfuller(train_slice)
    p_value = adf_result[1]
    # p-value must be << 0.05 confirming stationarity without differencing
    assert p_value < 0.05, f"Expected stationary series with p < 0.05, got {p_value}"


def test_arima_initialization_defaults():
    """Verifies that BestMaeARIMAForecaster initializes with verified config."""
    forecaster = BestMaeARIMAForecaster()
    assert forecaster.order == (1, 0, 0)
    assert forecaster.transformation == "log"
    assert forecaster.model_name == "Log-ARIMA(1,0,0)"
    assert forecaster.model_family == "ARIMA"
    assert forecaster.config_metadata["inversion_strategy"] == "naive_exp_targeting_conditional_geometric_median"


def test_arima_forecast_24h(sample_rtt_history):
    """Verifies 24-step forecast generation and value bounds."""
    forecaster = BestMaeARIMAForecaster(order=(1, 0, 0), transformation="log", recent_window=300)
    history = sample_rtt_history.iloc[:400]
    horizon = sample_rtt_history.iloc[400:424]

    result = forecaster.forecast_24h(history_df=history, horizon_df=horizon, steps=24)

    assert isinstance(result, ForecastResult)
    assert len(result.predictions_ms) == 24
    assert len(result.timestamps) == 24
    # Check physical floor bounds (RTT is at least 50 ms)
    assert np.all(result.predictions_ms >= 50.0)
    assert np.all(result.predictions_ms <= 500.0)
    assert "MAE" in result.metrics
    assert "RMSE" in result.metrics
    assert result.metrics["MAE"] > 0.0


def test_arima_standalone_script_import():
    """Verifies that the dedicated Model A driver script can be imported."""
    from models.arima.run_best_mae_arima import run_best_mae_arima
    assert callable(run_best_mae_arima)
