"""
Unit tests for the Stage 2 Final MAE Minimization Challenge.
Tests baseline definitions, macro vs micro error computation,
quantile boosting, and log-arima transforms.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.final_mae_challenge import (
    compute_macro_and_micro_metrics,
    generate_baseline_forecasts,
    fit_and_forecast_log_arima,
    train_quantile_xgboost
)


def test_compute_macro_and_micro_metrics():
    # 2 windows of 3 steps
    y_true = np.array([50.0, 50.0, 100.0, 200.0, 50.0, 50.0])
    y_pred = np.array([50.0, 50.0, 50.0, 100.0, 50.0, 50.0])
    
    # window 1 errors: [0, 0, 50] -> mae = 50/3 = 16.666
    # window 2 errors: [100, 0, 0] -> mae = 100/3 = 33.333
    # macro mae = (16.666 + 33.333)/2 = 25.0
    # micro mae = (0 + 0 + 50 + 100 + 0 + 0)/6 = 150/6 = 25.0
    res = compute_macro_and_micro_metrics(y_true, y_pred, window_length=3)
    assert np.isclose(res["macro_mae"], 25.0)
    assert np.isclose(res["micro_mae"], 25.0)
    assert len(res["window_maes"]) == 2
    assert np.isclose(res["window_maes"][0], 50.0 / 3)
    assert np.isclose(res["window_maes"][1], 100.0 / 3)


def test_generate_baseline_forecasts():
    history = pd.Series([50.0, 50.0, 60.0, 450.0])
    baselines = generate_baseline_forecasts(history, train_median=50.0, steps=10)
    
    # Persistence must be last observation (450.0)
    assert len(baselines["persistence"]) == 10
    assert np.all(baselines["persistence"] == 450.0)
    
    # Fixed median must be 50.0
    assert len(baselines["fixed_median"]) == 10
    assert np.all(baselines["fixed_median"] == 50.0)
    
    # Check distinct objects
    assert not np.array_equal(baselines["persistence"], baselines["fixed_median"])


def test_fit_and_forecast_log_arima():
    np.random.seed(42)
    s = pd.Series(np.random.choice([50.0, 50.0, 100.0], size=200))
    pred = fit_and_forecast_log_arima(s, order=(1, 0, 0), steps=12)
    assert len(pred) == 12
    assert not np.any(np.isnan(pred))
    assert np.all(pred >= 50.0) and np.all(pred <= 500.0)
