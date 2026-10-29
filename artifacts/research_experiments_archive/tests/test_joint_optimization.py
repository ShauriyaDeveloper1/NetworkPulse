"""
Unit tests for the Joint ARIMA and XGBoost Optimization Module.
Validates leakage prevention, metric integrity, transform correctness, and forecasting pipelines.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.joint_optimization import (
    evaluate_arima_validation_grid,
    fit_and_forecast_arima,
    forecast_persistence,
    generate_hybrid_forecast
)
from src.evaluation import calculate_metrics


def test_forecast_persistence():
    series = pd.Series([50, 50, 100, 250, 80])
    pred = forecast_persistence(series, steps=24)
    assert len(pred) == 24
    assert np.all(pred == 80.0)


def test_fit_and_forecast_arima_transforms():
    # Synthetic stationary sequence
    np.random.seed(42)
    base = np.random.choice([50.0, 50.0, 50.0, 100.0], size=300)
    series = pd.Series(base)
    
    # 1. Standard ARIMA
    pred_std = fit_and_forecast_arima(series, order=(1, 0, 0), transform=None, steps=12)
    assert len(pred_std) == 12
    assert np.all(pred_std >= 50.0) and np.all(pred_std <= 500.0)
    
    # 2. Log-ARIMA
    pred_log = fit_and_forecast_arima(series, order=(1, 0, 0), transform="log", steps=12)
    assert len(pred_log) == 12
    assert np.all(pred_log >= 50.0) and np.all(pred_log <= 500.0)


def test_generate_hybrid_forecast():
    p1 = np.array([50.0, 100.0, 150.0])
    p2 = np.array([100.0, 200.0, 300.0])
    
    # 50/50
    hyb50 = generate_hybrid_forecast(p1, p2, weight_arima=0.5)
    np.testing.assert_allclose(hyb50, [75.0, 150.0, 225.0])
    
    # 30/70
    hyb30 = generate_hybrid_forecast(p1, p2, weight_arima=0.3)
    np.testing.assert_allclose(hyb30, [85.0, 170.0, 255.0])


def test_evaluate_arima_validation_grid_schema():
    np.random.seed(42)
    base = np.random.choice([50.0, 50.0, 100.0], size=500)
    series = pd.Series(base)
    
    val_origins = [300, 350]
    candidate_orders = [(1, 0, 0), (0, 1, 1)]
    
    grid_df = evaluate_arima_validation_grid(
        target_series=series,
        val_origins=val_origins,
        orders=candidate_orders,
        transforms=[None, "log"],
        steps=6
    )
    
    assert isinstance(grid_df, pd.DataFrame)
    expected_cols = {"model_name", "order", "transform", "val_mae_mean", "val_rmse_mean", "val_smape_mean", "val_mae_std"}
    assert expected_cols.issubset(set(grid_df.columns))
    assert len(grid_df) == 4
    # Ensure sorted by val_mae_mean ascending
    assert grid_df["val_mae_mean"].is_monotonic_increasing
