"""
Unit tests for Dual-Objective Forecast Optimization (MAE, MSE/RMSE, Pareto).
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dual_objective_optimization import (
    generate_pareto_forecast,
    train_best_mae_model,
    train_best_rmse_model
)


def test_generate_pareto_forecast():
    p_mae = np.array([50.0, 60.0, 70.0])
    p_rmse = np.array([100.0, 120.0, 140.0])
    
    # 50/50
    p_mid = generate_pareto_forecast(p_mae, p_rmse, weight_mae=0.50)
    np.testing.assert_allclose(p_mid, [75.0, 90.0, 105.0])
    
    # 80/20
    p_80 = generate_pareto_forecast(p_mae, p_rmse, weight_mae=0.80)
    np.testing.assert_allclose(p_80, [60.0, 72.0, 84.0])


def test_train_dual_models_synthetic():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f_{i}" for i in range(5)])
    y = pd.Series(np.random.choice([50.0, 50.0, 50.0, 200.0], size=100))
    
    m_mae = train_best_mae_model(X, y, alpha=0.48)
    preds_mae = m_mae.predict(X)
    assert len(preds_mae) == 100
    assert not np.any(np.isnan(preds_mae))
    
    m_rmse = train_best_rmse_model(X, y, max_depth=3, learning_rate=0.03, n_estimators=50)
    preds_rmse = m_rmse.predict(X)
    assert len(preds_rmse) == 100
    assert not np.any(np.isnan(preds_rmse))
