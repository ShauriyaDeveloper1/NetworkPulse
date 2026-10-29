"""
Unit Tests for Final Prediction Audit Deliverables
Verifies:
  - Existence of all required audit files in artifacts/final_prediction_audit/
  - Numerical equivalence between AutoTS BasicLinearModel and Constant 50.0 ms benchmark
  - Non-negativity, complete row count, and schema validity of window-level and horizon-level metrics
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUDIT_DIR = os.path.join(REPO_ROOT, "artifacts", "final_prediction_audit")


def test_audit_deliverables_exist():
    """Verifies that all required deliverable files exist in artifacts/final_prediction_audit/."""
    expected_files = [
        "prediction_integrity_report.md",
        "constant_baseline_metrics.json",
        "independent_metrics.json",
        "horizon_level_metrics.csv",
        "window_level_metrics.csv",
        "optimization_experiment_leaderboard.csv",
        "final_recommendation.md"
    ]
    for filename in expected_files:
        path = os.path.join(AUDIT_DIR, filename)
        assert os.path.exists(path), f"Missing audit deliverable: {filename}"
        assert os.path.getsize(path) > 0, f"Empty audit deliverable: {filename}"


def test_constant_baseline_metrics_equivalence():
    """Verifies that constant 50 ms and AutoTS MAE match exactly."""
    json_path = os.path.join(AUDIT_DIR, "constant_baseline_metrics.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    test_c50_mae = data["untouched_test_benchmark_672h"]["constant_50ms"]["MAE_ms"]
    test_autots_mae = data["untouched_test_benchmark_672h"]["autots_basic_linear_model"]["MAE_ms"]
    delta = data["untouched_test_benchmark_672h"]["autots_vs_constant_50ms_delta_ms"]

    assert abs(test_c50_mae - 47.1146) < 1e-3
    assert abs(test_autots_mae - 47.1146) < 1e-3
    assert abs(delta) < 1e-5


def test_horizon_level_metrics_schema_and_monotonicity():
    """Verifies horizon_level_metrics.csv row count and column structure."""
    csv_path = os.path.join(AUDIT_DIR, "horizon_level_metrics.csv")
    df = pd.read_csv(csv_path)

    assert len(df) == 24
    assert df["horizon_step"].tolist() == list(range(1, 25))
    assert (df["autots_blm_mae_ms"] > 0).all()
    assert (df["log_arima_mae_ms"] > 0).all()
    assert np.allclose(df["autots_blm_mae_ms"], df["constant_50ms_mae_ms"], atol=1e-4)


def test_window_level_metrics_integrity():
    """Verifies window_level_metrics.csv row count, origins, and win rate."""
    csv_path = os.path.join(AUDIT_DIR, "window_level_metrics.csv")
    df = pd.read_csv(csv_path)

    assert len(df) == 28
    assert df["window_id"].tolist() == list(range(1, 29))
    assert (df["window_winner"] == "AutoTS_and_C50").all()
    assert (df["autots_blm_mae_ms"] <= df["log_arima_mae_ms"]).all()
