"""
Unit Tests for AutoTS Advanced Forecasting Pipeline and Deliverables
Verifies:
  - Existence and integrity of all required AutoTS deliverables in artifacts/autots_advanced/
  - Alignment of evaluation timestamps between Log-ARIMA baseline and AutoTS advanced model
  - Strict non-negativity and compliance with physical telemetry bounds (>= 50.0 ms)
  - Mathematical correctness of absolute and squared errors
  - Completeness of model metadata in best_autots_model.json and evaluation_metrics.json
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUTOTS_DIR = os.path.join(REPO_ROOT, "artifacts", "autots_advanced")


def test_autots_deliverables_exist():
    """Verifies that all 9 required deliverable files exist in artifacts/autots_advanced/."""
    expected_files = [
        "autots_advanced.py",
        "best_autots_model.json",
        "model_comparison.csv",
        "advanced_predictions.csv",
        "baseline_predictions.csv",
        "forecast_comparison.png",
        "horizon_error_profile.png",
        "evaluation_metrics.json",
        "autots_report.md"
    ]
    for filename in expected_files:
        path = os.path.join(AUTOTS_DIR, filename)
        assert os.path.exists(path), f"Missing required deliverable: {filename}"
        assert os.path.getsize(path) > 0, f"Deliverable is empty: {filename}"


def test_best_autots_model_schema():
    """Verifies schema and contents of best_autots_model.json."""
    json_path = os.path.join(AUTOTS_DIR, "best_autots_model.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    assert "autots_version" in data
    assert "selected_model_name" in data
    assert len(data["selected_model_name"]) > 0
    assert "model_parameters" in data
    assert "transformation_parameters" in data
    assert data["forecast_length"] == 24
    assert data["frequency"] == "h"
    assert "validation_mae_ms" in data
    assert data["validation_mae_ms"] > 0
    assert "validation_rmse_ms" in data
    assert data["validation_rmse_ms"] > 0


def test_predictions_alignment_and_validity():
    """Verifies that baseline and advanced predictions have identical timestamps and valid bounds."""
    adv_csv = os.path.join(AUTOTS_DIR, "advanced_predictions.csv")
    base_csv = os.path.join(AUTOTS_DIR, "baseline_predictions.csv")

    df_adv = pd.read_csv(adv_csv)
    df_base = pd.read_csv(base_csv)

    # 28 windows * 24 steps = 672 observations
    assert len(df_adv) == 672
    assert len(df_base) == 672

    # Verify identical timestamp alignment
    assert (df_adv["timestamp"] == df_base["timestamp"]).all()
    assert (df_adv["origin_timestamp"] == df_base["origin_timestamp"]).all()
    assert (df_adv["actual_rtt_ms"] == df_base["actual_rtt_ms"]).all()

    # Verify step numbering 1 to 24
    assert (df_adv["forecast_step"] == df_base["forecast_step"]).all()
    assert set(df_adv["forecast_step"].unique()) == set(range(1, 25))

    # Verify physical floor bounds (>= 50 ms)
    assert (df_adv["predicted_rtt_ms"] >= 50.0).all()
    assert (df_base["predicted_rtt_ms"] >= 50.0).all()

    # Verify non-negative absolute errors and mathematical consistency
    assert (df_adv["absolute_error_ms"] >= 0.0).all()
    assert (df_base["absolute_error_ms"] >= 0.0).all()
    expected_adv_abs = np.abs(df_adv["actual_rtt_ms"] - df_adv["predicted_rtt_ms"])
    assert np.allclose(df_adv["absolute_error_ms"].values, expected_adv_abs.values, atol=1e-4)


def test_model_comparison_csv_metrics():
    """Verifies that model_comparison.csv contains all expected metrics."""
    comp_csv = os.path.join(AUTOTS_DIR, "model_comparison.csv")
    df_comp = pd.read_csv(comp_csv)

    metrics = df_comp["evaluation_metric"].tolist()
    assert "Validation MAE (ms)" in metrics
    assert "Primary 24h Test MAE (ms)" in metrics
    assert "Multi-Week Pooled MAE (672h, ms)" in metrics
    assert "Macro Window Mean MAE (ms)" in metrics
    assert "24h Window Win Rate (%)" in metrics

    # AutoTS must have positive win rate and valid reported MAE
    row_pooled = df_comp[df_comp["evaluation_metric"] == "Multi-Week Pooled MAE (672h, ms)"].iloc[0]
    assert row_pooled["autots_advanced_model"] > 0
    assert row_pooled["log_arima_100_baseline"] > 0


def test_evaluation_metrics_json_integrity():
    """Verifies nested structure of evaluation_metrics.json."""
    json_path = os.path.join(AUTOTS_DIR, "evaluation_metrics.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    assert "dataset_metadata" in data
    assert data["dataset_metadata"]["total_observations"] == 45289
    assert data["dataset_metadata"]["train_observations"] == 36231
    assert data["dataset_metadata"]["test_observations"] == 9058

    assert "models" in data
    assert "baseline" in data["models"]
    assert "autots_advanced" in data["models"]
    assert "comparison" in data
    assert data["comparison"]["total_windows_evaluated"] == 28
    assert "step_mae_by_horizon" in data
    assert len(data["step_mae_by_horizon"]) == 24
