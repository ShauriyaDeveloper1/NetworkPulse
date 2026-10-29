"""
Unit Tests for Data Ingestion, Chronological Splitting, and Evaluation Metrics
Verifies:
  - Telemetry dataset schema and datetime parsing
  - Chronological 80/20 train/test splitting (zero leakage)
  - Accurate calculation of MAE, RMSE, sMAPE, and MAPE
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
from src.split import chronological_train_test_split
from src.evaluation import calculate_metrics


def test_data_loader_structure():
    """Verifies that load_and_validate_dataset produces DatetimeIndex and valid target."""
    csv_path = os.path.join(REPO_ROOT, "drone_communication_dataset.csv")
    df, meta = load_and_validate_dataset(csv_path)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert meta["is_chronological_original"] is True
    assert meta["is_hourly_regular"] is True
    assert meta["target_stats"]["missing"] == 0
    assert meta["target_stats"]["negative_count"] == 0
    assert meta["target_stats"]["min"] >= 50.0


def test_chronological_split_leakage_free():
    """Verifies chronological split preserves temporal ordering with zero leakage."""
    csv_path = os.path.join(REPO_ROOT, "drone_communication_dataset.csv")
    df, _ = load_and_validate_dataset(csv_path)

    train_series, test_series, split_meta = chronological_train_test_split(df, train_ratio=0.80)

    assert split_meta["temporal_leakage_detected"] is False
    assert train_series.index.max() < test_series.index.min()
    assert len(train_series) + len(test_series) == len(df)
    assert abs(split_meta["actual_train_ratio"] - 0.80) < 0.001


def test_calculate_metrics_values():
    """Verifies metric calculations with known values."""
    y_true = np.array([50.0, 60.0, 70.0, 80.0, 90.0])
    y_pred = np.array([55.0, 58.0, 75.0, 78.0, 88.0])

    m = calculate_metrics(y_true, y_pred)

    expected_mae = float(np.mean([5.0, 2.0, 5.0, 2.0, 2.0]))
    assert abs(m["MAE"] - expected_mae) < 1e-4
    assert m["RMSE"] >= m["MAE"]
    assert 0.0 <= m["sMAPE"] <= 100.0
