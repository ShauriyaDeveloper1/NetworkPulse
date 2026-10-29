import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_and_validate_dataset
from src.split import chronological_train_test_split
from src.evaluation import calculate_metrics


def test_load_and_validate():
    df, meta = load_and_validate_dataset("drone_communication_dataset.csv")
    assert df.shape[0] == 45289
    assert meta["duplicate_timestamps"] == 0
    assert meta["is_hourly_regular"] is True
    assert meta["target_stats"]["missing"] == 0
    assert meta["target_stats"]["negative_count"] == 0


def test_chronological_split():
    df, _ = load_and_validate_dataset("drone_communication_dataset.csv")
    train_s, test_s, meta = chronological_train_test_split(df, train_ratio=0.80)
    assert len(train_s) == int(len(df) * 0.8)
    assert len(test_s) == len(df) - len(train_s)
    assert train_s.index.max() < test_s.index.min()
    assert meta["temporal_leakage_detected"] is False


def test_calculate_metrics():
    y_true = np.array([50.0, 50.0, 100.0, 200.0])
    y_pred = np.array([50.0, 60.0, 100.0, 180.0])
    metrics = calculate_metrics(y_true, y_pred)
    assert np.isclose(metrics["MAE"], 7.5)
    assert metrics["RMSE"] > 0
    assert metrics["sMAPE"] > 0
    assert metrics["MAPE"] > 0
