"""
NetworkPulse: Test Suite for Isolation Forest Anomaly Detection
==============================================================
Validates feature engineering causality, model training, continuous anomaly scoring,
and artifact serialization/loading.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from src.anomaly_detection.feature_engineering import extract_anomaly_features
from src.anomaly_detection.isolation_forest_detector import IsolationForestDetector
import config


@pytest.fixture
def sample_telemetry_df():
    """Generates synthetic 200-hour telemetry data with known spikes."""
    np.random.seed(42)
    n = 200
    timestamps = pd.date_range("2024-01-01 00:00", periods=n, freq="h")
    
    # Baseline 50 ms with occasional spikes
    rtt = np.full(n, 50.0)
    spike_indices = [15, 30, 75, 120, 180]
    rtt[spike_indices] = [180.0, 320.0, 450.0, 210.0, 390.0]
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "round_trip_time": rtt,
        "packet_loss_rate": np.where(rtt > 50.0, 0.05, 0.0),
        "signal_noise_ratio": np.where(rtt > 50.0, 12.0, 30.0),
        "signal_strength": np.where(rtt > 50.0, -85.0, -60.0),
        "sequence_number_gap": np.where(rtt > 50.0, 5.0, 0.0),
        "network_traffic_volume": np.random.uniform(100.0, 2000.0, size=n),
        "base_station_load": np.random.uniform(20.0, 90.0, size=n)
    })
    return df


def test_anomaly_feature_engineering(sample_telemetry_df):
    """Verifies that causal features are extracted with correct dimensions and zero NaNs."""
    feat_df = extract_anomaly_features(sample_telemetry_df)
    
    assert len(feat_df) == len(sample_telemetry_df)
    for col in config.IF_CONFIG["features"]:
        assert col in feat_df.columns
    
    # Check that rolling mean and std are populated without lookahead NaNs
    assert not feat_df.isna().any().any()


def test_isolation_forest_detector_fit_and_predict(sample_telemetry_df):
    """Tests fitting IsolationForestDetector, scoring output bounds [0, 1], and labels."""
    detector = IsolationForestDetector(n_estimators=50, contamination=0.10, random_state=42)
    detector.fit(sample_telemetry_df)
    
    preds = detector.predict(sample_telemetry_df)
    
    assert "raw_decision_score" in preds
    assert "anomaly_score" in preds
    assert "is_anomaly" in preds
    assert "anomaly_label" in preds
    
    scores = preds["anomaly_score"].values
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
    
    labels = set(preds["anomaly_label"])
    assert labels.issubset({"Normal", "Anomaly"})
    assert np.sum(preds["is_anomaly"]) > 0


def test_isolation_forest_model_persistence(tmp_path, sample_telemetry_df):
    """Tests saving and loading Isolation Forest detector and scaler."""
    detector = IsolationForestDetector(n_estimators=30, contamination=0.08, random_state=42)
    detector.fit(sample_telemetry_df)
    preds1 = detector.predict(sample_telemetry_df)
    
    save_dir = str(tmp_path / "if_test_model")
    detector.save_model(save_dir)
    
    assert os.path.exists(os.path.join(save_dir, "isolation_forest.pkl"))
    assert os.path.exists(os.path.join(save_dir, "scaler.pkl"))
    assert os.path.exists(os.path.join(save_dir, "feature_config.json"))
    
    loaded_detector = IsolationForestDetector.load_model(save_dir)
    preds2 = loaded_detector.predict(sample_telemetry_df)
    
    np.testing.assert_allclose(preds1["anomaly_score"].values, preds2["anomaly_score"].values, rtol=1e-5)
    np.testing.assert_array_equal(preds1["is_anomaly"].values, preds2["is_anomaly"].values)


def test_isolation_forest_saved_artifacts_exist():
    """Validates that production Isolation Forest artifacts exist and are loadable."""
    assert os.path.exists(os.path.join(config.ISOLATION_FOREST_DIR, "isolation_forest.pkl"))
    assert os.path.exists(os.path.join(config.ISOLATION_FOREST_DIR, "scaler.pkl"))
    assert os.path.exists(os.path.join(config.ISOLATION_FOREST_DIR, "feature_config.json"))
    assert os.path.exists(config.ANOMALY_PREDICTIONS_PATH)
    
    # Verify loaded detector works
    detector = IsolationForestDetector.load_model(config.ISOLATION_FOREST_DIR)
    assert detector.model is not None
    assert detector.scaler is not None
    assert detector.is_fitted is True
