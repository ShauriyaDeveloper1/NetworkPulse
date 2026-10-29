"""
NetworkPulse: Test Suite for Web Dashboard REST API
===================================================
Validates that all Flask API routes and frontend endpoints respond
with HTTP 200, return valid JSON structures, and adhere to expected schemas.
"""

import pytest
import json
from app import app


@pytest.fixture
def client():
    """Provides Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """GET / serves the dashboard HTML SPA."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"NetworkPulse" in response.data
    assert b"<!DOCTYPE html>" in response.data


def test_api_health(client):
    """GET /api/health returns health status and dataset metadata."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"
    assert "dataset_records" in data
    assert data["nominal_rtt_floor_ms"] == 50.0


def test_api_metrics_summary(client):
    """GET /api/metrics/summary returns KPI metrics cards data."""
    response = client.get("/api/metrics/summary")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "current_rtt_ms" in data
    assert "average_rtt_ms" in data
    assert "total_anomalies" in data
    assert "forecasted_max_rtt_ms" in data
    assert "health_status" in data


def test_api_telemetry_history(client):
    """GET /api/telemetry/history returns sliced time-series with rolling stats."""
    response = client.get("/api/telemetry/history?limit=48")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data["timestamps"]) == 48
    assert len(data["rtt_ms"]) == 48
    assert len(data["rolling_mean_6h"]) == 48
    assert "anomalies" in data


def test_api_forecast_windows(client):
    """GET /api/forecast/windows returns all 28 rolling evaluation horizons."""
    response = client.get("/api/forecast/windows")
    assert response.status_code == 200
    windows = json.loads(response.data)
    assert len(windows) == 28
    assert windows[0]["window_id"] == 1
    assert "mae_arima_ms" in windows[0]
    assert "mae_autots_ms" in windows[0]


def test_api_forecast_comparison(client):
    """GET /api/forecast/comparison returns actual vs 3 models for window N."""
    response = client.get("/api/forecast/comparison?window=1")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["window_id"] == 1
    assert len(data["actual_rtt_ms"]) == 24
    assert len(data["arima_pred_ms"]) == 24
    assert len(data["autots_pred_ms"]) == 24
    assert len(data["xgboost_pred_ms"]) == 24
    assert "metrics" in data
    assert "MAE" in data["metrics"]["autots"]


def test_api_forecast_horizon_profile(client):
    """GET /api/forecast/horizon_profile returns step-by-step h=1..24 error profile."""
    response = client.get("/api/forecast/horizon_profile")
    assert response.status_code == 200
    profile = json.loads(response.data)
    assert len(profile) == 24
    assert profile[0]["step"] == 1
    assert profile[23]["step"] == 24


def test_api_models_comparison(client):
    """GET /api/models/comparison returns structured 3-model comparison data."""
    response = client.get("/api/models/comparison")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data["models"]) == 3
    model_ids = [m["id"] for m in data["models"]]
    assert "log_arima_100" in model_ids
    assert "autots_basic_linear" in model_ids
    assert "xgboost_quantile" in model_ids


def test_api_anomalies_pagination(client):
    """GET /api/anomalies returns paginated anomaly events with severities."""
    response = client.get("/api/anomalies?page=1&page_size=10")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["page"] == 1
    assert data["page_size"] == 10
    assert len(data["records"]) == 10
    assert "observed_rtt_ms" in data["records"][0]
    assert "severity" in data["records"][0]


def test_api_data_explorer(client):
    """GET /api/data/explorer returns searchable, paginated raw dataset rows."""
    response = client.get("/api/data/explorer?page=1&page_size=10&anomalies_only=true")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data["records"]) == 10
    assert all(r["is_anomaly"] for r in data["records"])


def test_api_export_csv(client):
    """GET /api/export/csv streams valid CSV files."""
    for exp_type in ["anomalies", "forecast", "telemetry"]:
        response = client.get(f"/api/export/csv?type={exp_type}")
        assert response.status_code == 200
        assert "text/csv" in response.headers["Content-Type"]
        assert len(response.data) > 0


def test_api_reports_summary(client):
    """GET /api/reports/summary returns executive report data."""
    response = client.get("/api/reports/summary")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "title" in data
    assert "model_champion" in data
    assert "key_findings" in data
