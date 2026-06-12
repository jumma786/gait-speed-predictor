"""
test_api.py
-----------
Pytest tests for the Gait Speed Predictor FastAPI service.
Run: pytest tests/test_api.py -v
"""

import pytest
import sys
import joblib
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.main import app
import api.main as main_module

# Load model explicitly for tests (TestClient doesn't trigger startup event)
MODEL_PATH = Path(__file__).parent.parent / "api" / "model" / "xgb_speed_model.pkl"
main_module.model = joblib.load(MODEL_PATH)

client = TestClient(app)

# Median feature values for a typical subject at ~1.15 m/s
SAMPLE_PAYLOAD = {
    "impulse_ns_left": 356.5,
    "impulse_ns_right": 355.5,
    "loading_rate_left": 1.90,
    "loading_rate_right": 1.90,
    "mean_ap_force_left": 54.8,
    "mean_ap_force_right": 54.3,
    "mean_ml_force_left": 27.0,
    "mean_ml_force_right": 27.6,
    "mean_vgrf_left": 476.4,
    "mean_vgrf_right": 467.8,
    "peak_ap_force_left": 121.7,
    "peak_ap_force_right": 120.5,
    "peak_ml_force_left": 51.7,
    "peak_ml_force_right": 52.0,
    "peak_vgrf_left": 734.2,
    "peak_vgrf_right": 716.0,
    "stance_duration_ms_left": 708.6,
    "stance_duration_ms_right": 706.0,
    "ai_stance_duration_ms": 1.61,
    "ai_peak_vgrf": 1.34,
    "ai_mean_vgrf": 0.98,
    "ai_impulse_ns": 1.98,
    "ai_loading_rate": 2.62,
    "ratio_impulse_ns": 1.003,
    "ratio_peak_vgrf": 1.025,
    "ratio_mean_vgrf": 1.018,
    "ratio_loading_rate": 1.002,
    "ratio_stance_duration_ms": 1.004,
    "ratio_peak_ap_force": 1.010,
    "ratio_peak_ml_force": 0.994,
    "step_freq_left": 1.411,
    "step_freq_right": 1.418,
    "step_freq_mean": 1.414,
    "vgrf_to_ap_left": 8.69,
    "vgrf_to_ml_left": 17.65,
    "vgrf_to_ap_right": 8.62,
    "vgrf_to_ml_right": 16.94,
    "impulse_per_ms_left": 0.503,
    "impulse_per_ms_right": 0.504,
    "asymmetry_score": 1.71,
    "mean_impulse_ns": 356.0,
    "mean_peak_vgrf": 725.1,
    "mean_mean_vgrf": 472.1,
    "mean_loading_rate": 1.90,
    "mean_stance_duration_ms": 707.3,
}


# ── Health endpoint ────────────────────────────────────────────────────────────

def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200


def test_health_model_loaded():
    r = client.get("/health")
    assert r.json()["model_loaded"] is True


def test_health_status_ok():
    r = client.get("/health")
    assert r.json()["status"] == "ok"


def test_health_has_metrics():
    r = client.get("/health")
    data = r.json()
    assert "mae_ms" in data
    assert "r2" in data
    assert data["r2"] > 0.9


# ── Predict endpoint ───────────────────────────────────────────────────────────

def test_predict_returns_200():
    r = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert r.status_code == 200


def test_predict_returns_speed():
    r = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = r.json()
    assert "predicted_speed_ms" in data


def test_predict_speed_in_valid_range():
    r = client.post("/predict", json=SAMPLE_PAYLOAD)
    speed = r.json()["predicted_speed_ms"]
    assert 0.4 <= speed <= 2.0, f"Speed {speed} out of valid range"


def test_predict_speed_near_median():
    """Median features should predict close to median speed (1.15 m/s)."""
    r = client.post("/predict", json=SAMPLE_PAYLOAD)
    speed = r.json()["predicted_speed_ms"]
    assert 0.8 <= speed <= 1.5, f"Expected ~1.15 m/s, got {speed}"


def test_predict_confidence_interval_present():
    r = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = r.json()
    assert "confidence_lower" in data
    assert "confidence_upper" in data


def test_predict_confidence_interval_valid():
    r = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = r.json()
    assert data["confidence_lower"] < data["predicted_speed_ms"]
    assert data["confidence_upper"] > data["predicted_speed_ms"]


def test_predict_missing_field_returns_422():
    incomplete = {k: v for k, v in SAMPLE_PAYLOAD.items()
                  if k != "step_freq_mean"}
    r = client.post("/predict", json=incomplete)
    assert r.status_code == 422


def test_predict_empty_payload_returns_422():
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_model_name_in_response():
    r = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert "model" in r.json()


# ── Features endpoint ──────────────────────────────────────────────────────────

def test_features_returns_200():
    r = client.get("/features")
    assert r.status_code == 200


def test_features_total_count():
    r = client.get("/features")
    assert r.json()["total_features"] == 45


def test_features_has_shap_list():
    r = client.get("/features")
    data = r.json()
    assert "top_shap_features" in data
    assert len(data["top_shap_features"]) > 0


def test_features_vgrf_to_ap_is_top_shap():
    """vgrf_to_ap_left should be top SHAP feature based on evaluation."""
    r = client.get("/features")
    top = r.json()["top_shap_features"]
    assert "vgrf_to_ap_left" in top
