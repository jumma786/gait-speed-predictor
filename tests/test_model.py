"""
test_model.py
-------------
Pytest tests for model loading and prediction logic.
Run: pytest tests/test_model.py -v
"""

import pytest
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

MODEL_PATH = Path(__file__).parent.parent / "api" / "model" / "xgb_speed_model.pkl"

SAMPLE_FEATURES = {
    "impulse_ns_left": 356.5, "impulse_ns_right": 355.5,
    "loading_rate_left": 1.90, "loading_rate_right": 1.90,
    "mean_ap_force_left": 54.8, "mean_ap_force_right": 54.3,
    "mean_ml_force_left": 27.0, "mean_ml_force_right": 27.6,
    "mean_vgrf_left": 476.4, "mean_vgrf_right": 467.8,
    "peak_ap_force_left": 121.7, "peak_ap_force_right": 120.5,
    "peak_ml_force_left": 51.7, "peak_ml_force_right": 52.0,
    "peak_vgrf_left": 734.2, "peak_vgrf_right": 716.0,
    "stance_duration_ms_left": 708.6, "stance_duration_ms_right": 706.0,
    "ai_stance_duration_ms": 1.61, "ai_peak_vgrf": 1.34,
    "ai_mean_vgrf": 0.98, "ai_impulse_ns": 1.98, "ai_loading_rate": 2.62,
    "ratio_impulse_ns": 1.003, "ratio_peak_vgrf": 1.025,
    "ratio_mean_vgrf": 1.018, "ratio_loading_rate": 1.002,
    "ratio_stance_duration_ms": 1.004, "ratio_peak_ap_force": 1.010,
    "ratio_peak_ml_force": 0.994,
    "step_freq_left": 1.411, "step_freq_right": 1.418, "step_freq_mean": 1.414,
    "vgrf_to_ap_left": 8.69, "vgrf_to_ml_left": 17.65,
    "vgrf_to_ap_right": 8.62, "vgrf_to_ml_right": 16.94,
    "impulse_per_ms_left": 0.503, "impulse_per_ms_right": 0.504,
    "asymmetry_score": 1.71, "mean_impulse_ns": 356.0,
    "mean_peak_vgrf": 725.1, "mean_mean_vgrf": 472.1,
    "mean_loading_rate": 1.90, "mean_stance_duration_ms": 707.3,
}


@pytest.fixture
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture
def sample_X():
    return pd.DataFrame([SAMPLE_FEATURES])


# ── Model loading ──────────────────────────────────────────────────────────────

def test_model_file_exists():
    assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}"


def test_model_loads_without_error(model):
    assert model is not None


def test_model_has_predict_method(model):
    assert hasattr(model, "predict")


def test_model_is_pipeline(model):
    from sklearn.pipeline import Pipeline
    assert isinstance(model, Pipeline)


# ── Prediction ─────────────────────────────────────────────────────────────────

def test_predict_returns_single_value(model, sample_X):
    result = model.predict(sample_X)
    assert len(result) == 1


def test_predict_returns_float(model, sample_X):
    result = model.predict(sample_X)
    assert isinstance(float(result[0]), float)


def test_predict_in_valid_range(model, sample_X):
    result = float(model.predict(sample_X)[0])
    assert 0.4 <= result <= 2.0


def test_predict_median_features_near_median_speed(model, sample_X):
    result = float(model.predict(sample_X)[0])
    assert 0.8 <= result <= 1.5


def test_predict_batch(model):
    """Model should handle multiple rows."""
    X = pd.DataFrame([SAMPLE_FEATURES] * 5)
    result = model.predict(X)
    assert len(result) == 5


def test_predict_consistent(model, sample_X):
    """Same input should always return same output."""
    r1 = float(model.predict(sample_X)[0])
    r2 = float(model.predict(sample_X)[0])
    assert r1 == r2
