"""
schema.py
---------
Pydantic request/response models for the Gait Speed Predictor API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    """45 biomechanical features extracted from force plate data."""

    # Bilateral force features
    impulse_ns_left: float = Field(..., example=356.5)
    impulse_ns_right: float = Field(..., example=355.5)
    loading_rate_left: float = Field(..., example=1.90)
    loading_rate_right: float = Field(..., example=1.90)
    mean_ap_force_left: float = Field(..., example=54.8)
    mean_ap_force_right: float = Field(..., example=54.3)
    mean_ml_force_left: float = Field(..., example=27.0)
    mean_ml_force_right: float = Field(..., example=27.6)
    mean_vgrf_left: float = Field(..., example=476.4)
    mean_vgrf_right: float = Field(..., example=467.8)
    peak_ap_force_left: float = Field(..., example=121.7)
    peak_ap_force_right: float = Field(..., example=120.5)
    peak_ml_force_left: float = Field(..., example=51.7)
    peak_ml_force_right: float = Field(..., example=52.0)
    peak_vgrf_left: float = Field(..., example=734.2)
    peak_vgrf_right: float = Field(..., example=716.0)
    stance_duration_ms_left: float = Field(..., example=708.6)
    stance_duration_ms_right: float = Field(..., example=706.0)

    # Asymmetry indices
    ai_stance_duration_ms: float = Field(..., example=1.61)
    ai_peak_vgrf: float = Field(..., example=1.34)
    ai_mean_vgrf: float = Field(..., example=0.98)
    ai_impulse_ns: float = Field(..., example=1.98)
    ai_loading_rate: float = Field(..., example=2.62)

    # Bilateral ratios
    ratio_impulse_ns: float = Field(..., example=1.003)
    ratio_peak_vgrf: float = Field(..., example=1.025)
    ratio_mean_vgrf: float = Field(..., example=1.018)
    ratio_loading_rate: float = Field(..., example=1.002)
    ratio_stance_duration_ms: float = Field(..., example=1.004)
    ratio_peak_ap_force: float = Field(..., example=1.010)
    ratio_peak_ml_force: float = Field(..., example=0.994)

    # Step frequency proxies
    step_freq_left: float = Field(..., example=1.411)
    step_freq_right: float = Field(..., example=1.418)
    step_freq_mean: float = Field(..., example=1.414)

    # Force balance ratios
    vgrf_to_ap_left: float = Field(..., example=8.69)
    vgrf_to_ml_left: float = Field(..., example=17.65)
    vgrf_to_ap_right: float = Field(..., example=8.62)
    vgrf_to_ml_right: float = Field(..., example=16.94)

    # Impulse efficiency
    impulse_per_ms_left: float = Field(..., example=0.503)
    impulse_per_ms_right: float = Field(..., example=0.504)

    # Composite features
    asymmetry_score: float = Field(..., example=1.71)
    mean_impulse_ns: float = Field(..., example=356.0)
    mean_peak_vgrf: float = Field(..., example=725.1)
    mean_mean_vgrf: float = Field(..., example=472.1)
    mean_loading_rate: float = Field(..., example=1.90)
    mean_stance_duration_ms: float = Field(..., example=707.3)

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictResponse(BaseModel):
    predicted_speed_ms: float = Field(..., example=1.15)
    confidence_lower: float = Field(..., example=1.07)
    confidence_upper: float = Field(..., example=1.23)
    model: str = Field(..., example="Random Forest")
    note: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    mae_ms: float
    r2: float
