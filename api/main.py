"""
main.py
-------
FastAPI service for walking speed prediction from ground reaction forces.

Endpoint: POST /predict
Input:    Ground reaction force features (45 biomechanical features)
Output:   Predicted walking speed in m/s

Run locally:
    uvicorn api.main:app --reload --port 8000

Test:
    http://localhost:8000/docs  (Swagger UI)
    http://localhost:8000/health
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schema import PredictRequest, PredictResponse, HealthResponse

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Gait Speed Predictor API",
    description=(
        "Predicts walking speed (m/s) from ground reaction force features. "
        "Model: Random Forest trained on GaitPhase dataset (Hebenstreit et al. 2014). "
        "Validation: Leave-One-Subject-Out CV | MAE: 0.057 m/s | R²: 0.958"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading ──────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model" / "xgb_speed_model.pkl"
model = None


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Model loaded from {MODEL_PATH}")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check API and model status."""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_type="XGBoost (Pipeline with StandardScaler)",
        mae_ms=0.058,
        r2=0.956,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Predict walking speed from ground reaction force features.

    Input: 45 biomechanical features extracted from force plate data.
    Output: Predicted walking speed in m/s with confidence band.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # Build feature dict from request
        features = request.dict()
        X = pd.DataFrame([features])

        # Predict
        predicted_speed = float(model.predict(X)[0])

        # Clip to valid protocol range
        predicted_speed = np.clip(predicted_speed, 0.4, 2.0)

        # Confidence band based on LOSO MAE ± 1 std
        lower = round(max(0.4, predicted_speed - 0.082), 3)
        upper = round(min(2.0, predicted_speed + 0.082), 3)

        return PredictResponse(
            predicted_speed_ms=round(predicted_speed, 4),
            confidence_lower=lower,
            confidence_upper=upper,
            model="Random Forest",
            note="Prediction based on force plate biomechanical features only."
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {str(e)}")


@app.get("/features", tags=["Info"])
def list_features():
    """Return list of required input features with descriptions."""
    return {
        "total_features": 45,
        "feature_groups": {
            "bilateral_force": [
                "impulse_ns_left", "impulse_ns_right",
                "peak_vgrf_left", "peak_vgrf_right",
                "mean_vgrf_left", "mean_vgrf_right",
                "loading_rate_left", "loading_rate_right",
                "peak_ap_force_left", "peak_ap_force_right",
                "mean_ap_force_left", "mean_ap_force_right",
                "peak_ml_force_left", "peak_ml_force_right",
                "mean_ml_force_left", "mean_ml_force_right",
                "stance_duration_ms_left", "stance_duration_ms_right",
            ],
            "asymmetry_indices": [
                "ai_stance_duration_ms", "ai_peak_vgrf",
                "ai_mean_vgrf", "ai_impulse_ns", "ai_loading_rate",
            ],
            "bilateral_ratios": [
                "ratio_impulse_ns", "ratio_peak_vgrf", "ratio_mean_vgrf",
                "ratio_loading_rate", "ratio_stance_duration_ms",
                "ratio_peak_ap_force", "ratio_peak_ml_force",
            ],
            "step_frequency": [
                "step_freq_left", "step_freq_right", "step_freq_mean",
            ],
            "force_balance": [
                "vgrf_to_ap_left", "vgrf_to_ml_left",
                "vgrf_to_ap_right", "vgrf_to_ml_right",
            ],
            "impulse_efficiency": [
                "impulse_per_ms_left", "impulse_per_ms_right",
            ],
            "composite": [
                "asymmetry_score",
                "mean_impulse_ns", "mean_peak_vgrf", "mean_mean_vgrf",
                "mean_loading_rate", "mean_stance_duration_ms",
            ],
        },
        "top_shap_features": [
            "vgrf_to_ap_left",
            "vgrf_to_ap_right",
            "stance_duration_ms_left",
            "step_freq_left",
            "step_freq_mean",
        ]
    }
