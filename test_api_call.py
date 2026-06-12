"""
Quick test of the /predict endpoint using median feature values.
Run from: C:/Users/jumma/Downloads/GaitPhase
"""

import requests

# Median feature values from model_features.csv (typical subject at ~1.15 m/s)
payload = {
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

print("=== Testing /predict endpoint ===")
r = requests.post("http://localhost:8000/predict", json=payload)
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")

print("\n=== Testing /health endpoint ===")
r2 = requests.get("http://localhost:8000/health")
print(f"Response: {r2.json()}")

print("\n=== Testing /features endpoint ===")
r3 = requests.get("http://localhost:8000/features")
data = r3.json()
print(f"Total features: {data['total_features']}")
print(f"Top SHAP features: {data['top_shap_features']}")
