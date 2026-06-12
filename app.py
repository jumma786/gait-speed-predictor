"""
app.py
------
Streamlit demo app for the Gait Speed Predictor.
Calls the FastAPI /predict endpoint and displays results visually.

Run locally:
    streamlit run app.py
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gait Speed Predictor",
    page_icon="🦶",
    layout="wide",
)

# ── API URL ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🦶 Gait Speed Predictor")
st.markdown(
    "Predicts walking speed (m/s) from ground reaction force data. "
    "**Random Forest | MAE: 0.057 m/s | R²: 0.958 | LOSO Cross-Validation**"
)
st.divider()

# ── Sidebar: feature inputs ────────────────────────────────────────────────────
st.sidebar.header("⚙️ Force Plate Inputs")
st.sidebar.markdown("Adjust biomechanical features to simulate different gait patterns.")

st.sidebar.subheader("Stance Duration (ms)")
stance_left = st.sidebar.slider("Left foot stance duration", 500, 1200, 708)
stance_right = st.sidebar.slider("Right foot stance duration", 500, 1200, 706)

st.sidebar.subheader("Vertical GRF (N)")
peak_vgrf_left = st.sidebar.slider("Peak VGRF — Left", 400, 1200, 734)
peak_vgrf_right = st.sidebar.slider("Peak VGRF — Right", 400, 1200, 716)
mean_vgrf_left = st.sidebar.slider("Mean VGRF — Left", 300, 800, 476)
mean_vgrf_right = st.sidebar.slider("Mean VGRF — Right", 300, 800, 468)

st.sidebar.subheader("Anterior-Posterior Force (N)")
peak_ap_left = st.sidebar.slider("Peak AP Force — Left", 30, 250, 122)
peak_ap_right = st.sidebar.slider("Peak AP Force — Right", 30, 250, 120)
mean_ap_left = st.sidebar.slider("Mean AP Force — Left", 15, 120, 55)
mean_ap_right = st.sidebar.slider("Mean AP Force — Right", 15, 120, 54)

st.sidebar.subheader("Mediolateral Force (N)")
peak_ml_left = st.sidebar.slider("Peak ML Force — Left", 20, 110, 52)
peak_ml_right = st.sidebar.slider("Peak ML Force — Right", 20, 110, 52)
mean_ml_left = st.sidebar.slider("Mean ML Force — Left", 10, 50, 27)
mean_ml_right = st.sidebar.slider("Mean ML Force — Right", 10, 50, 28)

st.sidebar.subheader("Loading Rate")
loading_left = st.sidebar.slider("Loading Rate — Left", 0.8, 3.5, 1.9)
loading_right = st.sidebar.slider("Loading Rate — Right", 0.8, 3.5, 1.9)

st.sidebar.subheader("Impulse (N·s)")
impulse_left = st.sidebar.slider("Impulse — Left", 200, 700, 356)
impulse_right = st.sidebar.slider("Impulse — Right", 200, 700, 355)


# ── Derived features ───────────────────────────────────────────────────────────
def compute_features(
    stance_left, stance_right,
    peak_vgrf_left, peak_vgrf_right,
    mean_vgrf_left, mean_vgrf_right,
    peak_ap_left, peak_ap_right,
    mean_ap_left, mean_ap_right,
    peak_ml_left, peak_ml_right,
    mean_ml_left, mean_ml_right,
    loading_left, loading_right,
    impulse_left, impulse_right,
):
    step_freq_left = 1000.0 / stance_left
    step_freq_right = 1000.0 / stance_right
    step_freq_mean = (step_freq_left + step_freq_right) / 2

    vgrf_to_ap_left = mean_vgrf_left / mean_ap_left if mean_ap_left else 0
    vgrf_to_ml_left = mean_vgrf_left / mean_ml_left if mean_ml_left else 0
    vgrf_to_ap_right = mean_vgrf_right / mean_ap_right if mean_ap_right else 0
    vgrf_to_ml_right = mean_vgrf_right / mean_ml_right if mean_ml_right else 0

    impulse_per_ms_left = impulse_left / stance_left
    impulse_per_ms_right = impulse_right / stance_right

    def ai(a, b): return abs(a - b) / (0.5 * (a + b)) * 100 if (a + b) else 0

    ai_stance = ai(stance_left, stance_right)
    ai_peak_vgrf = ai(peak_vgrf_left, peak_vgrf_right)
    ai_mean_vgrf = ai(mean_vgrf_left, mean_vgrf_right)
    ai_impulse = ai(impulse_left, impulse_right)
    ai_loading = ai(loading_left, loading_right)
    asymmetry_score = np.mean([ai_stance, ai_peak_vgrf, ai_mean_vgrf, ai_impulse, ai_loading])

    return {
        "impulse_ns_left": float(impulse_left),
        "impulse_ns_right": float(impulse_right),
        "loading_rate_left": float(loading_left),
        "loading_rate_right": float(loading_right),
        "mean_ap_force_left": float(mean_ap_left),
        "mean_ap_force_right": float(mean_ap_right),
        "mean_ml_force_left": float(mean_ml_left),
        "mean_ml_force_right": float(mean_ml_right),
        "mean_vgrf_left": float(mean_vgrf_left),
        "mean_vgrf_right": float(mean_vgrf_right),
        "peak_ap_force_left": float(peak_ap_left),
        "peak_ap_force_right": float(peak_ap_right),
        "peak_ml_force_left": float(peak_ml_left),
        "peak_ml_force_right": float(peak_ml_right),
        "peak_vgrf_left": float(peak_vgrf_left),
        "peak_vgrf_right": float(peak_vgrf_right),
        "stance_duration_ms_left": float(stance_left),
        "stance_duration_ms_right": float(stance_right),
        "ai_stance_duration_ms": ai_stance,
        "ai_peak_vgrf": ai_peak_vgrf,
        "ai_mean_vgrf": ai_mean_vgrf,
        "ai_impulse_ns": ai_impulse,
        "ai_loading_rate": ai_loading,
        "ratio_impulse_ns": impulse_left / impulse_right if impulse_right else 1,
        "ratio_peak_vgrf": peak_vgrf_left / peak_vgrf_right if peak_vgrf_right else 1,
        "ratio_mean_vgrf": mean_vgrf_left / mean_vgrf_right if mean_vgrf_right else 1,
        "ratio_loading_rate": loading_left / loading_right if loading_right else 1,
        "ratio_stance_duration_ms": stance_left / stance_right if stance_right else 1,
        "ratio_peak_ap_force": peak_ap_left / peak_ap_right if peak_ap_right else 1,
        "ratio_peak_ml_force": peak_ml_left / peak_ml_right if peak_ml_right else 1,
        "step_freq_left": step_freq_left,
        "step_freq_right": step_freq_right,
        "step_freq_mean": step_freq_mean,
        "vgrf_to_ap_left": vgrf_to_ap_left,
        "vgrf_to_ml_left": vgrf_to_ml_left,
        "vgrf_to_ap_right": vgrf_to_ap_right,
        "vgrf_to_ml_right": vgrf_to_ml_right,
        "impulse_per_ms_left": impulse_per_ms_left,
        "impulse_per_ms_right": impulse_per_ms_right,
        "asymmetry_score": asymmetry_score,
        "mean_impulse_ns": (impulse_left + impulse_right) / 2,
        "mean_peak_vgrf": (peak_vgrf_left + peak_vgrf_right) / 2,
        "mean_mean_vgrf": (mean_vgrf_left + mean_vgrf_right) / 2,
        "mean_loading_rate": (loading_left + loading_right) / 2,
        "mean_stance_duration_ms": (stance_left + stance_right) / 2,
    }


# ── Predict button ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    predict_btn = st.button("🔍 Predict Walking Speed", type="primary", use_container_width=True)

with col2:
    reset_btn = st.button("↺ Reset to Median", use_container_width=True)

st.divider()

# ── Prediction ─────────────────────────────────────────────────────────────────
if predict_btn:
    features = compute_features(
        stance_left, stance_right,
        peak_vgrf_left, peak_vgrf_right,
        mean_vgrf_left, mean_vgrf_right,
        peak_ap_left, peak_ap_right,
        mean_ap_left, mean_ap_right,
        peak_ml_left, peak_ml_right,
        mean_ml_left, mean_ml_right,
        loading_left, loading_right,
        impulse_left, impulse_right,
    )

    try:
        response = requests.post(f"{API_URL}/predict", json=features, timeout=10)
        result = response.json()

        speed = result["predicted_speed_ms"]
        lower = result["confidence_lower"]
        upper = result["confidence_upper"]

        # ── Speed gauge ────────────────────────────────────────────────────────
        col_gauge, col_metrics = st.columns([1, 1])

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=speed,
                delta={"reference": 1.15, "valueformat": ".3f"},
                title={"text": "Predicted Walking Speed (m/s)", "font": {"size": 18}},
                number={"suffix": " m/s", "valueformat": ".3f"},
                gauge={
                    "axis": {"range": [0.4, 2.0], "tickwidth": 1},
                    "bar": {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0.4, 0.8], "color": "#ff7f7f"},
                        {"range": [0.8, 1.2], "color": "#ffdd57"},
                        {"range": [1.2, 1.7], "color": "#90ee90"},
                        {"range": [1.7, 2.0], "color": "#4CAF50"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 1.15
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_metrics:
            st.markdown("### Prediction Results")
            st.metric("Predicted Speed", f"{speed:.3f} m/s")
            st.metric("Confidence Band", f"{lower:.3f} – {upper:.3f} m/s")
            st.metric("Step Frequency", f"{features['step_freq_mean']:.3f} steps/s")
            st.metric("Asymmetry Score", f"{features['asymmetry_score']:.2f}%")

            if speed < 0.8:
                st.warning("⚠️ Very slow — potential mobility concern")
            elif speed < 1.0:
                st.info("🚶 Slow walking pace")
            elif speed < 1.3:
                st.success("✅ Normal walking pace")
            else:
                st.success("🏃 Fast walking pace")

        st.divider()

        # ── Key feature summary ────────────────────────────────────────────────
        st.markdown("### Key Biomechanical Features")

        shap_features = {
            "vgrf_to_ap_left": features["vgrf_to_ap_left"],
            "vgrf_to_ap_right": features["vgrf_to_ap_right"],
            "stance_duration_ms_left": features["stance_duration_ms_left"],
            "step_freq_left": features["step_freq_left"],
            "step_freq_mean": features["step_freq_mean"],
        }

        fig2 = px.bar(
            x=list(shap_features.values()),
            y=list(shap_features.keys()),
            orientation="h",
            title="Top SHAP Features — Current Input Values",
            labels={"x": "Feature Value", "y": "Feature"},
            color=list(shap_features.values()),
            color_continuous_scale="Blues",
        )
        fig2.update_layout(height=300, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure uvicorn is running on port 8000.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("👈 Adjust the sliders in the sidebar and click **Predict Walking Speed**")

# ── Model info footer ──────────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ About this model"):
    st.markdown("""
    **Model:** Random Forest (200 trees, max_depth=6)  
    **Validation:** Leave-One-Subject-Out CV (21 folds)  
    **MAE:** 0.057 m/s | **R²:** 0.958  
    **Dataset:** GaitPhase Database — Hebenstreit et al. (2014), FAU Erlangen-Nürnberg  
    **Top predictive feature:** Vertical-to-anterior force ratio (vgrf_to_ap_left)  
    
    > *Contrary to expectation, step frequency alone does not drive predictions —  
    the balance between vertical loading and forward propulsion is the strongest signal.*
    
    [GitHub](https://github.com/jumma786/gait-speed-predictor) | [API Docs](http://localhost:8000/docs)
    """)
