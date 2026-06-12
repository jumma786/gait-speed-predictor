"""
test_features.py
----------------
Pytest tests for feature engineering functions.
Run: pytest tests/test_features.py -v
"""

import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from feature_engineering import (
    add_bilateral_ratios,
    add_step_frequency_proxy,
    add_force_balance_ratios,
    add_impulse_per_stance,
    add_combined_asymmetry_score,
    add_mean_bilateral_features,
)


@pytest.fixture
def sample_df():
    """Minimal feature DataFrame for testing."""
    return pd.DataFrame({
        "subject_id": ["GP10", "GP10"],
        "speed": [0.6, 1.2],
        "impulse_ns_left": [636.0, 400.0],
        "impulse_ns_right": [592.0, 390.0],
        "peak_vgrf_left": [520.0, 600.0],
        "peak_vgrf_right": [515.0, 595.0],
        "mean_vgrf_left": [390.0, 460.0],
        "mean_vgrf_right": [385.0, 455.0],
        "loading_rate_left": [1.45, 2.10],
        "loading_rate_right": [1.50, 2.05],
        "stance_duration_ms_left": [950.0, 680.0],
        "stance_duration_ms_right": [940.0, 675.0],
        "peak_ap_force_left": [80.0, 120.0],
        "peak_ap_force_right": [78.0, 118.0],
        "peak_ml_force_left": [45.0, 55.0],
        "peak_ml_force_right": [44.0, 54.0],
        "mean_ap_force_left": [40.0, 60.0],
        "mean_ap_force_right": [38.0, 58.0],
        "mean_ml_force_left": [22.0, 28.0],
        "mean_ml_force_right": [21.0, 27.0],
        "ai_stance_duration_ms": [1.06, 0.74],
        "ai_peak_vgrf": [0.97, 0.84],
        "ai_mean_vgrf": [1.30, 1.10],
        "ai_impulse_ns": [7.03, 2.50],
        "ai_loading_rate": [3.39, 2.38],
    })


# ── Bilateral ratios ───────────────────────────────────────────────────────────

def test_bilateral_ratios_added(sample_df):
    df = add_bilateral_ratios(sample_df.copy())
    assert "ratio_impulse_ns" in df.columns
    assert "ratio_peak_vgrf" in df.columns
    assert "ratio_stance_duration_ms" in df.columns


def test_bilateral_ratio_calculation(sample_df):
    df = add_bilateral_ratios(sample_df.copy())
    expected = 636.0 / 592.0
    assert abs(df["ratio_impulse_ns"].iloc[0] - expected) < 1e-6


def test_bilateral_ratio_positive(sample_df):
    df = add_bilateral_ratios(sample_df.copy())
    assert (df["ratio_impulse_ns"] > 0).all()


# ── Step frequency ─────────────────────────────────────────────────────────────

def test_step_freq_columns_added(sample_df):
    df = add_step_frequency_proxy(sample_df.copy())
    assert "step_freq_left" in df.columns
    assert "step_freq_right" in df.columns
    assert "step_freq_mean" in df.columns


def test_step_freq_increases_with_speed(sample_df):
    df = add_step_frequency_proxy(sample_df.copy())
    # Higher speed (row 1) should have higher step frequency
    assert df["step_freq_mean"].iloc[1] > df["step_freq_mean"].iloc[0]


def test_step_freq_formula(sample_df):
    df = add_step_frequency_proxy(sample_df.copy())
    expected = 1000.0 / 950.0
    assert abs(df["step_freq_left"].iloc[0] - expected) < 1e-6


# ── Force balance ──────────────────────────────────────────────────────────────

def test_force_balance_columns_added(sample_df):
    df = add_force_balance_ratios(sample_df.copy())
    assert "vgrf_to_ap_left" in df.columns
    assert "vgrf_to_ml_left" in df.columns
    assert "vgrf_to_ap_right" in df.columns
    assert "vgrf_to_ml_right" in df.columns


def test_vgrf_to_ap_positive(sample_df):
    df = add_force_balance_ratios(sample_df.copy())
    assert (df["vgrf_to_ap_left"] > 0).all()


# ── Impulse per stance ─────────────────────────────────────────────────────────

def test_impulse_per_ms_added(sample_df):
    df = add_impulse_per_stance(sample_df.copy())
    assert "impulse_per_ms_left" in df.columns
    assert "impulse_per_ms_right" in df.columns


def test_impulse_per_ms_positive(sample_df):
    df = add_impulse_per_stance(sample_df.copy())
    assert (df["impulse_per_ms_left"] > 0).all()


# ── Asymmetry score ────────────────────────────────────────────────────────────

def test_asymmetry_score_added(sample_df):
    df = add_combined_asymmetry_score(sample_df.copy())
    assert "asymmetry_score" in df.columns


def test_asymmetry_score_is_mean_of_ai_cols(sample_df):
    df = add_combined_asymmetry_score(sample_df.copy())
    ai_cols = [c for c in df.columns if c.startswith("ai_")]
    expected = df[ai_cols].mean(axis=1).iloc[0]
    assert abs(df["asymmetry_score"].iloc[0] - expected) < 1e-6


def test_asymmetry_score_non_negative(sample_df):
    df = add_combined_asymmetry_score(sample_df.copy())
    assert (df["asymmetry_score"] >= 0).all()


# ── Mean bilateral features ────────────────────────────────────────────────────

def test_mean_bilateral_added(sample_df):
    df = add_mean_bilateral_features(sample_df.copy())
    assert "mean_impulse_ns" in df.columns
    assert "mean_peak_vgrf" in df.columns
    assert "mean_stance_duration_ms" in df.columns


def test_mean_bilateral_calculation(sample_df):
    df = add_mean_bilateral_features(sample_df.copy())
    expected = (636.0 + 592.0) / 2.0
    assert abs(df["mean_impulse_ns"].iloc[0] - expected) < 1e-6
