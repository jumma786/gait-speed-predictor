"""
feature_engineering.py
-----------------------
Derives higher-level biomechanical features from preprocessed GaitPhase data.
Input: features.csv (output of preprocessor.py)
Output: model_features.csv — final dataset ready for ML modelling.

New features added:
- Bilateral ratios (left/right)
- Step frequency proxy (1 / stance_duration)
- Force balance ratios (vertical vs horizontal)
- Speed-normalised impulse
- Combined asymmetry score
"""

import numpy as np
import pandas as pd
from pathlib import Path


def add_bilateral_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute left/right ratios for key force features.
    Ratio > 1 means left dominant, < 1 means right dominant.
    """
    feature_pairs = [
        ("impulse_ns", "impulse_ns_left", "impulse_ns_right"),
        ("peak_vgrf", "peak_vgrf_left", "peak_vgrf_right"),
        ("mean_vgrf", "mean_vgrf_left", "mean_vgrf_right"),
        ("loading_rate", "loading_rate_left", "loading_rate_right"),
        ("stance_duration_ms", "stance_duration_ms_left", "stance_duration_ms_right"),
        ("peak_ap_force", "peak_ap_force_left", "peak_ap_force_right"),
        ("peak_ml_force", "peak_ml_force_left", "peak_ml_force_right"),
    ]

    for name, left_col, right_col in feature_pairs:
        df[f"ratio_{name}"] = df[left_col] / df[right_col].replace(0, np.nan)

    return df


def add_step_frequency_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step frequency proxy = 1000 / stance_duration_ms (steps per second).
    Higher speed → shorter stance → higher frequency.
    """
    df["step_freq_left"] = 1000.0 / df["stance_duration_ms_left"].replace(0, np.nan)
    df["step_freq_right"] = 1000.0 / df["stance_duration_ms_right"].replace(0, np.nan)
    df["step_freq_mean"] = (df["step_freq_left"] + df["step_freq_right"]) / 2.0
    return df


def add_force_balance_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vertical vs horizontal force balance.
    - vgrf_to_ap: how much of force is vertical vs anterior-posterior
    - vgrf_to_ml: how much of force is vertical vs mediolateral
    Higher values = more upright, efficient gait.
    """
    for side in ["left", "right"]:
        vgrf = df[f"mean_vgrf_{side}"]
        ap = df[f"mean_ap_force_{side}"].replace(0, np.nan)
        ml = df[f"mean_ml_force_{side}"].replace(0, np.nan)

        df[f"vgrf_to_ap_{side}"] = vgrf / ap
        df[f"vgrf_to_ml_{side}"] = vgrf / ml

    return df


def add_impulse_per_stance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impulse normalised by stance duration — force efficiency metric.
    Higher = more force delivered per millisecond of contact.
    """
    for side in ["left", "right"]:
        impulse = df[f"impulse_ns_{side}"]
        stance = df[f"stance_duration_ms_{side}"].replace(0, np.nan)
        df[f"impulse_per_ms_{side}"] = impulse / stance

    return df


def add_combined_asymmetry_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single composite asymmetry score = mean of all asymmetry indices.
    Useful as one summary feature for injury/rehabilitation framing.
    """
    ai_cols = [c for c in df.columns if c.startswith("ai_")]
    df["asymmetry_score"] = df[ai_cols].mean(axis=1)
    return df


def add_mean_bilateral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average of left and right for key features.
    Gives single bilateral summary per step cycle.
    """
    pairs = [
        ("impulse_ns", "impulse_ns_left", "impulse_ns_right"),
        ("peak_vgrf", "peak_vgrf_left", "peak_vgrf_right"),
        ("mean_vgrf", "mean_vgrf_left", "mean_vgrf_right"),
        ("loading_rate", "loading_rate_left", "loading_rate_right"),
        ("stance_duration_ms", "stance_duration_ms_left", "stance_duration_ms_right"),
    ]
    for name, left_col, right_col in pairs:
        df[f"mean_{name}"] = (df[left_col] + df[right_col]) / 2.0

    return df


def engineer_features(input_path: str, save_path: str = None) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    input_path : str
        Path to features.csv from preprocessor.py
    save_path : str, optional
        If provided, saves final feature set as CSV.

    Returns
    -------
    pd.DataFrame
        Enriched feature dataset ready for modelling.
    """
    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded features: {df.shape}")

    df = add_bilateral_ratios(df)
    df = add_step_frequency_proxy(df)
    df = add_force_balance_ratios(df)
    df = add_impulse_per_stance(df)
    df = add_combined_asymmetry_score(df)
    df = add_mean_bilateral_features(df)

    # Drop any rows with NaN introduced by division
    original_len = len(df)
    df = df.dropna()
    if len(df) < original_len:
        print(f"[INFO] Dropped {original_len - len(df)} rows with NaN values")

    print(f"[INFO] Final feature dataset: {df.shape}")
    print(f"[INFO] Features added: {df.shape[1] - 25} new columns")
    print(f"[INFO] All columns:\n{list(df.columns)}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[INFO] Saved to {save_path}")

    return df


def get_model_features(df: pd.DataFrame) -> tuple:
    """
    Return X (features) and y (target) ready for sklearn.

    Target: speed (continuous — regression problem)
    Features: all engineered columns except identifiers and target.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    feature_names : list
    """
    exclude_cols = ["subject_id", "speed", "file_type"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df["speed"]

    print(f"[INFO] X shape: {X.shape} | y shape: {y.shape}")
    print(f"[INFO] Target (speed) distribution:\n{y.value_counts().sort_index()}")

    return X, y, feature_cols


if __name__ == "__main__":
    INPUT_PATH = r"C:\Users\jumma\Downloads\GaitPhase\data\processed\features.csv"
    SAVE_PATH = r"C:\Users\jumma\Downloads\GaitPhase\data\processed\model_features.csv"

    print("=== Feature Engineering Pipeline ===")
    df = engineer_features(INPUT_PATH, save_path=SAVE_PATH)

    print("\n=== Model Ready Features ===")
    X, y, feature_names = get_model_features(df)
    print(f"\nSample feature correlations with speed:")
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    print(correlations.head(10))
