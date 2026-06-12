"""
preprocessor.py
---------------
Segments continuous GRF signal into individual steps and extracts
per-step summaries ready for feature engineering.

Sampling rate: 1000 Hz (force data)
Foot contact threshold: 20N vertical force (FP1_z / FP2_z)
"""

import numpy as np
import pandas as pd
from pathlib import Path


# Standard biomechanics foot-contact threshold (Newtons)
FORCE_THRESHOLD = 20.0

# Minimum step duration in samples (1000 Hz → 100ms minimum)
MIN_STEP_SAMPLES = 100

# Maximum step duration in samples (1000 Hz → 2000ms maximum)
MAX_STEP_SAMPLES = 2000


def detect_foot_contacts(force_series: pd.Series,
                         threshold: float = FORCE_THRESHOLD) -> list:
    """
    Detect heel-strike and toe-off events from vertical force signal.

    Parameters
    ----------
    force_series : pd.Series
        Vertical ground reaction force (FP1_z or FP2_z).
    threshold : float
        Force threshold in Newtons to define foot contact.

    Returns
    -------
    list of tuples [(start_idx, end_idx), ...]
        Each tuple is one stance phase (foot on ground).
    """
    in_contact = force_series > threshold
    contacts = []
    start = None

    for i, contact in enumerate(in_contact):
        if contact and start is None:
            start = i  # heel strike
        elif not contact and start is not None:
            end = i  # toe off
            duration = end - start
            if MIN_STEP_SAMPLES <= duration <= MAX_STEP_SAMPLES:
                contacts.append((start, end))
            start = None

    # Close any open contact at end of signal
    if start is not None:
        end = len(in_contact)
        duration = end - start
        if MIN_STEP_SAMPLES <= duration <= MAX_STEP_SAMPLES:
            contacts.append((start, end))

    return contacts


def extract_step_features(force_df: pd.DataFrame,
                           subject_id: str,
                           speed: float) -> pd.DataFrame:
    """
    Extract per-step biomechanical features from force DataFrame.

    For each detected stance phase (left and right foot separately),
    computes summary statistics used as ML features.

    Parameters
    ----------
    force_df : pd.DataFrame
        Single subject + single speed force DataFrame (from data_loader).
    subject_id : str
        Subject identifier e.g. 'GP10'.
    speed : float
        Walking speed in m/s.

    Returns
    -------
    pd.DataFrame
        One row per step with extracted features.
    """
    records = []

    for foot, force_col in [("left", "FP1_z"), ("right", "FP2_z")]:
        contacts = detect_foot_contacts(force_df[force_col])

        for step_num, (start, end) in enumerate(contacts):
            segment = force_df.iloc[start:end]
            fz = segment[force_col].values

            # Mediolateral and anterior-posterior forces
            if foot == "left":
                fx = segment["FP1_x"].values
                fy = segment["FP1_y"].values
            else:
                fx = segment["FP2_x"].values
                fy = segment["FP2_y"].values

            # Duration
            stance_duration_ms = (end - start)  # samples at 1000 Hz = ms

            # Vertical force features
            peak_vgrf = fz.max()
            mean_vgrf = fz.mean()
            impulse = np.trapz(fz) / 1000.0  # N·s (divide by Hz)

            # Loading rate (steepest rise in first 50% of stance)
            half = len(fz) // 2
            loading_rate = (fz[:half].max() - fz[0]) / max(half, 1)

            # Mediolateral features
            peak_ml_force = np.abs(fy).max()
            mean_ml_force = np.abs(fy).mean()

            # Anterior-posterior features
            peak_ap_force = np.abs(fx).max()
            mean_ap_force = np.abs(fx).mean()

            # Symmetry placeholder (filled later in asymmetry analysis)
            records.append({
                "subject_id": subject_id,
                "speed": speed,
                "foot": foot,
                "step_num": step_num,
                "stance_duration_ms": stance_duration_ms,
                "peak_vgrf": peak_vgrf,
                "mean_vgrf": mean_vgrf,
                "impulse_ns": impulse,
                "loading_rate": loading_rate,
                "peak_ml_force": peak_ml_force,
                "mean_ml_force": mean_ml_force,
                "peak_ap_force": peak_ap_force,
                "mean_ap_force": mean_ap_force,
            })

    return pd.DataFrame(records)


def compute_asymmetry(step_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute left-right asymmetry index per subject per speed.

    Asymmetry Index (AI) = |Left - Right| / (0.5 * (Left + Right)) * 100

    Parameters
    ----------
    step_df : pd.DataFrame
        Output of extract_step_features, all subjects and speeds.

    Returns
    -------
    pd.DataFrame
        One row per subject+speed with asymmetry metrics.
    """
    feature_cols = [
        "stance_duration_ms", "peak_vgrf", "mean_vgrf",
        "impulse_ns", "loading_rate"
    ]

    records = []
    grouped = step_df.groupby(["subject_id", "speed", "foot"])

    for (subject_id, speed), group in step_df.groupby(["subject_id", "speed"]):
        left = group[group["foot"] == "left"][feature_cols].mean()
        right = group[group["foot"] == "right"][feature_cols].mean()

        row = {"subject_id": subject_id, "speed": speed}
        for col in feature_cols:
            denom = 0.5 * (left[col] + right[col])
            ai = abs(left[col] - right[col]) / denom * 100 if denom != 0 else 0
            row[f"ai_{col}"] = ai
            row[f"left_{col}"] = left[col]
            row[f"right_{col}"] = right[col]

        records.append(row)

    return pd.DataFrame(records)


def build_feature_dataset(data_dir: str,
                           save_path: str = None) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    Load all force files → segment steps → extract features → compute asymmetry.

    Parameters
    ----------
    data_dir : str
        Path to raw GaitPhase CSV folder.
    save_path : str, optional
        If provided, saves processed dataset as CSV.

    Returns
    -------
    pd.DataFrame
        Per-subject per-speed feature dataset ready for modelling.
    """
    from data_loader import load_all, get_subject_list

    data_dir = Path(data_dir)
    subjects = get_subject_list(str(data_dir))
    all_steps = []

    print(f"[INFO] Processing {len(subjects)} subjects...")

    for subject_id in subjects:
        from data_loader import load_subject
        force_df = load_subject(str(data_dir), subject_id, file_type="force")

        for speed, group in force_df.groupby("speed"):
            group = group.reset_index(drop=True)
            step_features = extract_step_features(group, subject_id, speed)
            all_steps.append(step_features)

    step_df = pd.concat(all_steps, ignore_index=True)
    print(f"[INFO] Total steps detected: {len(step_df):,}")

    # Aggregate to subject+speed level (mean per step)
    agg_cols = [
        "stance_duration_ms", "peak_vgrf", "mean_vgrf",
        "impulse_ns", "loading_rate", "peak_ml_force",
        "mean_ml_force", "peak_ap_force", "mean_ap_force"
    ]

    agg_df = (
        step_df.groupby(["subject_id", "speed", "foot"])[agg_cols]
        .mean()
        .reset_index()
    )

    # Pivot foot (left/right) into columns
    pivot_df = agg_df.pivot_table(
        index=["subject_id", "speed"],
        columns="foot",
        values=agg_cols
    )
    pivot_df.columns = [f"{col}_{foot}" for col, foot in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    # Add asymmetry indices
    asym_df = compute_asymmetry(step_df)
    asym_cols = [c for c in asym_df.columns if c.startswith("ai_")]
    final_df = pivot_df.merge(
        asym_df[["subject_id", "speed"] + asym_cols],
        on=["subject_id", "speed"],
        how="left"
    )

    print(f"[INFO] Feature dataset shape: {final_df.shape}")
    print(f"[INFO] Columns: {list(final_df.columns)}")

    if save_path:
        final_df.to_csv(save_path, index=False)
        print(f"[INFO] Saved to {save_path}")

    return final_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    DATA_DIR = r"C:\Users\jumma\Downloads\GaitPhase\data"
    SAVE_PATH = r"C:\Users\jumma\Downloads\GaitPhase\data\processed\features.csv"

    # Create processed folder if needed
    Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Quick test on single subject first
    print("=== Quick Test: GP10 at 0.6 m/s ===")
    from data_loader import load_subject
    test_df = load_subject(DATA_DIR, "GP10", file_type="force")
    test_speed = test_df[test_df["speed"] == 0.6].reset_index(drop=True)
    steps = extract_step_features(test_speed, "GP10", 0.6)
    print(f"Steps detected: {len(steps)}")
    print(steps.head())

    print("\n=== Left foot contacts at 0.6 m/s ===")
    contacts = detect_foot_contacts(test_speed["FP1_z"])
    print(f"Left foot stance phases: {len(contacts)}")
    print(f"First 3 contacts (start, end): {contacts[:3]}")

    print("\n=== Build Full Feature Dataset ===")
    features = build_feature_dataset(DATA_DIR, save_path=SAVE_PATH)
    print(features.head())
