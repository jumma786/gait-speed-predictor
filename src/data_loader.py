"""
data_loader.py
--------------
Loads raw GaitPhase CSV files (force, marker, oversteps) for all subjects
and speeds. Returns clean DataFrames with subject_id and speed columns attached.
"""

import os
import re
import pandas as pd
from pathlib import Path


# Column definitions from GaitPhase_Desc.pdf
FORCE_COLS = ["FP1_x", "FP2_x", "FP1_y", "FP2_y", "FP1_z", "FP2_z"]

MARKER_COLS = [
    "L_FCC_x", "L_FM1_x", "L_FM2_x", "L_FM5_x",
    "R_FCC_x", "R_FM1_x", "R_FM2_x", "R_FM5_x",
    "L_FCC_y", "L_FM1_y", "L_FM2_y", "L_FM5_y",
    "R_FCC_y", "R_FM1_y", "R_FM2_y", "R_FM5_y",
    "L_FCC_z", "L_FM1_z", "L_FM2_z", "L_FM5_z",
    "R_FCC_z", "R_FM1_z", "R_FM2_z", "R_FM5_z",
]

# All 12 speeds in the protocol
SPEEDS = [round(s * 0.1, 1) for s in range(6, 18)]  # 0.6 to 1.7


def parse_filename(filename: str):
    """
    Extract subject_id and speed from filename.
    Example: GP10_0.6_force.csv -> ('GP10', 0.6, 'force')
    """
    pattern = r"^(GP\d+)_([\d.]+)_(force|marker|oversteps)\.csv$"
    match = re.match(pattern, filename)
    if match:
        subject_id = match.group(1)
        speed = float(match.group(2))
        file_type = match.group(3)
        return subject_id, speed, file_type
    return None, None, None


def load_force(filepath: str, subject_id: str, speed: float) -> pd.DataFrame:
    """Load a single force CSV and attach metadata."""
    df = pd.read_csv(filepath)
    df.columns = FORCE_COLS
    df["subject_id"] = subject_id
    df["speed"] = speed
    df["file_type"] = "force"
    return df


def load_marker(filepath: str, subject_id: str, speed: float) -> pd.DataFrame:
    """Load a single marker CSV and attach metadata."""
    df = pd.read_csv(filepath)
    df.columns = MARKER_COLS
    df["subject_id"] = subject_id
    df["speed"] = speed
    df["file_type"] = "marker"
    return df


def load_oversteps(filepath: str, subject_id: str, speed: float) -> pd.DataFrame:
    """Load a single oversteps CSV and attach metadata."""
    df = pd.read_csv(filepath, header=None, names=["overstep_time"])
    df["subject_id"] = subject_id
    df["speed"] = speed
    df["file_type"] = "oversteps"
    return df


def load_all(data_dir: str, file_type: str = "force") -> pd.DataFrame:
    """
    Load all CSV files of a given type from the data directory.

    Parameters
    ----------
    data_dir : str
        Path to folder containing raw GaitPhase CSVs.
    file_type : str
        One of 'force', 'marker', 'oversteps'.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame for all subjects and speeds.
    """
    data_dir = Path(data_dir)
    frames = []
    skipped = []

    for filepath in sorted(data_dir.glob(f"*_{file_type}.csv")):
        subject_id, speed, ftype = parse_filename(filepath.name)

        if subject_id is None:
            skipped.append(filepath.name)
            continue

        try:
            if file_type == "force":
                df = load_force(filepath, subject_id, speed)
            elif file_type == "marker":
                df = load_marker(filepath, subject_id, speed)
            elif file_type == "oversteps":
                df = load_oversteps(filepath, subject_id, speed)
            else:
                raise ValueError(f"Unknown file_type: {file_type}")

            frames.append(df)

        except Exception as e:
            print(f"[WARNING] Failed to load {filepath.name}: {e}")
            skipped.append(filepath.name)

    if skipped:
        print(f"[INFO] Skipped {len(skipped)} files: {skipped}")

    if not frames:
        raise FileNotFoundError(
            f"No '{file_type}' files found in {data_dir}. "
            "Check the data directory path."
        )

    combined = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Loaded {len(frames)} '{file_type}' files | "
          f"Total rows: {len(combined):,} | "
          f"Subjects: {combined['subject_id'].nunique()} | "
          f"Speeds: {sorted(combined['speed'].unique())}")

    return combined


def load_subject(data_dir: str, subject_id: str,
                 file_type: str = "force") -> pd.DataFrame:
    """
    Load all speeds for a single subject.

    Parameters
    ----------
    data_dir : str
        Path to folder containing raw GaitPhase CSVs.
    subject_id : str
        Subject ID e.g. 'GP10'.
    file_type : str
        One of 'force', 'marker', 'oversteps'.
    """
    data_dir = Path(data_dir)
    frames = []

    for filepath in sorted(data_dir.glob(f"{subject_id}_*_{file_type}.csv")):
        _, speed, _ = parse_filename(filepath.name)
        if speed is None:
            continue
        if file_type == "force":
            frames.append(load_force(filepath, subject_id, speed))
        elif file_type == "marker":
            frames.append(load_marker(filepath, subject_id, speed))
        elif file_type == "oversteps":
            frames.append(load_oversteps(filepath, subject_id, speed))

    if not frames:
        raise FileNotFoundError(
            f"No '{file_type}' files found for subject '{subject_id}' in {data_dir}."
        )

    return pd.concat(frames, ignore_index=True)


def get_subject_list(data_dir: str) -> list:
    """Return sorted list of unique subject IDs found in data directory."""
    data_dir = Path(data_dir)
    subjects = set()
    for filepath in data_dir.glob("*_force.csv"):
        subject_id, _, _ = parse_filename(filepath.name)
        if subject_id:
            subjects.add(subject_id)
    return sorted(subjects)


if __name__ == "__main__":
    # Quick smoke test — update path before running
    DATA_DIR = r"C:\Users\jumma\Downloads\GaitPhase\data"

    print("=== Subject List ===")
    subjects = get_subject_list(DATA_DIR)
    print(subjects)

    print("\n=== Load All Force Files ===")
    force_df = load_all(DATA_DIR, file_type="force")
    print(force_df.head())
    print(force_df.dtypes)

    print("\n=== Load Single Subject (GP10) Force ===")
    gp10 = load_subject(DATA_DIR, "GP10", file_type="force")
    print(gp10.shape)
    print(gp10["speed"].value_counts().sort_index())