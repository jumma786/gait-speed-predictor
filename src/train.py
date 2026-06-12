"""
train.py
--------
Trains and compares regression models to predict walking speed
from ground reaction force features.

Models:
    1. DummyRegressor (baseline)
    2. Linear Regression
    3. Ridge Regression
    4. Random Forest
    5. XGBoost

Evaluation: MAE, RMSE, R²
Cross-validation: Leave-One-Subject-Out (LOSO) — gold standard for gait data.
Final model: XGBoost saved to api/model/xgb_speed_model.pkl
"""

import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Paths
DATA_PATH = r"C:\Users\jumma\Downloads\GaitPhase\data\processed\model_features.csv"
MODEL_SAVE_PATH = r"C:\Users\jumma\Downloads\GaitPhase\api\model\xgb_speed_model.pkl"
RESULTS_SAVE_PATH = r"C:\Users\jumma\Downloads\GaitPhase\data\processed\model_results.csv"


def load_data(path: str) -> tuple:
    """Load model_features.csv and return X, y, groups."""
    df = pd.read_csv(path)

    exclude_cols = ["subject_id", "speed"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df["speed"]
    groups = df["subject_id"]  # for LOSO cross-validation

    print(f"[INFO] Dataset: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"[INFO] Target range: {y.min()} – {y.max()} m/s")
    print(f"[INFO] Subjects: {groups.nunique()}")

    return X, y, groups, feature_cols


def get_models() -> dict:
    """Define all models to compare."""
    return {
        "Baseline (Dummy)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DummyRegressor(strategy="mean"))
        ]),
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            ))
        ]),
    }


def evaluate_loso(models: dict, X: pd.DataFrame,
                  y: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """
    Leave-One-Subject-Out cross-validation for all models.
    LOSO is the correct validation strategy for gait data:
    ensures model generalises to unseen subjects.
    """
    logo = LeaveOneGroupOut()
    results = []

    print("\n=== Leave-One-Subject-Out Cross-Validation ===\n")

    for name, pipeline in models.items():
        maes, rmses, r2s = [], [], []

        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            maes.append(mean_absolute_error(y_test, y_pred))
            rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2s.append(r2_score(y_test, y_pred))

        result = {
            "Model": name,
            "MAE_mean": np.mean(maes),
            "MAE_std": np.std(maes),
            "RMSE_mean": np.mean(rmses),
            "RMSE_std": np.std(rmses),
            "R2_mean": np.mean(r2s),
            "R2_std": np.std(r2s),
        }
        results.append(result)

        print(f"{name}")
        print(f"  MAE:  {result['MAE_mean']:.4f} ± {result['MAE_std']:.4f} m/s")
        print(f"  RMSE: {result['RMSE_mean']:.4f} ± {result['RMSE_std']:.4f} m/s")
        print(f"  R²:   {result['R2_mean']:.4f} ± {result['R2_std']:.4f}")
        print()

    return pd.DataFrame(results)


def train_final_model(X: pd.DataFrame, y: pd.Series,
                      save_path: str) -> Pipeline:
    """
    Train final XGBoost model on full dataset and save.
    Used for API serving.
    """
    print("=== Training Final XGBoost Model (Full Dataset) ===")

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        ))
    ])

    final_model.fit(X, y)

    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, save_path)
    print(f"[INFO] Model saved to {save_path}")

    # Quick sanity check
    y_pred = final_model.predict(X)
    train_mae = mean_absolute_error(y, y_pred)
    train_r2 = r2_score(y, y_pred)
    print(f"[INFO] Training MAE: {train_mae:.4f} m/s | R²: {train_r2:.4f}")

    return final_model


def print_summary(results_df: pd.DataFrame) -> None:
    """Print ranked model comparison table."""
    print("\n=== Model Comparison (ranked by MAE) ===\n")
    ranked = results_df.sort_values("MAE_mean")
    print(ranked[["Model", "MAE_mean", "RMSE_mean", "R2_mean"]].to_string(index=False))

    best = ranked.iloc[0]
    baseline = results_df[results_df["Model"] == "Baseline (Dummy)"].iloc[0]
    mae_reduction = (baseline["MAE_mean"] - best["MAE_mean"]) / baseline["MAE_mean"] * 100

    print(f"\n✅ Best model: {best['Model']}")
    print(f"✅ MAE: {best['MAE_mean']:.4f} m/s (±{best['MAE_std']:.4f})")
    print(f"✅ R²: {best['R2_mean']:.4f}")
    print(f"✅ MAE reduction vs baseline: {mae_reduction:.1f}%")
    print(f"✅ Interpretation: Predicts walking speed within ±{best['MAE_mean']:.3f} m/s")


if __name__ == "__main__":
    # Load data
    X, y, groups, feature_cols = load_data(DATA_PATH)

    # Run LOSO evaluation
    models = get_models()
    results_df = evaluate_loso(models, X, y, groups)

    # Save results
    Path(RESULTS_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_SAVE_PATH, index=False)
    print(f"[INFO] Results saved to {RESULTS_SAVE_PATH}")

    # Print summary
    print_summary(results_df)

    # Train and save final model
    train_final_model(X, y, MODEL_SAVE_PATH)
