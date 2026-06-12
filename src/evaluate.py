"""
evaluate.py
-----------
Model evaluation and SHAP explainability for the gait speed predictor.

Generates:
    - Actual vs Predicted plot
    - Residuals plot
    - SHAP summary plot (feature importance)
    - SHAP dependence plot (top feature vs speed)
    - Model results table

All plots saved to: data/processed/plots/
"""

import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # non-interactive backend for saving plots

# Paths
DATA_PATH = r"C:\Users\jumma\Downloads\GaitPhase\data\processed\model_features.csv"
PLOTS_DIR = r"C:\Users\jumma\Downloads\GaitPhase\data\processed\plots"
MODEL_PATH = r"C:\Users\jumma\Downloads\GaitPhase\api\model\xgb_speed_model.pkl"


def load_data(path: str) -> tuple:
    df = pd.read_csv(path)
    exclude_cols = ["subject_id", "speed"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    y = df["speed"]
    groups = df["subject_id"]
    return X, y, groups, feature_cols, df


def train_random_forest(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Train final Random Forest on full dataset for SHAP analysis."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        ))
    ])
    model.fit(X, y)
    return model


def get_loso_predictions(X: pd.DataFrame,
                          y: pd.Series,
                          groups: pd.Series) -> pd.DataFrame:
    """Run LOSO and collect all predictions for plotting."""
    logo = LeaveOneGroupOut()
    records = []

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        ))
    ])

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        for actual, predicted, subject in zip(
            y_test.values, y_pred, groups.iloc[test_idx].values
        ):
            records.append({
                "actual": actual,
                "predicted": predicted,
                "residual": actual - predicted,
                "subject_id": subject
            })

    return pd.DataFrame(records)


def plot_actual_vs_predicted(pred_df: pd.DataFrame, plots_dir: str) -> None:
    """Scatter plot of actual vs predicted walking speed."""
    fig, ax = plt.subplots(figsize=(8, 7))

    scatter = ax.scatter(
        pred_df["actual"], pred_df["predicted"],
        alpha=0.6, c=pred_df["actual"],
        cmap="viridis", edgecolors="white", linewidths=0.5, s=80
    )

    # Perfect prediction line
    lims = [0.5, 1.8]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect prediction", alpha=0.8)

    plt.colorbar(scatter, ax=ax, label="Actual Speed (m/s)")

    mae = mean_absolute_error(pred_df["actual"], pred_df["predicted"])
    r2 = r2_score(pred_df["actual"], pred_df["predicted"])

    ax.set_xlabel("Actual Walking Speed (m/s)", fontsize=13)
    ax.set_ylabel("Predicted Walking Speed (m/s)", fontsize=13)
    ax.set_title(
        f"Actual vs Predicted Walking Speed\n"
        f"Random Forest | LOSO CV | MAE = {mae:.3f} m/s | R² = {r2:.3f}",
        fontsize=13
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = Path(plots_dir) / "actual_vs_predicted.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {path}")


def plot_residuals(pred_df: pd.DataFrame, plots_dir: str) -> None:
    """Residuals vs actual speed plot."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        pred_df["actual"], pred_df["residual"],
        alpha=0.6, color="steelblue", edgecolors="white",
        linewidths=0.5, s=80
    )
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2, alpha=0.8)

    ax.set_xlabel("Actual Walking Speed (m/s)", fontsize=13)
    ax.set_ylabel("Residual (Actual − Predicted) m/s", fontsize=13)
    ax.set_title("Residuals vs Actual Speed\nRandom Forest | LOSO CV", fontsize=13)
    ax.grid(True, alpha=0.3)

    path = Path(plots_dir) / "residuals.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {path}")


def plot_shap_summary(model: Pipeline, X: pd.DataFrame,
                      feature_cols: list, plots_dir: str) -> None:
    """SHAP beeswarm summary plot — global feature importance."""
    print("[INFO] Computing SHAP values (this may take ~30 seconds)...")

    # Extract fitted RF from pipeline
    rf_model = model.named_steps["model"]
    scaler = model.named_steps["scaler"]
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_scaled_df)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_scaled_df,
        plot_type="dot",
        max_display=15,
        show=False
    )
    plt.title("SHAP Feature Importance — Walking Speed Prediction", fontsize=13, pad=15)

    path = Path(plots_dir) / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {path}")

    # Print top 10 features by mean |SHAP|
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_shap
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n=== Top 10 Features by SHAP Importance ===")
    print(shap_df.head(10).to_string(index=False))

    return shap_df


def plot_shap_dependence(model: Pipeline, X: pd.DataFrame,
                          feature_cols: list, plots_dir: str,
                          top_feature: str = "step_freq_mean") -> None:
    """SHAP dependence plot for top feature."""
    rf_model = model.named_steps["model"]
    scaler = model.named_steps["scaler"]
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_scaled_df)

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot(
        top_feature, shap_values, X_scaled_df,
        ax=ax, show=False
    )
    ax.set_title(
        f"SHAP Dependence: {top_feature}\n"
        "How step frequency drives speed prediction",
        fontsize=12
    )

    path = Path(plots_dir) / f"shap_dependence_{top_feature}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {path}")


def print_final_summary(pred_df: pd.DataFrame) -> None:
    """Print portfolio-ready metrics summary."""
    mae = mean_absolute_error(pred_df["actual"], pred_df["predicted"])
    rmse = np.sqrt(mean_squared_error(pred_df["actual"], pred_df["predicted"]))
    r2 = r2_score(pred_df["actual"], pred_df["predicted"])

    print("\n" + "=" * 55)
    print("  FINAL MODEL PERFORMANCE (LOSO Cross-Validation)")
    print("=" * 55)
    print(f"  Model:      Random Forest (200 trees, max_depth=6)")
    print(f"  Validation: Leave-One-Subject-Out (21 folds)")
    print(f"  MAE:        {mae:.4f} m/s")
    print(f"  RMSE:       {rmse:.4f} m/s")
    print(f"  R²:         {r2:.4f}")
    print(f"  Headline:   Predicts walking speed within ±{mae:.3f} m/s")
    print("=" * 55)


if __name__ == "__main__":
    # Setup
    plots_dir = Path(PLOTS_DIR)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=== Loading Data ===")
    X, y, groups, feature_cols, df = load_data(DATA_PATH)

    # LOSO predictions for plotting
    print("\n=== Running LOSO for Evaluation Plots ===")
    pred_df = get_loso_predictions(X, y, groups)

    # Plots
    print("\n=== Generating Plots ===")
    plot_actual_vs_predicted(pred_df, PLOTS_DIR)
    plot_residuals(pred_df, PLOTS_DIR)

    # Train full model for SHAP
    print("\n=== Training Full Model for SHAP ===")
    full_model = train_random_forest(X, y)
    shap_df = plot_shap_summary(full_model, X, feature_cols, PLOTS_DIR)
    plot_shap_dependence(full_model, X, feature_cols, PLOTS_DIR, "step_freq_mean")

    # Final summary
    print_final_summary(pred_df)

    print(f"\n[INFO] All plots saved to: {PLOTS_DIR}")
