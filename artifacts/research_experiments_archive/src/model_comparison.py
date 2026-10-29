"""
NetworkPulse: Stage 2 Model Comparison & Visualization Module
Generates side-by-side comparative plots, residual analyses, and metrics tables for ARIMA vs XGBoost.
"""

from typing import Dict, Any, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.evaluation import calculate_metrics


def generate_comparison_plots(
    y_true: pd.Series,
    y_arima: pd.Series,
    y_xgb: pd.Series,
    model: xgb.XGBRegressor,
    feature_names: List[str],
    output_dir: str = "artifacts/stage2"
) -> Dict[str, str]:
    """
    Generates all required Stage 2 visualizations and returns their filepaths.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    arima_metrics = calculate_metrics(y_true.values, y_arima.values)
    xgb_metrics = calculate_metrics(y_true.values, y_xgb.values)

    # -------------------------------------------------------------
    # 1. Actual vs XGBoost 24-Hour Forecast
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_true.index, y_true.values, label="Actual RTT (ms)", color="#1d3557", marker="o", linewidth=2)
    ax.plot(y_xgb.index, y_xgb.values, label=f"XGBoost Forecast (MAE: {xgb_metrics['MAE']:.2f} ms)", color="#2a9d8f", marker="^", linestyle="--", linewidth=2)
    ax.set_title(
        f"NetworkPulse 24-Hour Horizon: Actual vs. XGBoost Forecast\n"
        f"(MAE: {xgb_metrics['MAE']:.2f} ms | RMSE: {xgb_metrics['RMSE']:.2f} ms | sMAPE: {xgb_metrics['sMAPE']:.2f}%)",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Round Trip Time (ms)", fontsize=11)
    ax.set_xlabel("Timestamp (Hourly)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()
    p1 = os.path.join(output_dir, "actual_vs_xgboost_24h.png")
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    plot_paths["actual_vs_xgboost"] = p1

    # -------------------------------------------------------------
    # 2. ARIMA vs XGBoost vs Actual RTT Comparison Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.plot(y_true.index, y_true.values, label="Actual RTT (ms)", color="#1d3557", marker="o", linewidth=2.2, zorder=4)
    ax.plot(y_arima.index, y_arima.values, label=f"ARIMA(0,1,1) Baseline (MAE: {arima_metrics['MAE']:.2f} ms, RMSE: {arima_metrics['RMSE']:.2f} ms)", color="#e63946", marker="s", linestyle="--", linewidth=1.8, alpha=0.9, zorder=3)
    ax.plot(y_xgb.index, y_xgb.values, label=f"XGBoost Regressor (MAE: {xgb_metrics['MAE']:.2f} ms, RMSE: {xgb_metrics['RMSE']:.2f} ms)", color="#2a9d8f", marker="^", linestyle="-.", linewidth=2.0, alpha=0.95, zorder=5)
    
    ax.set_title(
        f"NetworkPulse 24-Hour Forecast Comparison: ARIMA(0,1,1) Baseline vs. XGBoost\n"
        f"Evaluation Window: {y_true.index.min().strftime('%Y-%m-%d %H:%M')} to {y_true.index.max().strftime('%Y-%m-%d %H:%M')}",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Round Trip Time (ms)", fontsize=11)
    ax.set_xlabel("Timestamp (Hourly)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    p2 = os.path.join(output_dir, "arima_vs_xgboost_comparison_24h.png")
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    plot_paths["arima_vs_xgboost"] = p2

    # -------------------------------------------------------------
    # 3. XGBoost Residual Diagnostics 4-Panel Plot
    # -------------------------------------------------------------
    residuals = y_true.values - y_xgb.values
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Residuals over time
    axes[0, 0].plot(y_true.index, residuals, marker="o", color="#2a9d8f", linewidth=1.5)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0, 0].set_title("XGBoost Residuals over Time (y_true - y_pred)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Residual (ms)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # (b) Residual Distribution / Histogram
    axes[0, 1].hist(residuals, bins=15, density=True, alpha=0.6, color="#457b9d", edgecolor="black")
    x_grid = np.linspace(min(residuals), max(residuals), 100)
    if res_std > 0:
        axes[0, 1].plot(x_grid, stats.norm.pdf(x_grid, res_mean, res_std), color="#e63946", linewidth=2, label="Normal Fit")
    axes[0, 1].set_title(f"Residual Distribution (Mean: {res_mean:.2f} ms, Std: {res_std:.2f} ms)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Residual (ms)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # (c) Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Normal Q-Q Plot of XGBoost Residuals", fontsize=11, fontweight="bold")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # (d) Autocorrelation of Residuals
    plot_acf(residuals, ax=axes[1, 1], lags=min(12, len(residuals) - 1), title="XGBoost Residual Autocorrelation (ACF)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    p3 = os.path.join(output_dir, "xgboost_residual_diagnostics.png")
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    plot_paths["xgboost_residuals"] = p3

    # -------------------------------------------------------------
    # 4. Feature Importance Plot (Top 15 Features by Gain)
    # -------------------------------------------------------------
    booster = model.get_booster()
    importance_gain = booster.get_score(importance_type="gain")
    
    # Map f0, f1... to feature names if necessary
    mapped_importances = {}
    for k, v in importance_gain.items():
        if k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            name = feature_names[idx] if idx < len(feature_names) else k
        else:
            name = k
        mapped_importances[name] = v

    imp_series = pd.Series(mapped_importances).sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    imp_series.plot(kind="barh", ax=ax, color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax.set_title("Top 15 Predictor Features by XGBoost Gain (Relative Importance)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Average Gain per Split")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p4 = os.path.join(output_dir, "xgboost_feature_importance.png")
    fig.savefig(p4, dpi=200)
    plt.close(fig)
    plot_paths["feature_importance"] = p4

    return plot_paths


def build_comparison_dataframe(
    y_true: pd.Series,
    y_arima: pd.Series,
    y_xgb: pd.Series
) -> pd.DataFrame:
    """
    Constructs an aligned DataFrame with timestamps, actuals, forecasts, and errors.
    """
    df = pd.DataFrame({
        "timestamp": y_true.index,
        "actual_rtt": y_true.values,
        "arima_forecast": y_arima.values,
        "xgboost_forecast": y_xgb.values,
        "arima_abs_error": np.abs(y_true.values - y_arima.values),
        "xgboost_abs_error": np.abs(y_true.values - y_xgb.values),
        "arima_squared_error": (y_true.values - y_arima.values) ** 2,
        "xgboost_squared_error": (y_true.values - y_xgb.values) ** 2
    })
    return df
