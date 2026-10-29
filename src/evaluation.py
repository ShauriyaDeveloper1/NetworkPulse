"""
NetworkPulse: Evaluation and Residual Diagnostics Module
Computes MAE, RMSE, MAPE, sMAPE, and performs statistical residual diagnostics.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes MAE, RMSE, MAPE, and sMAPE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Check for near-zero values for standard MAPE
    epsilon = 1e-8
    has_zeros = np.any(np.abs(y_true) < epsilon)
    
    if not has_zeros:
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    else:
        mape = float("nan")

    # Symmetric MAPE (sMAPE) bounded in [0, 200%]
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Avoid zero division if both true and pred are 0
    non_zero_mask = denominator > epsilon
    if np.any(non_zero_mask):
        smape = float(np.mean(np.abs(y_true[non_zero_mask] - y_pred[non_zero_mask]) / denominator[non_zero_mask]) * 100.0)
    else:
        smape = 0.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape
    }


def evaluate_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    conf_int: pd.DataFrame = None,
    plot_prefix: str = "artifacts/arima_24h_evaluation"
) -> Dict[str, Any]:
    """
    Evaluates forecast against true observations and generates diagnostic plots.
    """
    import os
    os.makedirs(os.path.dirname(plot_prefix), exist_ok=True)

    metrics = calculate_metrics(y_true.values, y_pred.values)

    # Residuals
    residuals = y_true.values - y_pred.values
    res_series = pd.Series(residuals, index=y_true.index)

    # Residual summary
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals))
    
    # Ljung-Box test for remaining autocorrelation
    # For 24-step or horizon, test up to min(10, len(residuals)-1) lags
    max_lag = min(10, len(residuals) - 1)
    if max_lag >= 1:
        lb_res = acorr_ljungbox(residuals, lags=[max_lag], return_df=True)
        lb_stat = float(lb_res["lb_stat"].iloc[-1])
        lb_pvalue = float(lb_res["lb_pvalue"].iloc[-1])
    else:
        lb_stat = float("nan")
        lb_pvalue = float("nan")

    # Jarque-Bera normality test
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)

    residual_diagnostics = {
        "mean_residual": res_mean,
        "std_residual": res_std,
        "ljung_box_stat": lb_stat,
        "ljung_box_pvalue": lb_pvalue,
        "is_white_noise": bool(lb_pvalue > 0.05) if not np.isnan(lb_pvalue) else None,
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
    }

    # 1. Actual vs Predicted Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_true.index, y_true.values, label="Actual RTT (ms)", color="#1d3557", marker="o", linewidth=2)
    ax.plot(y_pred.index, y_pred.values, label="ARIMA Forecast (ms)", color="#e63946", linestyle="--", marker="s", linewidth=2)
    
    if conf_int is not None and not conf_int.empty:
        lower_col = conf_int.columns[0]
        upper_col = conf_int.columns[1]
        ax.fill_between(
            y_pred.index,
            conf_int[lower_col].values,
            conf_int[upper_col].values,
            color="#e63946",
            alpha=0.15,
            label="95% Confidence Interval"
        )

    ax.set_title(f"NetworkPulse 24-Hour Horizon: Actual vs. ARIMA Forecast\n(MAE: {metrics['MAE']:.2f} ms | RMSE: {metrics['RMSE']:.2f} ms | sMAPE: {metrics['sMAPE']:.2f}%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Round Trip Time (ms)", fontsize=11)
    ax.set_xlabel("Timestamp (Hourly)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()
    pred_plot_path = f"{plot_prefix}_actual_vs_pred.png"
    fig.savefig(pred_plot_path, dpi=200)
    plt.close(fig)

    # 2. Residual Diagnostics 4-panel Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Residuals over time
    axes[0, 0].plot(y_true.index, residuals, marker="o", color="#457b9d", linewidth=1.5)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0, 0].set_title("Forecast Residuals over Time (y_true - y_pred)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Residual (ms)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # (b) Residual Distribution / Histogram
    axes[0, 1].hist(residuals, bins=15, density=True, alpha=0.6, color="#2a9d8f", edgecolor="black")
    x_grid = np.linspace(min(residuals), max(residuals), 100)
    if res_std > 0:
        axes[0, 1].plot(x_grid, stats.norm.pdf(x_grid, res_mean, res_std), color="#e76f51", linewidth=2, label="Normal Fit")
    axes[0, 1].set_title(f"Residual Distribution (Mean: {res_mean:.2f}, Std: {res_std:.2f})", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Residual (ms)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # (c) Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Normal Q-Q Plot of Residuals", fontsize=11, fontweight="bold")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # (d) Autocorrelation of Residuals
    plot_acf(residuals, ax=axes[1, 1], lags=min(12, len(residuals) - 1), title="Residual Autocorrelation (ACF)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    res_plot_path = f"{plot_prefix}_residual_diagnostics.png"
    fig.savefig(res_plot_path, dpi=200)
    plt.close(fig)

    return {
        "metrics": metrics,
        "residual_diagnostics": residual_diagnostics,
        "actual_vs_pred_plot": pred_plot_path,
        "residual_plot": res_plot_path
    }
