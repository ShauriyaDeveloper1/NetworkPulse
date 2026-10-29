"""
NetworkPulse: Stationarity Analysis Module
Implements Augmented Dickey-Fuller (ADF) test and differencing inspection on training data.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def run_adf_test(series: pd.Series, maxlag: int = None) -> Dict[str, Any]:
    """
    Executes Augmented Dickey-Fuller (ADF) test on a given series.
    
    H0 (Null Hypothesis): The series has a unit root (non-stationary).
    H1 (Alternative Hypothesis): The series has no unit root (stationary).
    """
    cleaned_series = series.dropna()
    adf_result = adfuller(cleaned_series, maxlag=maxlag, autolag="AIC")
    
    test_statistic = float(adf_result[0])
    p_value = float(adf_result[1])
    lags_used = int(adf_result[2])
    n_obs = int(adf_result[3])
    critical_values = {k: float(v) for k, v in adf_result[4].items()}
    icbest = float(adf_result[5]) if len(adf_result) > 5 else None

    # Decision rule: reject H0 if p_value < 0.05 and test_statistic < critical_values['5%']
    is_stationary_5pct = (p_value < 0.05) and (test_statistic < critical_values["5%"])
    is_stationary_1pct = (p_value < 0.01) and (test_statistic < critical_values["1%"])

    return {
        "test_statistic": test_statistic,
        "p_value": p_value,
        "lags_used": lags_used,
        "n_obs": n_obs,
        "critical_values": critical_values,
        "aic_at_best_lag": icbest,
        "is_stationary_at_5pct": is_stationary_5pct,
        "is_stationary_at_1pct": is_stationary_1pct,
        "conclusion": (
            "Reject null hypothesis (H0): Series is stationary."
            if is_stationary_5pct
            else "Fail to reject null hypothesis (H0): Series is non-stationary."
        )
    }


def analyze_stationarity(
    train_series: pd.Series,
    output_plot_path: str = "artifacts/stationarity_analysis.png"
) -> Dict[str, Any]:
    """
    Analyzes stationarity of original and first-differenced training series.
    Generates rolling mean/std visualization.
    """
    import os
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)

    # 1. ADF on raw training series
    raw_adf = run_adf_test(train_series)

    # 2. First difference
    diff_series = train_series.diff().dropna()
    diff_adf = run_adf_test(diff_series)

    # 3. Rolling statistics (24h window)
    window_size = 24
    rolling_mean = train_series.rolling(window=window_size).mean()
    rolling_std = train_series.rolling(window=window_size).std()

    # 4. Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    
    # Subplot 1: Raw training series with rolling statistics
    axes[0].plot(train_series.index, train_series.values, label="Raw RTT (ms)", color="#2b5c8f", alpha=0.5, linewidth=0.8)
    axes[0].plot(rolling_mean.index, rolling_mean.values, label=f"{window_size}h Rolling Mean", color="#e63946", linewidth=1.5)
    axes[0].set_title(f"NetworkPulse Training Series: Round Trip Time (RTT) & {window_size}h Rolling Mean", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("RTT (ms)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Rolling standard deviation
    axes[1].plot(rolling_std.index, rolling_std.values, label=f"{window_size}h Rolling Std Dev", color="#f4a261", linewidth=1.2)
    axes[1].set_title(f"Rolling Volatility ({window_size}h Window)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Std Dev (ms)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: First-differenced series
    axes[2].plot(diff_series.index, diff_series.values, label="First Differenced RTT", color="#2a9d8f", alpha=0.6, linewidth=0.8)
    axes[2].set_title(f"First-Differenced Series: Δ RTT (t) - RTT (t-1)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Δ RTT (ms)")
    axes[2].set_xlabel("Timestamp")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_plot_path, dpi=200)
    plt.close(fig)

    # Determine recommended differencing order d
    if raw_adf["is_stationary_at_5pct"]:
        recommended_d = 0
        rationale = (
            f"Raw series ADF test statistic ({raw_adf['test_statistic']:.4f}) is less than the 1% critical value "
            f"({raw_adf['critical_values']['1%']:.4f}) with p-value={raw_adf['p_value']:.4e}. "
            "The null hypothesis of a unit root is rejected. Differencing d=0 is statistically justified, "
            "though d=1 can also be considered to ensure zero mean-shift risk."
        )
    else:
        recommended_d = 1
        rationale = (
            f"Raw series has ADF p-value={raw_adf['p_value']:.4f} >= 0.05. "
            f"First difference achieved p-value={diff_adf['p_value']:.4e}. Differencing order d=1 is required."
        )

    return {
        "raw_series_adf": raw_adf,
        "first_diff_adf": diff_adf,
        "recommended_d": recommended_d,
        "rationale": rationale,
        "plot_path": output_plot_path
    }
