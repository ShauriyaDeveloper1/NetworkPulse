"""
NetworkPulse: ARIMA Baseline Pipeline Module
Handles ACF/PACF inspection, candidate (p,d,q) model comparison, model fitting, and forecasting.
All candidate selection and model fitting are strictly restricted to training data.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings


def plot_and_inspect_acf_pacf(
    series: pd.Series,
    lags: int = 40,
    output_plot_path: str = "artifacts/train_acf_pacf.png"
) -> Dict[str, Any]:
    """
    Computes ACF and PACF on the training series and saves the visualization.
    """
    import os
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)

    series_clean = series.dropna()
    acf_vals = acf(series_clean, nlags=lags)
    pacf_vals = pacf(series_clean, nlags=lags, method="yw")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series_clean, lags=lags, ax=axes[0], color="#2b5c8f", vlines_kwargs={"colors": "#2b5c8f"})
    axes[0].set_title(f"Autocorrelation Function (ACF) - First {lags} Lags", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    plot_pacf(series_clean, lags=lags, ax=axes[1], color="#e76f51", vlines_kwargs={"colors": "#e76f51"}, method="yw")
    axes[1].set_title(f"Partial Autocorrelation Function (PACF) - First {lags} Lags", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Partial Autocorrelation")
    axes[1].set_xlabel("Lag (Hours)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_plot_path, dpi=200)
    plt.close(fig)

    return {
        "acf_lags_1_to_5": [float(v) for v in acf_vals[1:6]],
        "pacf_lags_1_to_5": [float(v) for v in pacf_vals[1:6]],
        "plot_path": output_plot_path
    }


def compare_candidate_models(
    train_series: pd.Series,
    candidates: List[Tuple[int, int, int]] = None
) -> Tuple[pd.DataFrame, Tuple[int, int, int]]:
    """
    Fits a small, practical set of candidate (p,d,q) configurations on training data only.
    Compares models via AIC and BIC to select the baseline.
    """
    if candidates is None:
        candidates = [
            (1, 0, 0),  # AR(1) mean-reverting baseline
            (1, 0, 1),  # ARMA(1,1)
            (2, 0, 1),  # ARMA(2,1)
            (0, 0, 1),  # MA(1)
            (1, 1, 1),  # ARIMA(1,1,1)
            (0, 1, 1),  # ARIMA(0,1,1) exponential smoothing equivalent
            (1, 1, 0),  # ARIMA(1,1,0) differenced AR(1)
        ]

    results = []
    print("\n--- Evaluating Candidate ARIMA Models on Training Data ---")
    for order in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(train_series, order=order)
                fitted = model.fit()
                
            results.append({
                "order": str(order),
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "aic": float(fitted.aic),
                "bic": float(fitted.bic),
                "log_likelihood": float(fitted.llf),
                "converged": bool(fitted.mle_retvals.get("converged", True)) if hasattr(fitted, "mle_retvals") and fitted.mle_retvals else True,
            })
            print(f"  Candidate ARIMA{order}: AIC = {fitted.aic:.2f}, BIC = {fitted.bic:.2f}")
        except Exception as e:
            print(f"  Candidate ARIMA{order} failed: {e}")
            results.append({
                "order": str(order),
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "aic": np.nan,
                "bic": np.nan,
                "log_likelihood": np.nan,
                "converged": False,
            })

    results_df = pd.DataFrame(results).sort_values("aic").reset_index(drop=True)
    
    # Best model by AIC
    best_row = results_df.dropna().iloc[0]
    best_order = (int(best_row["p"]), int(best_row["d"]), int(best_row["q"]))
    
    return results_df, best_order


def fit_arima_baseline(
    train_series: pd.Series,
    order: Tuple[int, int, int]
) -> Any:
    """
    Fits the final ARIMA model using the specified order on the training series.
    """
    print(f"\nFitting selected ARIMA{order} baseline on {len(train_series)} training observations...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(train_series, order=order)
        fitted_model = model.fit()
    print("Model fitting complete.")
    return fitted_model


def generate_forecast(
    fitted_model: Any,
    steps: int = 24,
    alpha: float = 0.05
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Generates out-of-sample forecast for a configurable horizon (default 24 steps).
    Returns point forecasts and prediction confidence intervals.
    """
    forecast_res = fitted_model.get_forecast(steps=steps)
    point_forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=alpha)
    return point_forecast, conf_int
