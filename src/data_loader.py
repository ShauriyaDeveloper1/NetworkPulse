"""
NetworkPulse: Data Ingestion and Validation Module
Handles loading, datetime parsing, chronological validation, and target verification.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


def load_and_validate_dataset(
    filepath: str = "drone_communication_dataset.csv",
    timestamp_col: str = "timestamp",
    target_col: str = "round_trip_time"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads dataset, parses timestamps with DD-MM-YYYY format,
    validates chronological order, verifies regularity and target bounds.

    Returns:
        df: pd.DataFrame with pd.DatetimeIndex
        metadata: Dict containing summary validation stats
    """
    # 1. Load raw CSV without altering or dropping data
    df = pd.read_csv(filepath)
    raw_shape = df.shape

    if timestamp_col not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found in dataset columns: {list(df.columns)}")

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    # 2. Parse timestamps with explicit format '%d-%m-%Y %H:%M' (day-first)
    df["datetime"] = pd.to_datetime(df[timestamp_col], format="%d-%m-%Y %H:%M")
    
    # 3. Check chronological ordering
    is_monotonic = df["datetime"].is_monotonic_increasing
    if not is_monotonic:
        # Sort if needed, but alert
        df = df.sort_values("datetime").reset_index(drop=True)
        was_sorted = False
    else:
        was_sorted = True

    # 4. Check for duplicate timestamps
    duplicate_count = int(df["datetime"].duplicated().sum())

    # 5. Check time frequency / interval regularity
    time_diffs = df["datetime"].diff().dropna()
    regular_1h_count = int((time_diffs == pd.Timedelta(hours=1)).sum())
    total_diffs = len(time_diffs)
    is_hourly_regular = (regular_1h_count == total_diffs)

    # 6. Target distribution and validity checks
    target_series = df[target_col]
    target_missing = int(target_series.isnull().sum())
    target_inf = int(np.isinf(target_series).sum())
    target_negative = int((target_series < 0).sum())
    target_zeros = int((target_series == 0).sum())

    target_stats = {
        "count": int(target_series.count()),
        "mean": float(target_series.mean()),
        "std": float(target_series.std()),
        "min": float(target_series.min()),
        "p25": float(target_series.quantile(0.25)),
        "median": float(target_series.median()),
        "p75": float(target_series.quantile(0.75)),
        "p95": float(target_series.quantile(0.95)),
        "p99": float(target_series.quantile(0.99)),
        "max": float(target_series.max()),
        "missing": target_missing,
        "inf_count": target_inf,
        "negative_count": target_negative,
        "zero_count": target_zeros,
    }

    # Set datetime as index and set frequency explicitly
    df = df.set_index("datetime")
    df.index.freq = pd.infer_freq(df.index) or "h"

    metadata = {
        "raw_shape": raw_shape,
        "start_timestamp": str(df.index.min()),
        "end_timestamp": str(df.index.max()),
        "is_chronological_original": was_sorted,
        "duplicate_timestamps": duplicate_count,
        "is_hourly_regular": is_hourly_regular,
        "total_observations": len(df),
        "target_stats": target_stats,
    }

    return df, metadata
