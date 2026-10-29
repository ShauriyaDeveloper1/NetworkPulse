"""
NetworkPulse: Chronological Dataset Splitter
Implements strict chronological train/test split to prevent temporal data leakage.
"""

from typing import Tuple, Dict, Any
import pandas as pd


def chronological_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    target_col: str = "round_trip_time"
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """
    Performs a strictly chronological split (train_ratio train, 1-train_ratio test).
    
    Args:
        df: Datetime-indexed DataFrame
        train_ratio: Fraction of data for training (default 0.80)
        target_col: Target column name
        
    Returns:
        train_series: Target time series for training
        test_series: Target time series for testing
        split_meta: Dictionary of split metrics and boundaries
    """
    total_len = len(df)
    train_size = int(total_len * train_ratio)
    test_size = total_len - train_size

    # Split without any shuffling
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    train_series = train_df[target_col]
    test_series = test_df[target_col]

    split_meta = {
        "total_observations": total_len,
        "train_observations": train_size,
        "test_observations": test_size,
        "actual_train_ratio": train_size / total_len,
        "actual_test_ratio": test_size / total_len,
        "train_start": str(train_series.index.min()),
        "train_end": str(train_series.index.max()),
        "test_start": str(test_series.index.min()),
        "test_end": str(test_series.index.max()),
        "temporal_leakage_detected": bool(train_series.index.max() >= test_series.index.min()),
    }

    return train_series, test_series, split_meta
