import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def create_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add hour, day_of_week, month, is_weekend, is_rush_hour features."""
    if df.empty:
        logger.warning("Empty DataFrame passed to create_time_features")
        return df

    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in DataFrame")

    result = df.copy()
    dt = pd.to_datetime(result[timestamp_col])
    result["hour"] = dt.dt.hour
    result["day_of_week"] = dt.dt.dayofweek
    result["month"] = dt.dt.month
    result["is_weekend"] = dt.dt.dayofweek >= 5
    result["is_rush_hour"] = (
        ((dt.dt.hour >= 7) & (dt.dt.hour <= 9)) |
        ((dt.dt.hour >= 16) & (dt.dt.hour <= 19))
    ).astype(int)
    logger.debug(f"Added time features to {len(result)} rows")
    return result


def create_rolling_features(df: pd.DataFrame, windows: Optional[list] = None) -> pd.DataFrame:
    """Add rolling averages for given window sizes (default [3, 6, 12] hours)."""
    if df.empty:
        logger.warning("Empty DataFrame passed to create_rolling_features")
        return df

    if windows is None:
        windows = [3, 6, 12]

    result = df.copy()
    target_col = "total"
    if target_col not in result.columns:
        logger.warning("'total' column not found; skipping rolling features")
        return result

    for w in windows:
        result[f"{target_col}_rolling_{w}"] = result[target_col].rolling(window=w, min_periods=1).mean()

    logger.debug(f"Added rolling features with windows {windows} for '{target_col}'")
    return result


def prepare_forecast_data(df: pd.DataFrame, target_col: str = "total") -> Tuple[np.ndarray, np.ndarray, list]:
    """Prepare X, y for time-series forecasting with lag features.
    Returns (X, y, feature_names)."""
    if df.empty:
        raise ValueError("DataFrame is empty")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    data = df.copy()

    if "hour" not in data.columns:
        logger.info("hour column missing — creating time features")
        data = create_time_features(data)

    n = len(data)
    lags = [l for l in [1, 2, 3, 6, 12, 24] if l < n]
    if not lags:
        lags = [1]

    for lag in lags:
        col = f"{target_col}_lag_{lag}"
        data[col] = data[target_col].shift(lag)
        if data[col].isna().all():
            data.drop(columns=[col], inplace=True)

    data = data.dropna()
    feature_cols = [c for c in data.columns if c != target_col and c != "timestamp"]

    X = data[feature_cols].values.astype(np.float64)
    y = data[target_col].values.astype(np.float64)

    logger.info(f"Prepared forecast data: X shape {X.shape}, y shape {y.shape}")
    return X, y, feature_cols
