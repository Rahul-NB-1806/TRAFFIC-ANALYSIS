import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_model(X, y, model_type: str = "linear") -> Tuple[object, Dict]:
    """Train model. model_type: 'linear' | 'rf' | 'auto'
    'auto' tries both and picks best by RMSE.
    Returns (model, metrics)."""
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Empty training data")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X rows ({X.shape[0]}) and y rows ({y.shape[0]}) must match")

    models_to_try = []
    if model_type == "linear":
        models_to_try = [("linear", LinearRegression())]
    elif model_type == "rf":
        models_to_try = [("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))]
    elif model_type == "auto":
        models_to_try = [
            ("linear", LinearRegression()),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ]
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'linear', 'rf', or 'auto'")

    best_model = None
    best_metrics = {"mae": float("inf"), "rmse": float("inf")}
    best_name = None

    for name, model in models_to_try:
        model.fit(X, y)
        y_pred = model.predict(X)
        mae = float(mean_absolute_error(y, y_pred))
        rmse_val = _rmse(y, y_pred)
        logger.info(f"{name}: MAE={mae:.4f}, RMSE={rmse_val:.4f}")

        if rmse_val < best_metrics["rmse"]:
            best_model = model
            best_metrics = {"mae": mae, "rmse": rmse_val}
            best_name = name

    best_metrics["model_type"] = best_name
    logger.info(f"Best model: {best_name} with RMSE={best_metrics['rmse']:.4f}")
    return best_model, best_metrics


def predict(model, X) -> np.ndarray:
    """Generate predictions."""
    if model is None:
        raise ValueError("Model is None")
    logger.debug(f"Generating predictions for {X.shape[0]} samples")
    return model.predict(X)


def evaluate(y_true, y_pred) -> Dict[str, float]:
    """Return dict with mae, rmse, mape."""
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Empty arrays for evaluation")
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) lengths must match")

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse_val = _rmse(y_true, y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        mape = float("inf")
    else:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    logger.info(f"Evaluation — MAE={mae:.4f}, RMSE={rmse_val:.4f}, MAPE={mape:.2f}%")
    return {"mae": mae, "rmse": rmse_val, "mape": mape}


def train_test_split_temporal(df, test_hours: int = 24):
    """Time-aware train/test split (no leakage)."""
    if df.empty:
        raise ValueError("DataFrame is empty")
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column")

    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = len(df_sorted) - test_hours
    if split_idx <= 0:
        raise ValueError(f"Not enough data: need at least {test_hours + 1} rows, got {len(df_sorted)}")

    train = df_sorted.iloc[:split_idx].copy()
    test = df_sorted.iloc[split_idx:].copy()
    logger.info(f"Temporal split: {len(train)} train / {len(test)} test rows")
    return train, test
