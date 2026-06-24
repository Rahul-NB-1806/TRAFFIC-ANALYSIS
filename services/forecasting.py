import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def prepare_trend_data(df: pd.DataFrame) -> tuple:
    if df.empty:
        return np.array([]).reshape(-1, 1), np.array([])
    df = df.copy().sort_values("timestamp")
    df["hour_index"] = np.arange(len(df))
    X = df[["hour_index"]].values
    y = df["total"].values
    return X, y


def train_trend_model(X, y, model_type: str = "auto") -> tuple:
    if len(X) < 2:
        raise ValueError(f"Need at least 2 samples, got {len(X)}")

    if model_type == "auto":
        model_type = "rf" if len(X) >= 10 else "linear"

    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = evaluate_forecast(y, y_pred)
    logger.info(
        "Trained %s model: MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%",
        model_type, metrics["mae"], metrics["rmse"], metrics["mape"],
    )
    return model, metrics


def generate_forecast(model, last_idx: int, steps: int) -> np.ndarray:
    future_idx = np.arange(last_idx + 1, last_idx + steps + 1).reshape(-1, 1)
    return model.predict(future_idx)


def evaluate_forecast(y_true, y_pred) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = 0.0
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}
