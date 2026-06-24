import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from models.schemas import AnalyticsResult

VEHICLE_COLUMNS = ["two_wheeler", "four_wheeler", "heavy_vehicle", "emergency_vehicle"]


def compute_vehicle_totals(df: pd.DataFrame) -> dict:
    return {
        col: int(df[col].sum()) for col in VEHICLE_COLUMNS
    } | {"total": int(df["total"].sum()) if "total" in df.columns else int(df[VEHICLE_COLUMNS].sum().sum())}


def compute_location_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "location" not in df.columns:
        return pd.DataFrame(columns=["location", "total"])
    return (
        df.groupby("location", as_index=False)["total"]
        .sum()
        .sort_values("total", ascending=False)
        .reset_index(drop=True)
    )


def compute_hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["hour"] + VEHICLE_COLUMNS)
    df = df.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    return (
        df.groupby("hour", as_index=False)[VEHICLE_COLUMNS]
        .sum()
        .sort_values("hour")
        .reset_index(drop=True)
    )


def compute_peak_hours(df: pd.DataFrame, threshold_pct: float = 0.8) -> list[dict]:
    if df.empty:
        return []
    df = df.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    hourly = df.groupby("hour")["total"].sum().reset_index()
    max_total = hourly["total"].max()
    if max_total == 0:
        return []
    threshold = max_total * threshold_pct
    peaks = hourly[hourly["total"] >= threshold]
    result = []
    for _, row in peaks.iterrows():
        hour_mask = df["hour"] == row["hour"]
        vehicles = {
            col: int(df.loc[hour_mask, col].sum()) for col in VEHICLE_COLUMNS
        }
        result.append({
            "hour": int(row["hour"]),
            "total": int(row["total"]),
            "vehicles": vehicles,
        })
    return sorted(result, key=lambda x: x["hour"])


def detect_anomalies(df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    if df.empty:
        result = df.copy()
        result["is_anomaly"] = pd.Series(dtype=bool)
        return result
    result = df.copy()
    mean = result["total"].mean()
    std = result["total"].std()
    if std == 0:
        result["is_anomaly"] = False
    else:
        result["is_anomaly"] = (result["total"] - mean).abs() / std > z_threshold
    return result


def compute_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date"] + VEHICLE_COLUMNS + ["total"])
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    return (
        df.groupby("date", as_index=False)[VEHICLE_COLUMNS + ["total"]]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )


def compute_location_comparison(df: pd.DataFrame) -> list[dict]:
    if df.empty or "location" not in df.columns:
        return []
    df = df.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    results = []
    for location, group in df.groupby("location"):
        total = int(group["total"].sum())
        hourly_avg = group.groupby("hour")["total"].sum()
        avg_hourly = float(hourly_avg.mean()) if not hourly_avg.empty else 0.0
        peak_hour = int(hourly_avg.idxmax()) if not hourly_avg.empty else 0
        grand_total = group[VEHICLE_COLUMNS].sum()
        mix = {}
        for col in VEHICLE_COLUMNS:
            v = int(grand_total[col])
            pct = round(v / grand_total.sum() * 100, 1) if grand_total.sum() > 0 else 0.0
            mix[col] = {"count": v, "pct": pct}
        results.append({
            "location": location,
            "total": total,
            "avg_hourly": avg_hourly,
            "peak_hour": peak_hour,
            "mix": mix,
        })
    return sorted(results, key=lambda x: x["total"], reverse=True)


def compute_time_range(df: pd.DataFrame) -> tuple:
    if df.empty or "timestamp" not in df.columns:
        return ("", "")
    ts = pd.to_datetime(df["timestamp"])
    return (str(ts.min()), str(ts.max()))
