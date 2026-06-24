from __future__ import annotations

import pandas as pd
import pytz


def normalize_timestamp_series(
    ts: pd.Series, timezone: str = "Asia/Kolkata"
) -> pd.Series:
    tz = pytz.timezone(timezone)
    result = pd.to_datetime(ts, errors="coerce")
    if result.dt.tz is None:
        result = result.dt.tz_localize(tz, ambiguous="infer", nonexistent="NaT")
    else:
        result = result.dt.tz_convert(tz)
    return result


def parse_datetime(value: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not parse datetime: {value!r}") from e


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    try:
        if b == 0:
            return default
        return a / b
    except (ZeroDivisionError, TypeError, ValueError):
        return default
