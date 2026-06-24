import os, glob, base64
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator

from backend.config import settings
from backend.logging_config import logger
from backend.database import get_session, bulk_insert_records, TrafficRecord
from backend.utils import normalize_timestamp_series, parse_datetime


UPLOAD_DIR: str = settings.upload_dir
DEFAULT_TZ: str = settings.default_tz
REQUIRED_VEHICLE_COLS = ["two_wheeler", "four_wheeler", "heavy_vehicle", "emergency_vehicle"]


class TrafficDataRow(BaseModel):
    timestamp: datetime
    location: str
    two_wheeler: int = Field(ge=0)
    four_wheeler: int = Field(ge=0)
    heavy_vehicle: int = Field(ge=0)
    emergency_vehicle: int = Field(ge=0)

    @field_validator("timestamp", mode="before")
    @classmethod
    def coerce_timestamp(cls, v):
        if isinstance(v, str):
            parsed = parse_datetime(v)
            if parsed is None:
                raise ValueError(f"Cannot parse timestamp: {v}")
            return parsed
        if isinstance(v, pd.Timestamp):
            return v.to_pydatetime()
        if isinstance(v, datetime):
            return v
        raise ValueError(f"Unexpected timestamp type: {type(v)}")


def read_csv_flexible(path: str, timezone: str | None = None) -> pd.DataFrame:
    tz = timezone or DEFAULT_TZ
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    col_map = {c.lower(): c for c in df.columns}

    if "timestamp" in col_map:
        ts_col = col_map["timestamp"]
    elif "date" in col_map and "time" in col_map:
        ts_col = "timestamp"
        df[ts_col] = df[col_map["date"]].astype(str) + " " + df[col_map["time"]].astype(str)
    else:
        raise ValueError(f"CSV {path} must have a 'timestamp' column or 'date'+'time' columns. Found: {list(df.columns)}")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    bad_mask = df[ts_col].isna()
    bad_count = bad_mask.sum()
    if bad_count:
        logger.warning("%s — dropped %d rows with unparseable timestamps", path, bad_count)
    df = df[~bad_mask].copy()
    df["timestamp"] = normalize_timestamp_series(df[ts_col], tz)
    df = df.sort_values("timestamp").reset_index(drop=True)

    location_col = col_map.get("location")
    if location_col:
        df["location"] = df[location_col].astype(str)

    for col in REQUIRED_VEHICLE_COLS:
        if col not in df.columns:
            if col in col_map:
                df[col] = pd.to_numeric(df[col_map[col]], errors="coerce").fillna(0).astype(int)
            else:
                df[col] = 0

    return df[["timestamp", "location"] + REQUIRED_VEHICLE_COLS]


def save_uploaded_contents(contents: str, filename: str) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    safe = filename
    path = os.path.join(UPLOAD_DIR, safe)
    if os.path.exists(path):
        base_name, ext = os.path.splitext(safe)
        safe = f"{base_name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}{ext}"
        path = os.path.join(UPLOAD_DIR, safe)
    with open(path, "wb") as f:
        f.write(decoded)
    logger.info("Saved uploaded file to %s", path)
    return path


def combine_all_uploads(timezone: str | None = None) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(UPLOAD_DIR, "*.csv")))
    if not files:
        logger.warning("No CSV files found in %s", UPLOAD_DIR)
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    for f in files:
        try:
            dfs.append(read_csv_flexible(f, timezone))
        except Exception as exc:
            logger.error("Skipping %s: %s", f, exc)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp", "location"], keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    logger.info("Combined %d rows from %d files", len(combined), len(files))
    return combined


def validate_and_ingest(contents: str, filename: str) -> dict:
    errors: list[str] = []
    try:
        path = save_uploaded_contents(contents, filename)
    except Exception as exc:
        logger.error("Failed to save upload: %s", exc)
        return {"status": "error", "rows_ingested": 0, "errors": [str(exc)]}

    try:
        df = read_csv_flexible(path)
    except Exception as exc:
        logger.error("Failed to read CSV %s: %s", path, exc)
        return {"status": "error", "rows_ingested": 0, "errors": [str(exc)]}

    valid_rows: list[TrafficDataRow] = []
    for idx, row in df.iterrows():
        try:
            validated = TrafficDataRow(
                timestamp=row["timestamp"],
                location=row["location"],
                two_wheeler=row["two_wheeler"],
                four_wheeler=row["four_wheeler"],
                heavy_vehicle=row["heavy_vehicle"],
                emergency_vehicle=row["emergency_vehicle"],
            )
            valid_rows.append(validated)
        except Exception as exc:
            msg = f"Row {idx}: {exc}"
            errors.append(msg)
            logger.warning("Validation error — %s", msg)

    if valid_rows:
        try:
            records_dicts = [r.model_dump() for r in valid_rows]
            with get_session() as session:
                bulk_insert_records(session, records_dicts)
            logger.info("Ingested %d valid rows from %s", len(valid_rows), filename)
        except Exception as exc:
            logger.error("DB bulk insert failed: %s", exc)
            return {"status": "error", "rows_ingested": 0, "errors": [str(exc)]}

    return {"status": "ok", "rows_ingested": len(valid_rows), "errors": errors}


def compute_pie_counts(df: pd.DataFrame) -> Optional[dict]:
    if df.empty:
        return None
    return {
        "4wheeler": int(df["four_wheeler"].sum()),
        "2wheeler": int(df["two_wheeler"].sum()),
        "heavy": int(df["heavy_vehicle"].sum()),
        "emergency": int(df["emergency_vehicle"].sum()),
    }


def compute_mean_vehicle_types(df: pd.DataFrame) -> Optional[dict]:
    if df.empty:
        return None
    return {
        "4wheeler": float(df["four_wheeler"].mean()),
        "2wheeler": float(df["two_wheeler"].mean()),
        "heavy": float(df["heavy_vehicle"].mean()),
        "emergency": float(df["emergency_vehicle"].mean()),
    }


def compute_hourly_avg(df: pd.DataFrame) -> tuple[float, float]:
    if df.empty:
        return 0.0, 1.0

    df = df.copy()
    df["total"] = df["two_wheeler"] + df["four_wheeler"] + df["heavy_vehicle"] + df["emergency_vehicle"]
    hourly_means = df.set_index("timestamp").resample("1h")["total"].mean()
    if hourly_means.empty:
        return 0.0, 1.0
    return float(hourly_means.mean()), float(hourly_means.max())


def compute_hourly_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total"] = df["two_wheeler"] + df["four_wheeler"] + df["heavy_vehicle"] + df["emergency_vehicle"]
    hourly = df.set_index("timestamp").resample("1h")["total"].mean().reset_index()
    hourly["hour_index"] = range(len(hourly))
    return hourly


def get_dataframe_from_db(
    location: str | None = None,
    from_ts: str | None = None,
    to_ts: str | None = None,
) -> pd.DataFrame:
    records: list[TrafficRecord] = []
    try:
        with get_session() as session:
            query = session.query(TrafficRecord)
            if location:
                query = query.filter(TrafficRecord.location == location)
            if from_ts:
                query = query.filter(TrafficRecord.timestamp >= parse_datetime(from_ts))
            if to_ts:
                query = query.filter(TrafficRecord.timestamp <= parse_datetime(to_ts))
            records = query.all()
    except Exception as exc:
        logger.error("DB query failed: %s — falling back to CSV uploads", exc)

    if records:
        rows = []
        for r in records:
            rows.append({
                "timestamp": r.timestamp,
                "location": r.location,
                "two_wheeler": r.two_wheeler,
                "four_wheeler": r.four_wheeler,
                "heavy_vehicle": r.heavy_vehicle,
                "emergency_vehicle": r.emergency_vehicle,
            })
        df = pd.DataFrame(rows)
        df["timestamp"] = normalize_timestamp_series(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)

    logger.info("DB empty, falling back to combine_all_uploads")
    return combine_all_uploads()
