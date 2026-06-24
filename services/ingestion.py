import os
import glob
import base64
import logging

import pandas as pd
from datetime import datetime
from typing import Optional

from core.config import settings
from core.database import get_session, bulk_insert, query_records, TrafficRecord, get_locations, get_record_count
from models.schemas import TrafficRow, UploadReport

logger = logging.getLogger(__name__)

VEHICLE_COLUMNS = ["two_wheeler", "four_wheeler", "heavy_vehicle", "emergency_vehicle"]


def _coalesce_timestamp_column(df: pd.DataFrame, timezone: Optional[str] = None) -> pd.Series:
    ts_cols = [c for c in df.columns if c.lower() in ("timestamp", "datetime", "date", "time")]
    if ts_cols:
        col = ts_cols[0]
        return pd.to_datetime(df[col])
    date_cols = [c for c in df.columns if "date" in c.lower()]
    time_cols = [c for c in df.columns if "time" in c.lower()]
    if date_cols and time_cols:
        return pd.to_datetime(
            df[date_cols[0]].astype(str) + " " + df[time_cols[0]].astype(str)
        )
    raise ValueError("No timestamp/date+time column found in CSV")


def _validate_vehicle_columns(df: pd.DataFrame) -> None:
    missing = [c for c in VEHICLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required vehicle columns: {missing}")
    for col in VEHICLE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


def read_csv(path: str, timezone: str = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = _coalesce_timestamp_column(df, timezone)
    if "location" not in df.columns:
        raise ValueError("Missing required column: location")
    df["location"] = df["location"].astype(str).str.strip()
    _validate_vehicle_columns(df)
    df["total"] = df[VEHICLE_COLUMNS].sum(axis=1)
    return df[["timestamp", "location"] + VEHICLE_COLUMNS + ["total"]]


def save_upload(contents: str, filename: str) -> str:
    upload_dir = settings.upload_dir
    os.makedirs(upload_dir, exist_ok=True)
    raw = base64.b64decode(contents)
    dest = os.path.join(upload_dir, filename)
    with open(dest, "wb") as f:
        f.write(raw)
    logger.info("Saved upload to %s (%d bytes)", dest, len(raw))
    return dest


def ingest_csv(contents: str, filename: str) -> UploadReport:
    errors: list[str] = []
    path = save_upload(contents, filename)
    try:
        df = read_csv(path, settings.default_tz)
    except Exception as e:
        logger.exception("CSV parse failed for %s", filename)
        return UploadReport(filename=filename, rows_ingested=0, errors=[str(e)])

    rows: list[dict] = []
    for _, row in df.iterrows():
        try:
            TrafficRow(**row.to_dict())
        except Exception as e:
            errors.append(f"Row {len(rows) + 1}: {e}")
            continue
        rows.append(
            {
                "timestamp": row["timestamp"].to_pydatetime(),
                "location": row["location"],
                "two_wheeler": int(row["two_wheeler"]),
                "four_wheeler": int(row["four_wheeler"]),
                "heavy_vehicle": int(row["heavy_vehicle"]),
                "emergency_vehicle": int(row["emergency_vehicle"]),
                "total": int(row["total"]),
                "source_file": filename,
            }
        )

    if not rows:
        return UploadReport(filename=filename, rows_ingested=0, errors=errors or ["No valid rows"])

    session = get_session()
    try:
        count = bulk_insert(session, rows)
        locations = get_locations(session)
        time_range = (str(rows[0]["timestamp"]), str(rows[-1]["timestamp"]))
        total_v = sum(r["total"] for r in rows)
    except Exception as e:
        logger.exception("DB insert failed")
        session.rollback()
        return UploadReport(filename=filename, rows_ingested=0, errors=[str(e)])
    finally:
        session.close()

    logger.info("Ingested %d rows from %s", count, filename)
    return UploadReport(
        filename=filename,
        rows_ingested=count,
        errors=errors,
        locations=locations,
        time_range=time_range,
        total_vehicles=total_v,
    )


def load_dataframe(
    location: Optional[str] = None,
    from_ts: Optional[str] = None,
    to_ts: Optional[str] = None,
) -> pd.DataFrame:
    session = get_session()
    try:
        count = get_record_count(session)
        if count == 0:
            logger.info("DB empty, falling back to CSV files in %s", settings.upload_dir)
            return _load_from_csv_fallback()
        from_ts_dt = pd.to_datetime(from_ts) if from_ts else None
        to_ts_dt = pd.to_datetime(to_ts) if to_ts else None
        records = query_records(session, location=location, from_ts=from_ts_dt, to_ts=to_ts_dt)
        if not records:
            return pd.DataFrame(columns=["timestamp", "location"] + VEHICLE_COLUMNS + ["total"])
        return pd.DataFrame(
            [
                {
                    "timestamp": r.timestamp,
                    "location": r.location,
                    "two_wheeler": r.two_wheeler,
                    "four_wheeler": r.four_wheeler,
                    "heavy_vehicle": r.heavy_vehicle,
                    "emergency_vehicle": r.emergency_vehicle,
                    "total": r.total,
                }
                for r in records
            ]
        )
    finally:
        session.close()


def _load_from_csv_fallback() -> pd.DataFrame:
    pattern = os.path.join(settings.upload_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame(columns=["timestamp", "location"] + VEHICLE_COLUMNS + ["total"])
    chunks = [read_csv(f, settings.default_tz) for f in files]
    return pd.concat(chunks, ignore_index=True)
