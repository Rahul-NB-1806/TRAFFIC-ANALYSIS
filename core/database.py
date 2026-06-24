from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, func
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from typing import Optional

from core.config import settings

Base = declarative_base()


class TrafficRecord(Base):
    __tablename__ = "traffic_records"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True)
    location = Column(String, index=True)
    two_wheeler = Column(Integer)
    four_wheeler = Column(Integer)
    heavy_vehicle = Column(Integer)
    emergency_vehicle = Column(Integer)
    total = Column(Integer)
    source_file = Column(String)
    is_anomaly = Column(Boolean, default=False)


_engine = None
_SessionLocal = None


def init_db(db_url: str = None) -> None:
    global _engine, _SessionLocal
    url = db_url or settings.database_url
    _engine = create_engine(url, echo=settings.debug)
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine)


def get_session():
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


def bulk_insert(session, records: list[dict]) -> int:
    session.bulk_insert_mappings(TrafficRecord, records)
    session.commit()
    return len(records)


def query_records(
    session,
    location: Optional[str] = None,
    from_ts: Optional[datetime] = None,
    to_ts: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> list[TrafficRecord]:
    query = session.query(TrafficRecord)
    if location:
        query = query.filter(TrafficRecord.location == location)
    if from_ts:
        query = query.filter(TrafficRecord.timestamp >= from_ts)
    if to_ts:
        query = query.filter(TrafficRecord.timestamp <= to_ts)
    query = query.order_by(TrafficRecord.timestamp.desc())
    if limit:
        query = query.limit(limit)
    return query.all()


def get_locations(session) -> list[str]:
    results = session.query(TrafficRecord.location).distinct().all()
    return [r[0] for r in results if r[0] is not None]


def get_time_range(session) -> tuple:
    result = session.query(
        func.min(TrafficRecord.timestamp),
        func.max(TrafficRecord.timestamp),
    ).first()
    return (result[0], result[1]) if result[0] else (None, None)


def get_record_count(session) -> int:
    return session.query(func.count(TrafficRecord.id)).scalar() or 0


def mark_anomalies(session, record_ids: list[int]) -> int:
    count = (
        session.query(TrafficRecord)
        .filter(TrafficRecord.id.in_(record_ids))
        .update({TrafficRecord.is_anomaly: True}, synchronize_session="fetch")
    )
    session.commit()
    return count
