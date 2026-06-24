from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from backend.config import settings

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
    source_file = Column(String)


_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def init_db(db_url: str | None = None) -> None:
    global _engine, _SessionLocal
    url = db_url or settings.database_url
    _engine = create_engine(url, echo=settings.debug)
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine)


def get_session() -> Session:
    if _SessionLocal is None:
        init_db()
    assert _SessionLocal is not None
    return _SessionLocal()


def bulk_insert_records(session: Session, records: list[dict[str, Any]]) -> int:
    session.bulk_insert_mappings(TrafficRecord, records)
    session.commit()
    return len(records)


def query_records(
    session: Session,
    location: str | None = None,
    from_ts: str | None = None,
    to_ts: str | None = None,
) -> list[TrafficRecord]:
    query = session.query(TrafficRecord)

    if location is not None:
        query = query.filter(TrafficRecord.location == location)

    if from_ts is not None:
        from_dt = datetime.fromisoformat(from_ts)
        query = query.filter(TrafficRecord.timestamp >= from_dt)

    if to_ts is not None:
        to_dt = datetime.fromisoformat(to_ts)
        query = query.filter(TrafficRecord.timestamp <= to_dt)

    return query.all()
