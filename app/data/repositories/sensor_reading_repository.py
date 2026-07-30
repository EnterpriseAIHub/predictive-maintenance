from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data.models.sensor_reading import SensorReading


def create(db: Session, reading: SensorReading) -> SensorReading:
    """Adds and flushes (not commits) — the caller decides the
    transaction boundary. A service-layer operation that writes a
    reading alongside other rows should be able to commit them
    together, atomically.
    """
    db.add(reading)
    db.flush()
    db.refresh(reading)
    return reading


def get_recent(db: Session, equipment_id: str, since: datetime) -> list[SensorReading]:
    """Readings for one asset from `since` onward, oldest first — the
    exact input shape the feature engineering module (later milestone)
    will consume for its rolling-window calculations.
    """
    stmt = (
        select(SensorReading)
        .where(SensorReading.equipment_id == equipment_id, SensorReading.timestamp >= since)
        .order_by(SensorReading.timestamp.asc())
    )
    return list(db.scalars(stmt))
