from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data.models.risk_score import RiskScore


def create(db: Session, risk_score: RiskScore) -> RiskScore:
    """Adds and flushes (not commits) — see sensor_reading_repository for why the transaction boundary belongs to the caller."""
    db.add(risk_score)
    db.flush()
    db.refresh(risk_score)
    return risk_score


def get_latest(db: Session, equipment_id: str) -> RiskScore | None:
    stmt = (
        select(RiskScore)
        .where(RiskScore.equipment_id == equipment_id)
        .order_by(RiskScore.created_at.desc())
        .limit(1)
    )
    return db.scalars(stmt).first()
