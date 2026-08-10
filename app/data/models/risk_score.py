"""Every prediction is persisted here, regardless of whether it
crosses the action threshold (EDD §6/§18) — this history is what the
future model-drift check and the orchestrator's daily briefing both
read from.
"""

from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.data.base import Base


class RiskScoreSource(StrEnum):
    REAL_TIME = "real_time"
    BATCH = "batch"


class RiskScore(Base):
    __tablename__ = "risk_score"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    equipment_id: Mapped[str] = mapped_column(
        ForeignKey("equipment.id"), nullable=False, index=True
    )
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[RiskScoreSource] = mapped_column(
        Enum(RiskScoreSource, name="risk_score_source"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # F821 below is a false positive — SQLAlchemy runtime forward-ref
    equipment: Mapped["Equipment"] = relationship(back_populates="risk_scores")  # noqa: F821
