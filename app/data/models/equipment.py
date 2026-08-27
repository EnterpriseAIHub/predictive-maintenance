from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.data.base import Base


class Equipment(Base):
    __tablename__ = "equipment"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    plant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    type: Mapped[str] = mapped_column(String, nullable=False)
    install_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    criticality_tier: Mapped[int] = mapped_column(Integer, nullable=False)

    # F821 on the three lines below is a false positive: these are
    # SQLAlchemy string forward-refs, resolved at runtime, not real
    # undefined names.
    sensor_readings: Mapped[list["SensorReading"]] = relationship(back_populates="equipment")  # noqa: F821
    work_orders: Mapped[list["WorkOrder"]] = relationship(back_populates="equipment")  # noqa: F821
    risk_scores: Mapped[list["RiskScore"]] = relationship(back_populates="equipment")  # noqa: F821
