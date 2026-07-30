"""Equipment table.

Note: Equipment is a platform-owned reference entity (see
app/schemas/equipment.py). This repo only reads it — it never
creates or updates equipment rows itself — but it still needs the
table locally since platform-data-contracts/a shared reference
service don't exist yet (same standalone-runnability reason as the
Pydantic schema).
"""

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

    sensor_readings: Mapped[list["SensorReading"]] = relationship(back_populates="equipment")
    work_orders: Mapped[list["WorkOrder"]] = relationship(back_populates="equipment")
    risk_scores: Mapped[list["RiskScore"]] = relationship(back_populates="equipment")
