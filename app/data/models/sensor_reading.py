"""Raw sensor readings — the input feature engineering will read from
in a later milestone. Deliberately just storage here: no aggregation,
no rolling windows, no derived values.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.data.base import Base


class SensorReading(Base):
    __tablename__ = "sensor_reading"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    equipment_id: Mapped[str] = mapped_column(
        ForeignKey("equipment.id"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    sensor_type: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    # noqa: F821 below — SQLAlchemy resolves this string forward-ref at runtime
    equipment: Mapped["Equipment"] = relationship(back_populates="sensor_readings")  # noqa: F821
