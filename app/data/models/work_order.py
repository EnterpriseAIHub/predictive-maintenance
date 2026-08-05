"""Work orders created by this repo (owned here — see NFR6:
boundedness).

`priority` is the currently-effective priority. `recommended_priority`
is what the model recommended, which can differ from `priority`: when
the model recommends URGENT, the system holds `priority` at ELEVATED
until a human explicitly approves the escalation (FR4) — see
app/services/work_order_service.py. The three approval columns are
null until that approval happens.
"""

from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.data.base import Base
from app.schemas.work_order import WorkOrderPriority, WorkOrderStatus


class WorkOrder(Base):
    __tablename__ = "work_order"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    equipment_id: Mapped[str] = mapped_column(ForeignKey("equipment.id"), nullable=False, index=True)
    opened_by: Mapped[str] = mapped_column(String, nullable=False)  # "system" | "human"
    priority: Mapped[WorkOrderPriority] = mapped_column(
        Enum(WorkOrderPriority, name="work_order_priority"), nullable=False
    )
    recommended_priority: Mapped[WorkOrderPriority | None] = mapped_column(
        Enum(WorkOrderPriority, name="work_order_priority"), nullable=True
    )
    status: Mapped[WorkOrderStatus] = mapped_column(
        Enum(WorkOrderStatus, name="work_order_status"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    priority_approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    priority_approved_by: Mapped[str | None] = mapped_column(String, nullable=True)

    equipment: Mapped["Equipment"] = relationship(back_populates="work_orders")