"""Work orders created by this repo (owned here — see NFR6:
boundedness). Schema intentionally matches app/schemas/work_order.py
exactly; the priority-approval workflow (FR4 — urgent escalations
require human sign-off before persisting) is business logic, added in
the services milestone, not represented as extra columns here yet.
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
    status: Mapped[WorkOrderStatus] = mapped_column(
        Enum(WorkOrderStatus, name="work_order_status"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    equipment: Mapped["Equipment"] = relationship(back_populates="work_orders")
