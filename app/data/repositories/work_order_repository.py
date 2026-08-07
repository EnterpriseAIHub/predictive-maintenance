from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data.models.work_order import WorkOrder, WorkOrderStatus


def get_by_id(db: Session, work_order_id: str) -> WorkOrder | None:
    return db.get(WorkOrder, work_order_id)


def create(db: Session, work_order: WorkOrder) -> WorkOrder:
    """Adds and flushes (not commits) — see sensor_reading_repository
    for why the transaction boundary belongs to the caller."""
    db.add(work_order)
    db.flush()
    db.refresh(work_order)
    return work_order


def get_open_for_equipment(db: Session, equipment_id: str) -> list[WorkOrder]:
    stmt = select(WorkOrder).where(
        WorkOrder.equipment_id == equipment_id,
        WorkOrder.status == WorkOrderStatus.OPEN,
    )
    return list(db.scalars(stmt))
