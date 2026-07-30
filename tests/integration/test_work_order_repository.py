from datetime import UTC, datetime

from app.data.models.equipment import Equipment
from app.data.models.work_order import WorkOrder, WorkOrderPriority, WorkOrderStatus
from app.data.repositories import work_order_repository


def _seed_equipment(db, id_: str = "eq-1") -> None:
    db.add(
        Equipment(
            id=id_,
            plant_id="plant-1",
            type="conveyor_motor",
            install_date=datetime(2022, 1, 1, tzinfo=UTC),
            criticality_tier=2,
        )
    )
    db.flush()


def test_create_and_get_open_for_equipment(db):
    _seed_equipment(db)

    open_wo = WorkOrder(
        id="wo-1",
        equipment_id="eq-1",
        opened_by="system",
        priority=WorkOrderPriority.ELEVATED,
        status=WorkOrderStatus.OPEN,
        created_at=datetime.now(UTC),
    )
    closed_wo = WorkOrder(
        id="wo-2",
        equipment_id="eq-1",
        opened_by="system",
        priority=WorkOrderPriority.ROUTINE,
        status=WorkOrderStatus.CLOSED,
        created_at=datetime.now(UTC),
    )
    work_order_repository.create(db, open_wo)
    work_order_repository.create(db, closed_wo)

    results = work_order_repository.get_open_for_equipment(db, "eq-1")

    assert [wo.id for wo in results] == ["wo-1"]
