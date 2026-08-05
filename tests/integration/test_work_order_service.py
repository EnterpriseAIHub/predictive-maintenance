from datetime import UTC, datetime

import pytest

from app.data.models.equipment import Equipment
from app.schemas.work_order import WorkOrderPriority, WorkOrderStatus
from app.services import work_order_service
from app.services.errors import InvalidApprovalError, WorkOrderNotFoundError

AS_OF = datetime(2026, 1, 8, tzinfo=UTC)


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


def test_probability_below_threshold_creates_no_work_order(db):
    _seed_equipment(db)
    result = work_order_service.create_work_order_for_prediction(db, "eq-1", 0.3, AS_OF)
    assert result is None


def test_elevated_probability_creates_an_elevated_work_order_directly(db):
    _seed_equipment(db)
    wo = work_order_service.create_work_order_for_prediction(db, "eq-1", 0.75, AS_OF)

    assert wo is not None
    assert wo.priority == WorkOrderPriority.ELEVATED
    assert wo.recommended_priority == WorkOrderPriority.ELEVATED
    assert wo.status == WorkOrderStatus.OPEN
    assert wo.priority_approved_at is None


def test_urgent_probability_is_held_at_elevated_pending_approval(db):
    _seed_equipment(db)
    wo = work_order_service.create_work_order_for_prediction(db, "eq-1", 0.95, AS_OF)

    assert wo is not None
    assert wo.priority == WorkOrderPriority.ELEVATED  # held back, NOT urgent
    assert wo.recommended_priority == WorkOrderPriority.URGENT  # but recorded as recommended
    assert wo.priority_approved_at is None
    assert wo.priority_approved_by is None


def test_approve_urgent_priority_escalates_a_pending_recommendation(db):
    _seed_equipment(db)
    wo = work_order_service.create_work_order_for_prediction(db, "eq-1", 0.95, AS_OF)
    db.commit()

    approved = work_order_service.approve_urgent_priority(db, wo.id, approved_by="tech_alice")

    assert approved.priority == WorkOrderPriority.URGENT
    assert approved.priority_approved_by == "tech_alice"
    assert approved.priority_approved_at is not None


def test_approve_urgent_priority_rejects_a_work_order_with_no_pending_urgent(db):
    _seed_equipment(db)
    wo = work_order_service.create_work_order_for_prediction(db, "eq-1", 0.75, AS_OF)  # elevated, not urgent
    db.commit()

    with pytest.raises(InvalidApprovalError):
        work_order_service.approve_urgent_priority(db, wo.id, approved_by="tech_alice")


def test_approve_urgent_priority_rejects_double_approval(db):
    _seed_equipment(db)
    wo = work_order_service.create_work_order_for_prediction(db, "eq-1", 0.95, AS_OF)
    db.commit()
    work_order_service.approve_urgent_priority(db, wo.id, approved_by="tech_alice")

    with pytest.raises(InvalidApprovalError):
        work_order_service.approve_urgent_priority(db, wo.id, approved_by="tech_bob")


def test_approve_urgent_priority_raises_for_unknown_work_order(db):
    with pytest.raises(WorkOrderNotFoundError):
        work_order_service.approve_urgent_priority(db, "does-not-exist", approved_by="tech_alice")