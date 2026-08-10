from app.events.publisher import (
    EquipmentFailureRiskEvent,
    WorkOrderApprovedEvent,
)


def test_equipment_failure_risk_event_constructs():
    event = EquipmentFailureRiskEvent(
        equipment_id="eq-1", probability=0.92, model_version="v1", work_order_id="wo-1"
    )
    assert event.equipment_id == "eq-1"
    assert event.probability == 0.92
    assert event.work_order_id == "wo-1"


def test_equipment_failure_risk_event_work_order_id_optional():
    event = EquipmentFailureRiskEvent(equipment_id="eq-1", probability=0.4, model_version="v1")
    assert event.work_order_id is None


def test_work_order_approved_event_constructs():
    event = WorkOrderApprovedEvent(
        work_order_id="wo-1", equipment_id="eq-1", approved_by="tech_alice"
    )
    assert event.work_order_id == "wo-1"
    assert event.approved_by == "tech_alice"


def test_events_are_frozen():
    event = EquipmentFailureRiskEvent(equipment_id="eq-1", probability=0.9, model_version="v1")
    try:
        event.probability = 0.5
        assert False, "should not allow mutation"
    except (AttributeError, TypeError):
        pass
