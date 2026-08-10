from datetime import UTC, datetime

from app.data.models.equipment import Equipment
from app.services import work_order_service

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


def test_approve_returns_404_for_unknown_work_order(client):
    response = client.post(
        "/work-orders/does-not-exist/approve", json={"approved_by": "tech_alice"}
    )
    assert response.status_code == 404


def test_approve_returns_409_when_no_pending_urgent_recommendation(client, db):
    _seed_equipment(db)
    wo = work_order_service.create_work_order_for_prediction(db, "eq-1", 0.75, AS_OF)  # elevated
    db.commit()

    response = client.post(f"/work-orders/{wo.id}/approve", json={"approved_by": "tech_alice"})
    assert response.status_code == 409


def test_approve_escalates_a_pending_urgent_recommendation(client, db):
    _seed_equipment(db)
    wo = work_order_service.create_work_order_for_prediction(db, "eq-1", 0.95, AS_OF)
    db.commit()

    response = client.post(f"/work-orders/{wo.id}/approve", json={"approved_by": "tech_alice"})

    assert response.status_code == 200
    body = response.json()
    assert body["priority"] == "urgent"
    assert body["priority_approved_by"] == "tech_alice"
