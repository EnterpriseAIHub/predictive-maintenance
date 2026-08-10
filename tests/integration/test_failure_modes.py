"""Failure-mode tests: verifying the documented behavior for each known
failure condition (model unavailable, unexpected/unhandled errors,
Redis unavailable during event publish) rather than just the happy path.
"""

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.api.deps import get_db
from app.data.models.equipment import Equipment
from app.data.models.sensor_reading import SensorReading
from app.main import app
from app.ml.explain import FeatureAttribution
from app.ml.inference import ModelNotFoundError, PredictionResult
from app.services import prediction_service, work_order_service

AS_OF = "2026-01-08T00:00:00Z"


class _FakeModel:
    def __init__(self, probability: float):
        self.probability = probability
        self.version = "fake-v1"

    def predict_with_explanation(self, feature_row, top_n=3):
        return PredictionResult(
            probability=self.probability,
            model_version=self.version,
            attributions=[FeatureAttribution("temperature_rolling_mean", 0.4, 80.0)],
        )


def _seed_equipment_and_readings(db, id_: str = "eq-1") -> None:
    db.add(
        Equipment(
            id=id_,
            plant_id="plant-1",
            type="conveyor_motor",
            install_date=datetime(2022, 1, 1, tzinfo=UTC),
            criticality_tier=2,
        )
    )
    db.add(
        SensorReading(
            equipment_id=id_,
            timestamp=datetime(2026, 1, 7, tzinfo=UTC),
            sensor_type="temperature",
            value=80.0,
        )
    )
    db.flush()


def test_predict_returns_503_when_no_model_is_registered(client, db, monkeypatch):
    _seed_equipment_and_readings(db)

    def _raise():
        raise ModelNotFoundError("no model registered")

    monkeypatch.setattr(prediction_service, "get_model", _raise)

    response = client.post("/predict", json={"equipment_id": "eq-1", "as_of": AS_OF})

    assert response.status_code == 503
    assert "no model registered" in response.json()["detail"]


def test_redis_unavailable_does_not_fail_the_prediction_request(client, db, monkeypatch):
    """The core Phase 7/10 contract: a work order still gets created and
    the API still returns 200 even if the event bus is completely down.
    """
    _seed_equipment_and_readings(db)
    monkeypatch.setattr(prediction_service, "get_model", lambda: _FakeModel(0.95))

    def _broken_publish(event):
        raise ConnectionError("Redis is down")

    monkeypatch.setattr(prediction_service, "publish_equipment_failure_risk", _broken_publish)

    response = client.post("/predict", json={"equipment_id": "eq-1", "as_of": AS_OF})

    assert response.status_code == 200
    body = response.json()
    assert body["work_order"] is not None
    assert body["work_order"]["recommended_priority"] == "urgent"


def test_redis_unavailable_does_not_fail_approval(db, monkeypatch):
    _seed_equipment_and_readings(db)
    wo = work_order_service.create_work_order_for_prediction(
        db, "eq-1", 0.95, datetime(2026, 1, 8, tzinfo=UTC)
    )
    db.commit()

    def _broken_publish(event):
        raise ConnectionError("Redis is down")

    monkeypatch.setattr(work_order_service, "publish_work_order_approved", _broken_publish)

    approved = work_order_service.approve_urgent_priority(db, wo.id, approved_by="tech_alice")

    assert approved.priority.value == "urgent"


def test_unhandled_exception_returns_generic_500_without_leaking_details(db, monkeypatch):
    """A truly unexpected error (simulated here as a bug in feature
    building) must not leak internal exception text to the client — the
    real error is logged server-side, but the response is generic.
    """
    _seed_equipment_and_readings(db)

    def _boom(*args, **kwargs):
        raise RuntimeError("SECRET INTERNAL DETAIL: connection string leaked here")

    monkeypatch.setattr(prediction_service, "build_feature_vector", _boom)

    def _override_get_db():
        yield db

    app.dependency_overrides[get_db] = _override_get_db
    try:
        # raise_server_exceptions=False is required here specifically —
        # the default TestClient re-raises unhandled exceptions instead
        # of exercising the registered 500 handler, which would defeat
        # the point of this test.
        unsafe_client = TestClient(app, raise_server_exceptions=False)
        response = unsafe_client.post("/predict", json={"equipment_id": "eq-1", "as_of": AS_OF})
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 500
    assert "SECRET INTERNAL DETAIL" not in response.text
    assert response.json() == {"detail": "An internal error occurred."}
