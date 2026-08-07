from datetime import UTC, datetime, timedelta

from app.data.models.equipment import Equipment
from app.data.models.sensor_reading import SensorReading
from app.ml.explain import FeatureAttribution
from app.ml.inference import PredictionResult
from app.services import prediction_service

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


def test_predict_returns_404_for_unknown_equipment(client):
    response = client.post("/predict", json={"equipment_id": "does-not-exist"})
    assert response.status_code == 404


def test_predict_below_threshold_returns_no_work_order(client, db, monkeypatch):
    _seed_equipment_and_readings(db)
    monkeypatch.setattr(prediction_service, "get_model", lambda: _FakeModel(0.2))

    response = client.post("/predict", json={"equipment_id": "eq-1", "as_of": AS_OF})

    assert response.status_code == 200
    body = response.json()
    assert body["probability"] == 0.2
    assert body["work_order"] is None
    assert len(body["attributions"]) == 1


def test_predict_urgent_creates_a_held_back_work_order(client, db, monkeypatch):
    _seed_equipment_and_readings(db)
    monkeypatch.setattr(prediction_service, "get_model", lambda: _FakeModel(0.95))

    response = client.post("/predict", json={"equipment_id": "eq-1", "as_of": AS_OF})

    assert response.status_code == 200
    body = response.json()
    assert body["work_order"]["priority"] == "elevated"
    assert body["work_order"]["recommended_priority"] == "urgent"


def test_predict_works_for_equipment_with_zero_sensor_readings(client, db, monkeypatch):
    """Regression test: a newly onboarded asset with no sensor history
    yet must not crash prediction — it should still return a result
    built from NaN-mean/zero-trend features (see app.ml.features).
    """
    db.add(
        Equipment(
            id="eq-brand-new",
            plant_id="plant-1",
            type="conveyor_motor",
            install_date=datetime(2026, 1, 1, tzinfo=UTC),
            criticality_tier=1,
        )
    )
    db.flush()
    monkeypatch.setattr(prediction_service, "get_model", lambda: _FakeModel(0.1))

    response = client.post("/predict", json={"equipment_id": "eq-brand-new", "as_of": AS_OF})

    assert response.status_code == 200
    assert response.json()["probability"] == 0.1


def test_get_latest_risk_returns_404_before_any_prediction(client, db):
    _seed_equipment_and_readings(db)
    response = client.get("/equipment/eq-1/risk")
    assert response.status_code == 404


def test_get_latest_risk_returns_the_persisted_score(client, db, monkeypatch):
    _seed_equipment_and_readings(db)
    monkeypatch.setattr(prediction_service, "get_model", lambda: _FakeModel(0.6))
    client.post("/predict", json={"equipment_id": "eq-1", "as_of": AS_OF})

    response = client.get("/equipment/eq-1/risk")

    assert response.status_code == 200
    assert response.json()["probability"] == 0.6
    assert response.json()["source"] == "real_time"
