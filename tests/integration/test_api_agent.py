from datetime import UTC, datetime

from app.data.models.equipment import Equipment
from app.data.models.sensor_reading import SensorReading
from app.ml.explain import FeatureAttribution
from app.ml.inference import PredictionResult
from app.services import prediction_service


class _FakeModel:
    def __init__(self, probability: float):
        self.probability = probability
        self.version = "fake-v1"

    def predict_with_explanation(self, feature_row, top_n=3):
        return PredictionResult(
            probability=self.probability,
            model_version=self.version,
            attributions=[FeatureAttribution("vibration_rate_of_change", 0.25, 0.9)],
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
            sensor_type="vibration",
            value=0.9,
        )
    )
    db.flush()


def test_agent_requires_equipment_id_in_context(client):
    response = client.post("/agent", json={"query": "how risky is this asset?", "context": {}})
    assert response.status_code == 422


def test_agent_returns_answer_confidence_and_provenance(client, db, monkeypatch):
    _seed_equipment_and_readings(db)
    monkeypatch.setattr(prediction_service, "get_model", lambda: _FakeModel(0.88))

    response = client.post(
        "/agent",
        json={
            "query": "how risky is eq-1?",
            "context": {"equipment_id": "eq-1", "as_of": "2026-01-08T00:00:00+00:00"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "eq-1" in body["answer"]
    assert body["confidence"] == 0.88
    assert len(body["provenance"]) == 1
    assert body["structured_data"]["equipment_id"] == "eq-1"
