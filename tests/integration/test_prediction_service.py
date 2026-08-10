from datetime import UTC, datetime, timedelta

import pytest

from app.data.models.equipment import Equipment
from app.data.models.sensor_reading import SensorReading
from app.ml.explain import FeatureAttribution
from app.ml.inference import PredictionResult
from app.schemas.work_order import WorkOrderPriority
from app.services import prediction_service
from app.services.errors import EquipmentNotFoundError

AS_OF = datetime(2026, 1, 8, tzinfo=UTC)


class _FakeModel:
    """Stand-in for a loaded model — the service layer's job (does it
    persist, does it branch correctly) is what this file tests, not
    ML correctness (already covered under tests/model/).
    """

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
            timestamp=AS_OF - timedelta(hours=1),
            sensor_type="temperature",
            value=80.0,
        )
    )
    db.flush()


def test_raises_for_unknown_equipment(db):
    with pytest.raises(EquipmentNotFoundError):
        prediction_service.run_prediction_for_equipment(db, "does-not-exist", as_of=AS_OF)


def test_persists_a_risk_score_even_when_no_work_order_is_created(db, monkeypatch):
    _seed_equipment_and_readings(db)
    monkeypatch.setattr(prediction_service, "get_model", lambda: _FakeModel(0.2))

    outcome = prediction_service.run_prediction_for_equipment(db, "eq-1", as_of=AS_OF)

    assert outcome.probability == 0.2
    assert outcome.work_order is None


def test_urgent_prediction_creates_a_held_back_work_order(db, monkeypatch):
    _seed_equipment_and_readings(db)
    monkeypatch.setattr(prediction_service, "get_model", lambda: _FakeModel(0.95))

    outcome = prediction_service.run_prediction_for_equipment(db, "eq-1", as_of=AS_OF)

    assert outcome.work_order is not None
    assert outcome.work_order.priority == WorkOrderPriority.ELEVATED
    assert outcome.work_order.recommended_priority == WorkOrderPriority.URGENT
    assert len(outcome.attributions) == 1
