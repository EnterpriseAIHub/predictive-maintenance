from datetime import UTC, datetime

from app.batch import nightly_job
from app.data.models.equipment import Equipment
from app.data.models.risk_score import RiskScoreSource
from app.ml.explain import FeatureAttribution
from app.ml.inference import PredictionResult
from app.services import prediction_service

AS_OF = datetime(2026, 1, 8, tzinfo=UTC)


class _FakeModel:
    def __init__(self, probability: float):
        self.probability = probability
        self.version = "fake-v1"

    def predict_with_explanation(self, feature_row, top_n=3):
        return PredictionResult(
            probability=self.probability,
            model_version=self.version,
            attributions=[FeatureAttribution("temperature_rolling_mean", 0.3, 75.0)],
        )


def _seed_equipment(db, id_: str) -> None:
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


def test_scores_every_equipment_asset(db, monkeypatch):
    _seed_equipment(db, "eq-1")
    _seed_equipment(db, "eq-2")
    monkeypatch.setattr(prediction_service, "get_model", lambda: _FakeModel(0.3))

    outcomes = nightly_job.run_nightly_scoring(db, as_of=AS_OF)

    assert {o.equipment_id for o in outcomes} == {"eq-1", "eq-2"}


def test_batch_predictions_are_tagged_with_batch_source(db, monkeypatch):
    from app.data.repositories import risk_score_repository

    _seed_equipment(db, "eq-1")
    monkeypatch.setattr(prediction_service, "get_model", lambda: _FakeModel(0.4))

    nightly_job.run_nightly_scoring(db, as_of=AS_OF)

    latest = risk_score_repository.get_latest(db, "eq-1")
    assert latest.source == RiskScoreSource.BATCH


def test_one_asset_failure_does_not_abort_the_whole_run(db, monkeypatch):
    _seed_equipment(db, "eq-bad")
    _seed_equipment(db, "eq-good")
    db.commit()  # equipment is committed reference data in reality, not pending mid-transaction —
    # required here so run_nightly_scoring's per-asset db.rollback() (which rolls back to the
    # last savepoint) can't undo the seed data itself, only a failed asset's own partial work.

    # equipment_repository.list_all orders by id, so "eq-bad" is
    # processed first — fail only that first call, succeed on the rest.
    calls = {"n": 0}

    def flaky_get_model():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated model load failure for the first asset")
        return _FakeModel(0.5)

    monkeypatch.setattr(prediction_service, "get_model", flaky_get_model)

    outcomes = nightly_job.run_nightly_scoring(db, as_of=AS_OF)

    assert len(outcomes) == 1
    assert outcomes[0].equipment_id == "eq-good"


def test_empty_equipment_table_returns_empty_list(db):
    outcomes = nightly_job.run_nightly_scoring(db, as_of=AS_OF)
    assert outcomes == []
