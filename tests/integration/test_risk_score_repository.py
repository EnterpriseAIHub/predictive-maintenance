from datetime import UTC, datetime, timedelta

from app.data.models.equipment import Equipment
from app.data.models.risk_score import RiskScore, RiskScoreSource
from app.data.repositories import risk_score_repository


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


def test_get_latest_returns_most_recent_score(db):
    _seed_equipment(db)
    now = datetime.now(UTC)

    older = RiskScore(
        equipment_id="eq-1",
        probability=0.4,
        model_version="v1",
        source=RiskScoreSource.BATCH,
        created_at=now - timedelta(days=1),
    )
    newer = RiskScore(
        equipment_id="eq-1",
        probability=0.82,
        model_version="v1",
        source=RiskScoreSource.REAL_TIME,
        created_at=now,
    )
    risk_score_repository.create(db, older)
    risk_score_repository.create(db, newer)

    latest = risk_score_repository.get_latest(db, "eq-1")

    assert latest is not None
    assert latest.probability == 0.82
