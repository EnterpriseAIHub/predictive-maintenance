from datetime import UTC, datetime

from app.data.models.equipment import Equipment
from app.data.repositories import equipment_repository


def _make_equipment(id_: str = "eq-1") -> Equipment:
    return Equipment(
        id=id_,
        plant_id="plant-1",
        type="conveyor_motor",
        install_date=datetime(2022, 1, 1, tzinfo=UTC),
        criticality_tier=2,
    )


def test_get_by_id_returns_none_when_missing(db):
    assert equipment_repository.get_by_id(db, "does-not-exist") is None


def test_get_by_id_returns_seeded_row(db):
    db.add(_make_equipment())
    db.flush()

    found = equipment_repository.get_by_id(db, "eq-1")

    assert found is not None
    assert found.plant_id == "plant-1"


def test_list_all_returns_every_row(db):
    db.add(_make_equipment("eq-1"))
    db.add(_make_equipment("eq-2"))
    db.flush()

    assert {e.id for e in equipment_repository.list_all(db)} >= {"eq-1", "eq-2"}
