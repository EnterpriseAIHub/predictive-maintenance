from datetime import UTC, datetime, timedelta

from app.data.models.equipment import Equipment
from app.data.models.sensor_reading import SensorReading
from app.data.repositories import sensor_reading_repository


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


def test_create_and_get_recent(db):
    _seed_equipment(db)
    now = datetime.now(UTC)

    old_reading = SensorReading(
        equipment_id="eq-1", timestamp=now - timedelta(days=30), sensor_type="temperature", value=70.0
    )
    recent_reading = SensorReading(
        equipment_id="eq-1", timestamp=now - timedelta(hours=1), sensor_type="temperature", value=75.0
    )
    sensor_reading_repository.create(db, old_reading)
    sensor_reading_repository.create(db, recent_reading)

    results = sensor_reading_repository.get_recent(db, "eq-1", since=now - timedelta(days=1))

    assert [r.value for r in results] == [75.0]  # old reading excluded, ordered oldest-first
