from datetime import UTC, datetime, timedelta

from app.data.repositories import equipment_repository, sensor_reading_repository
from app.scripts.seed_demo_data import _DEMO_ASSETS, seed_demo_data


def test_seed_creates_every_demo_asset(db, monkeypatch):
    # Point the script's own session factory at this test's
    # savepoint-isolated session so it doesn't touch the real database.
    import app.scripts.seed_demo_data as seed_module

    monkeypatch.setattr(seed_module, "SessionLocal", lambda: db)

    # The seed script closes its session at the end.
    # Prevent it from closing the shared test session.
    monkeypatch.setattr(db, "close", lambda: None)

    seed_demo_data()

    for asset in _DEMO_ASSETS:
        assert equipment_repository.get_by_id(db, asset["id"]) is not None


def test_seed_is_idempotent(db, monkeypatch):
    import app.scripts.seed_demo_data as seed_module

    monkeypatch.setattr(seed_module, "SessionLocal", lambda: db)
    monkeypatch.setattr(db, "close", lambda: None)

    seed_demo_data()
    seed_demo_data()

    for asset in _DEMO_ASSETS:
        assert equipment_repository.get_by_id(db, asset["id"]) is not None


def test_the_no_history_asset_genuinely_has_zero_readings(db, monkeypatch):
    import app.scripts.seed_demo_data as seed_module

    monkeypatch.setattr(seed_module, "SessionLocal", lambda: db)
    monkeypatch.setattr(db, "close", lambda: None)

    seed_demo_data()

    readings = sensor_reading_repository.get_recent(
        db,
        "demo-eq-new",
        since=datetime.now(UTC) - timedelta(hours=168),
    )

    assert readings == []


def test_the_degrading_asset_has_sensor_history(db, monkeypatch):
    import app.scripts.seed_demo_data as seed_module

    monkeypatch.setattr(seed_module, "SessionLocal", lambda: db)
    monkeypatch.setattr(db, "close", lambda: None)

    seed_demo_data()

    # The degrading demo asset was intentionally renamed from
    # demo-eq-degrading to the canonical Project 2 equipment ID P-101.
    readings = sensor_reading_repository.get_recent(
        db,
        "P-101",
        since=datetime.now(UTC) - timedelta(hours=168),
    )

    assert len(readings) > 0