"""Seeds a handful of demo equipment assets with realistic sensor
history, so a fresh deployment (or a fresh local environment) has
something to demonstrate through /predict rather than an empty
database.

Idempotent by design: each asset uses a fixed, well-known ID and is
skipped (not duplicated) if it already exists — safe to run on every
deploy, not just the first one.

Run directly: `python -m app.scripts.seed_demo_data`
"""

from datetime import UTC, datetime, timedelta

from app.data.base import SessionLocal
from app.data.models.equipment import Equipment
from app.data.models.sensor_reading import SensorReading
from app.data.repositories import equipment_repository
from app.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

_NOW = datetime.now(UTC)

# Four demo assets chosen to show the range of behavior the system
# handles: a clearly healthy asset, a clearly at-risk one, a
# borderline case, and one with NO sensor history at all (the
# zero-readings edge case fixed as Bug 5 in Phase 8 — worth
# demonstrating that it works correctly, not just that it doesn't
# crash).
_DEMO_ASSETS = [
    {
        "id": "demo-eq-healthy",
        "plant_id": "demo-plant",
        "type": "conveyor_motor",
        "install_date": _NOW - timedelta(days=400),
        "criticality_tier": 1,
        "profile": "healthy",
    },
    {
        "id": "demo-eq-degrading",
        "plant_id": "demo-plant",
        "type": "hydraulic_pump",
        "install_date": _NOW - timedelta(days=900),
        "criticality_tier": 3,
        "profile": "degrading",
    },
    {
        "id": "demo-eq-borderline",
        "plant_id": "demo-plant",
        "type": "conveyor_motor",
        "install_date": _NOW - timedelta(days=600),
        "criticality_tier": 2,
        "profile": "borderline",
    },
    {
        "id": "demo-eq-new",
        "plant_id": "demo-plant",
        "type": "hydraulic_pump",
        "install_date": _NOW - timedelta(days=2),
        "criticality_tier": 2,
        "profile": "no_history",  # zero sensor readings — a real, valid state
    },
]

_BASELINE = {"temperature": 70.0, "vibration": 0.5, "pressure": 100.0}
_PROFILE_DRIFT = {"healthy": 0.0, "borderline": 0.5, "degrading": 1.5, "no_history": 0.0}


def _generate_readings(equipment_id: str, profile: str) -> list[SensorReading]:
    if profile == "no_history":
        return []

    readings = []
    drift_scale = _PROFILE_DRIFT[profile]
    for hours_ago in range(0, 168, 12):  # one reading every 12h across the 7-day window
        timestamp = _NOW - timedelta(hours=hours_ago)
        # Drift grows as we approach "now" — same shape as the training
        # data generator, so a live /predict against this asset produces
        # a result consistent with what the model was trained to recognize.
        recency_factor = (168 - hours_ago) / 168
        for sensor_type, baseline in _BASELINE.items():
            value = baseline + (drift_scale * recency_factor * 10)
            readings.append(
                SensorReading(
                    equipment_id=equipment_id,
                    timestamp=timestamp,
                    sensor_type=sensor_type,
                    value=value,
                )
            )
    return readings


def seed_demo_data() -> None:
    session = SessionLocal()
    try:
        created = 0
        for asset in _DEMO_ASSETS:
            if equipment_repository.get_by_id(session, asset["id"]) is not None:
                logger.info("demo_asset_already_exists", equipment_id=asset["id"])
                continue

            session.add(
                Equipment(
                    id=asset["id"],
                    plant_id=asset["plant_id"],
                    type=asset["type"],
                    install_date=asset["install_date"],
                    criticality_tier=asset["criticality_tier"],
                )
            )
            for reading in _generate_readings(asset["id"], asset["profile"]):
                session.add(reading)

            created += 1
            logger.info("demo_asset_seeded", equipment_id=asset["id"], profile=asset["profile"])

        session.commit()
        logger.info(
            "seed_demo_data_complete", assets_created=created, assets_total=len(_DEMO_ASSETS)
        )
    finally:
        session.close()


if __name__ == "__main__":
    configure_logging()
    seed_demo_data()
