"""Nightly batch scoring.

Scores every known equipment asset once, reusing
app.services.prediction_service.run_prediction_for_equipment — the exact same code path the real-time API uses. This is the concrete payoff of Phase 6's design: there is no separate "batch prediction logic" to keep in sync with the real-time path, because there isn't a second implementation at all.

Run directly: `python -m app.batch.nightly_job`

Scheduling (an actual cron entry, a platform scheduler, Render's
scheduled jobs, etc.) is a deployment concern, not something this
script manages itself — kept simple deliberately: a plain,
cron-callable script demonstrates the pattern without adding a
scheduler dependency this portfolio project doesn't need to carry.
"""

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.data.base import SessionLocal
from app.data.models.risk_score import RiskScoreSource
from app.data.repositories import equipment_repository
from app.logging_config import configure_logging, get_logger
from app.services.prediction_service import PredictionOutcome, run_prediction_for_equipment

logger = get_logger(__name__)


def run_nightly_scoring(db: Session, as_of: datetime | None = None) -> list[PredictionOutcome]:
    """Scores every equipment asset. One asset's failure (e.g. a
    transient error) does not abort the run for every other asset — each asset is scored independently, and a failure is logged and skipped rather than raised.
    """
    as_of = as_of or datetime.now(UTC)
    outcomes: list[PredictionOutcome] = []

    for equipment in equipment_repository.list_all(db):
        try:
            outcome = run_prediction_for_equipment(
                db, equipment.id, as_of=as_of, source=RiskScoreSource.BATCH
            )
            outcomes.append(outcome)
        except Exception:
            # Roll back so a failed asset's partial work never bleeds
            # into the next asset's transaction (run_prediction_for_
            # equipment normally commits its own work internally, but
            # if it raised before reaching that commit, whatever it
            # flushed is still pending on this shared session).
            db.rollback()
            logger.warning("batch_scoring_failed_for_equipment", equipment_id=equipment.id, exc_info=True)

    logger.info(
        "batch_scoring_complete",
        total_equipment=len(outcomes),
        work_orders_created=sum(1 for o in outcomes if o.work_order is not None),
    )
    return outcomes


if __name__ == "__main__":
    configure_logging()
    session = SessionLocal()
    try:
        run_nightly_scoring(session)
    finally:
        session.close()
