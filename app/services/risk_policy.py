"""The risk-to-priority policy: the one place "how risky is risky
enough to act on" is decided. Deliberately a pure function — no DB, no
model, no I/O — so the business rule itself is unit-testable without
needing a database or a trained model, and so a future change to the
thresholds (a business decision) never requires touching orchestration
code.
"""

from app.config import settings
from app.schemas.work_order import WorkOrderPriority


def evaluate_risk(probability: float) -> WorkOrderPriority | None:
    """Returns the RECOMMENDED priority for this probability, or None
    if it doesn't warrant a work order at all.

    This is the recommendation only — whether URGENT actually gets
    persisted as URGENT (vs. held at ELEVATED pending human approval)
    is decided in work_order_service, not here (FR4's approval gate is
    an orchestration concern, not a risk-assessment one).
    """
    if probability >= settings.urgent_priority_threshold:
        return WorkOrderPriority.URGENT
    if probability >= settings.failure_risk_threshold:
        return WorkOrderPriority.ELEVATED
    return None