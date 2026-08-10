"""Work-order business logic: creating a work order from a risk
assessment, and the urgent-priority approval gate (FR4).

FR4, concretely: the system never persists `priority=URGENT` as a
direct result of a prediction. When the risk policy recommends
URGENT, the work order is created with `priority=ELEVATED` and
`recommended_priority=URGENT` — a human must call
`approve_urgent_priority` to actually escalate it. This function is
the ONLY code path that writes `priority=URGENT`; there is no
shortcut around it elsewhere in this service.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.data.models.work_order import WorkOrder
from app.data.repositories import work_order_repository
from app.events.publisher import WorkOrderApprovedEvent, publish_work_order_approved
from app.logging_config import get_logger
from app.schemas.work_order import WorkOrderPriority, WorkOrderStatus
from app.services import risk_policy
from app.services.errors import InvalidApprovalError, WorkOrderNotFoundError

logger = get_logger(__name__)


def create_work_order_for_prediction(
    db: Session, equipment_id: str, probability: float, as_of: datetime
) -> WorkOrder | None:
    """Creates (and flushes, not commits — see repository docstrings)
    a work order if the risk policy recommends one. Returns None if
    the probability doesn't cross the action threshold at all.
    """
    recommended = risk_policy.evaluate_risk(probability)
    if recommended is None:
        return None

    # Hold urgent recommendations at ELEVATED until a human approves —
    # this is the entire mechanism behind FR4.
    persisted_priority = (
        WorkOrderPriority.ELEVATED if recommended == WorkOrderPriority.URGENT else recommended
    )

    work_order = WorkOrder(
        id=str(uuid.uuid4()),
        equipment_id=equipment_id,
        opened_by="system",
        priority=persisted_priority,
        recommended_priority=recommended,
        status=WorkOrderStatus.OPEN,
        created_at=as_of,
    )
    work_order = work_order_repository.create(db, work_order)

    logger.info(
        "work_order_created",
        work_order_id=work_order.id,
        equipment_id=equipment_id,
        priority=persisted_priority.value,
        recommended_priority=recommended.value,
        held_pending_approval=recommended == WorkOrderPriority.URGENT,
    )
    return work_order


def approve_urgent_priority(db: Session, work_order_id: str, approved_by: str) -> WorkOrder:
    """The human action that fulfills FR4's approval requirement.
    Only valid when the work order has a pending URGENT recommendation
    that hasn't already been approved.
    """
    work_order = work_order_repository.get_by_id(db, work_order_id)
    if work_order is None:
        raise WorkOrderNotFoundError(f"No work order with id '{work_order_id}'.")

    if work_order.recommended_priority != WorkOrderPriority.URGENT:
        raise InvalidApprovalError(
            f"Work order '{work_order_id}' has no pending urgent-priority recommendation."
        )
    if work_order.priority_approved_at is not None:
        raise InvalidApprovalError(f"Work order '{work_order_id}' was already approved.")

    work_order.priority = WorkOrderPriority.URGENT
    work_order.priority_approved_at = datetime.now(UTC)
    work_order.priority_approved_by = approved_by

    db.commit()
    db.refresh(work_order)

    logger.info(
        "work_order_urgent_priority_approved",
        work_order_id=work_order.id,
        equipment_id=work_order.equipment_id,
        approved_by=approved_by,
    )

    # Publish async event. Same best-effort pattern as prediction_service:
    # the approval is durable, and if Redis is down, it's a transient issue
    # rather than a reason to fail an already-successful approval — but it
    # IS logged, since a silently dropped downstream notification is worth
    # being able to find later.
    try:
        publish_work_order_approved(
            WorkOrderApprovedEvent(
                work_order_id=work_order.id,
                equipment_id=work_order.equipment_id,
                approved_by=approved_by,
            )
        )
    except Exception as e:
        logger.warning(
            "event_publish_failed",
            event_type="work_order.approved",
            work_order_id=work_order.id,
            error=str(e),
        )

    return work_order
