"""Work order endpoints — currently just the urgent-priority approval
action (FR4). This is the ONLY way an urgent recommendation becomes an
actual URGENT-priority work order; see app.services.work_order_service.
"""

from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.services.work_order_service import approve_urgent_priority

router = APIRouter(tags=["work_orders"])


class ApproveRequest(BaseModel):
    approved_by: str


class WorkOrderApprovalResponse(BaseModel):
    id: str
    priority: str
    recommended_priority: str | None
    status: str
    priority_approved_at: datetime | None
    priority_approved_by: str | None


@router.post("/work-orders/{work_order_id}/approve", response_model=WorkOrderApprovalResponse)
def approve_work_order(
    work_order_id: str, request: ApproveRequest, db: Session = Depends(get_db)
) -> WorkOrderApprovalResponse:
    """Escalates a work order's priority from ELEVATED to URGENT.

    Raises (mapped to HTTP by app.api.error_handlers):
    - WorkOrderNotFoundError -> 404
    - InvalidApprovalError -> 409 (no pending urgent recommendation, or
      already approved)
    """
    work_order = approve_urgent_priority(db, work_order_id, approved_by=request.approved_by)

    return WorkOrderApprovalResponse(
        id=work_order.id,
        priority=work_order.priority.value,
        recommended_priority=(
            work_order.recommended_priority.value if work_order.recommended_priority else None
        ),
        status=work_order.status.value,
        priority_approved_at=work_order.priority_approved_at,
        priority_approved_by=work_order.priority_approved_by,
    )
