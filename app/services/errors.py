"""Service-layer exceptions.

Kept distinct from app.ml.inference.ModelNotFoundError — that one is an ML-layer concern (no model artifact available); these are business concerns the API layer (Phase 8) will map to specific HTTP status codes.
"""


class EquipmentNotFoundError(Exception):
    """Raised when a prediction is requested for an equipment_id that
    doesn't exist."""


class WorkOrderNotFoundError(Exception):
    """Raised when an approval action targets a work_order_id that
    doesn't exist."""


class InvalidApprovalError(Exception):
    """Raised when an approval action doesn't apply to the targeted
    work order's current state (no pending urgent escalation, or
    already approved)."""