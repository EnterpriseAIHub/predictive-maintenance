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
