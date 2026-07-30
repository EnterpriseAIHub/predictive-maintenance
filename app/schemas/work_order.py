"""WorkOrder schema.

Unlike `Equipment`, `WorkOrder` is owned by this repo — other repos
may read work orders this service creates, but only this service
writes them (platform bounded-context ownership rule). This will
still be published through `platform-data-contracts` once that
package exists, so other repos can depend on the shape without
importing this repo's code directly.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class WorkOrderStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    CLOSED = "closed"


class WorkOrderPriority(StrEnum):
    ROUTINE = "routine"
    ELEVATED = "elevated"
    URGENT = "urgent"


class WorkOrder(BaseModel):
    id: str
    equipment_id: str
    opened_by: str  # "system" | "human"
    priority: WorkOrderPriority
    status: WorkOrderStatus
    created_at: datetime
