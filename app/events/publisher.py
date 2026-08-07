"""Event publishing via Redis Streams.

This module publishes domain events so other platform repos can react asynchronously without this repo knowing about them. Events are published to named streams (one per event type); consumers (future repos) listen to streams they care about and pull at their own pace.

Example event from this phase:
  Stream: "equipment.failure_risk"
  Message: {"equipment_id": "eq-1", "probability": 0.92, "model_version": "v..."}

This is a thin wrapper around Redis Streams — the producer doesn't need to know if anyone's listening; the consumer doesn't need to know about this repo's internals. Perfect for loosely-coupled platform integration.
"""

import json
from dataclasses import asdict, dataclass

import redis
from redis.exceptions import ConnectionError

from app.config import settings


class EventPublishError(RuntimeError):
    """Raised when an event fails to publish (Redis unavailable, etc.)."""


@dataclass(frozen=True)
class EquipmentFailureRiskEvent:
    """A work order was created (or an URGENT escalation was approved)
    due to high predicted failure risk.
    """

    equipment_id: str
    probability: float
    model_version: str
    work_order_id: str | None = None  # None if risk didn't cross threshold
    schema_version: int = 1


@dataclass(frozen=True)
class WorkOrderApprovedEvent:
    """An URGENT-priority work order was escalated from ELEVATED to URGENT by a human approval.
    """

    work_order_id: str
    equipment_id: str
    approved_by: str
    schema_version: int = 1


def _get_redis_client() -> redis.Redis:
    """Lazy-load a single Redis connection per module."""
    return redis.from_url(settings.redis_url, decode_responses=True)


def publish_equipment_failure_risk(event: EquipmentFailureRiskEvent) -> None:
    """Publishes an EquipmentFailureRiskEvent to the 'equipment.failure_risk' stream. Raises EventPublishError if Redis is unavailable.
    """
    try:
        client = _get_redis_client()
        client.xadd(
            "equipment.failure_risk",
            asdict(event),
            maxlen=10000,  # keep the stream bounded to recent events
            approximate=True,
        )
    except ConnectionError as e:
        raise EventPublishError(f"Failed to publish failure_risk event: Redis unavailable") from e
    except Exception as e:
        raise EventPublishError(f"Failed to publish failure_risk event: {e}") from e


def publish_work_order_approved(event: WorkOrderApprovedEvent) -> None:
    """Publishes a WorkOrderApprovedEvent to the 'work_order.approved' stream. Raised EventPublishError if Redis is unavailable.
    """
    try:
        client = _get_redis_client()
        client.xadd(
            "work_order.approved",
            asdict(event),
            maxlen=5000,
            approximate=True,
        )
    except ConnectionError as e:
        raise EventPublishError(f"Failed to publish work_order_approved event: Redis unavailable") from e
    except Exception as e:
        raise EventPublishError(f"Failed to publish work_order_approved event: {e}") from e
