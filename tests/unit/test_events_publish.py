import pytest

from app.events.publisher import (
    EquipmentFailureRiskEvent,
    EventPublishError,
    WorkOrderApprovedEvent,
    publish_equipment_failure_risk,
    publish_work_order_approved,
)


def test_publish_equipment_failure_risk_to_redis(monkeypatch):
    """Test that publishing succeeds when Redis is available. Uses a mock
    to avoid requiring Redis to actually be running for every test run.
    """
    published_events = []

    def fake_xadd(stream, data, **kwargs):
        published_events.append((stream, data))
        return "1-0"  # fake stream ID

    def fake_redis(*args, **kwargs):
        class FakeClient:
            def xadd(self, *args, **kwargs):
                return fake_xadd(*args, **kwargs)

        return FakeClient()

    monkeypatch.setattr("app.events.publisher.redis.from_url", fake_redis)

    event = EquipmentFailureRiskEvent(
        equipment_id="eq-1", probability=0.92, model_version="v1", work_order_id="wo-1"
    )
    publish_equipment_failure_risk(event)

    assert len(published_events) == 1
    stream, data = published_events[0]
    assert stream == "equipment.failure_risk"
    assert data["equipment_id"] == "eq-1"
    assert data["probability"] == 0.92


def test_publish_work_order_approved_to_redis(monkeypatch):
    """Test that work order approval events publish correctly."""
    published_events = []

    def fake_xadd(stream, data, **kwargs):
        published_events.append((stream, data))
        return "1-0"

    def fake_redis(*args, **kwargs):
        class FakeClient:
            def xadd(self, *args, **kwargs):
                return fake_xadd(*args, **kwargs)

        return FakeClient()

    monkeypatch.setattr("app.events.publisher.redis.from_url", fake_redis)

    event = WorkOrderApprovedEvent(
        work_order_id="wo-1", equipment_id="eq-1", approved_by="tech_alice"
    )
    publish_work_order_approved(event)

    assert len(published_events) == 1
    stream, data = published_events[0]
    assert stream == "work_order.approved"
    assert data["approved_by"] == "tech_alice"


def test_publish_raises_event_publish_error_on_redis_unavailable(monkeypatch):
    """Test that a connection error is wrapped in EventPublishError."""
    from redis.exceptions import ConnectionError

    def fake_redis(*args, **kwargs):
        class FakeClient:
            def xadd(self, *args, **kwargs):
                raise ConnectionError("Redis is down")

        return FakeClient()

    monkeypatch.setattr("app.events.publisher.redis.from_url", fake_redis)

    event = EquipmentFailureRiskEvent(equipment_id="eq-1", probability=0.92, model_version="v1")
    with pytest.raises(EventPublishError):
        publish_equipment_failure_risk(event)


def test_publish_equipment_failure_risk_wraps_unexpected_errors_too(monkeypatch):
    """Not just ConnectionError — any unexpected failure during publish
    (a serialization bug, a malformed URL, etc.) must also come out as
    EventPublishError, not leak the raw exception type to the caller.
    """

    def fake_redis(*args, **kwargs):
        class FakeClient:
            def xadd(self, *args, **kwargs):
                raise ValueError("something unexpected")

        return FakeClient()

    monkeypatch.setattr("app.events.publisher.redis.from_url", fake_redis)

    event = EquipmentFailureRiskEvent(equipment_id="eq-1", probability=0.5, model_version="v1")
    with pytest.raises(EventPublishError):
        publish_equipment_failure_risk(event)


def test_publish_work_order_approved_wraps_unexpected_errors_too(monkeypatch):
    def fake_redis(*args, **kwargs):
        class FakeClient:
            def xadd(self, *args, **kwargs):
                raise ValueError("something unexpected")

        return FakeClient()

    monkeypatch.setattr("app.events.publisher.redis.from_url", fake_redis)

    event = WorkOrderApprovedEvent(
        work_order_id="wo-1", equipment_id="eq-1", approved_by="tech_alice"
    )
    with pytest.raises(EventPublishError):
        publish_work_order_approved(event)
