# Event publishing via Redis Streams (publisher.py). Publishes domain
# events (EquipmentFailureRiskEvent, WorkOrderApprovedEvent) after writes
# so other platform repos can react asynchronously. Events are best-effort:
# if Redis is unavailable, the event is dropped silently, but the write
# (prediction, approval) that triggered it is always durable — Redis
# unavailability never blocks the core business logic.
