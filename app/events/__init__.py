# The Redis Streams publisher lives here: serializes the
# `equipment.failure_risk.high` event and handles retry-with-backoff
# on publish failure, deliberately isolated so a Redis outage can
# never block the core work-order write (EDD §19). Added in the phase
# that implements inference and event publishing.
