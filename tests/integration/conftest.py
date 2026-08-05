"""Integration test fixtures.

These tests run against the real Postgres instance (see
docker-compose.override.yml / .env) rather than mocks, per the EDD's
testing strategy (§21) — repository/query behavior is exactly the
kind of thing that looks right and silently isn't.

Isolation: the service layer legitimately calls session.commit()
(prediction_service and work_order_service own their transaction
boundary — see their docstrings), so a plain "rollback in a finally
block" stops working the moment any test exercises code that commits.
Instead, each test's session is bound to a SAVEPOINT nested inside an
outer connection-level transaction: session.commit() releases the
savepoint (and opens a new one) without touching the outer
transaction, which is rolled back unconditionally when the test ends.
"""

import pytest
from sqlalchemy.orm import Session

from app.data import models  # noqa: F401 — ensures models are registered
from app.data.base import SessionLocal, engine


@pytest.fixture
def db() -> Session:
    connection = engine.connect()
    outer_transaction = connection.begin()
    session = SessionLocal(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        outer_transaction.rollback()
        connection.close()