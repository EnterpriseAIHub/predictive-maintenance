"""Integration test fixtures.

These tests run against the real Postgres instance (see
docker-compose.override.yml / .env) rather than mocks, per the EDD's testing strategy (§21) — repository/query behavior is exactly the kind of thing that looks right and silently isn't.

Isolation: the service layer legitimately calls session.commit()
(prediction_service and work_order_service own their transaction
boundary — see their docstrings), so a plain "rollback in a finally block" stops working the moment any test exercises code that commits. Instead, each test's session is bound to a SAVEPOINT nested inside an outer connection-level transaction: session.commit() releases the savepoint (and opens a new one) without touching the outer transaction, which is rolled back unconditionally when the test ends.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.data import models  # noqa: F401 — ensures models are registered
from app.data.base import SessionLocal, engine
from app.main import app


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


@pytest.fixture
def client(db: Session) -> TestClient:
    """A TestClient whose get_db dependency is overridden to use the
    SAME savepoint-isolated session as the `db` fixture above — so an
    API test can seed data via `db` directly and see it through HTTP,
    and any commit() the request triggers still rolls back cleanly at
    the end of the test.
    """

    def _override_get_db():
        yield db

    app.dependency_overrides[get_db] = _override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)
