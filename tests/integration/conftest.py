"""Integration test fixtures.

These tests run against the real PostgreSQL database.

Each test starts with a clean set of application tables and runs inside
its own transaction so test data cannot leak into other tests.

The application services legitimately call session.commit(), so the
session uses SQLAlchemy's SAVEPOINT-aware transaction mode.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.data import models  # noqa: F401
from app.data.base import Base, SessionLocal, engine
from app.main import app


@pytest.fixture
def db() -> Session:
    connection = engine.connect()
    outer_transaction = connection.begin()

    session = SessionLocal(
        bind=connection,
        join_transaction_mode="create_savepoint",
    )

    try:
        # Integration tests must start with an empty database.
        # The demo data belongs to the running application, not the tests.
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())

        session.flush()

        yield session

    finally:
        session.close()
        outer_transaction.rollback()
        connection.close()

@pytest.fixture
def client(db: Session) -> TestClient:
    """Provide a TestClient using the same isolated database session."""

    def _override_get_db():
        yield db

    app.dependency_overrides[get_db] = _override_get_db

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_db, None)