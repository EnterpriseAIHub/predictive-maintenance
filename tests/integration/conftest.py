"""Integration test fixtures.

These tests run against the real Postgres instance (see
docker-compose.override.yml / .env) rather than mocks, per the EDD's
testing strategy (§21) — repository/query behavior is exactly the
kind of thing that looks right and silently isn't. Each test runs
inside a transaction that's rolled back afterward, so tests don't
leave data behind for each other.
"""

import pytest
from sqlalchemy.orm import Session

from app.data import models  # noqa: F401 — ensures models are registered on Base.metadata
from app.data.base import SessionLocal, engine


@pytest.fixture
def db() -> Session:
    session = SessionLocal(bind=engine)
    try:
        yield session
    finally:
        session.rollback()
        session.close()
