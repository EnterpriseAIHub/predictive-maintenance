"""SQLAlchemy engine, session factory, and declarative base.

This is the one place the database connection is configured. ORM
models (added in a later phase, under `app/data/models/`) all inherit
from `Base`; Alembic's `env.py` points at the same `Base.metadata` so
migrations stay in sync with the models automatically.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

engine = create_engine(settings.database_url, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """Declarative base every ORM model in this repo inherits from."""


def get_db_session() -> Session:
    """Yields a database session for use as a FastAPI dependency.
    Kept here (rather than only in app/api/deps.py) so non-API entrypoints — the batch job in particular — can open a session the same way the API does, instead of duplicating connection setup.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
