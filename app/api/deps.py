"""Shared FastAPI dependencies.

Routes depend on these rather than importing `app.data.base` directly,
so route handlers stay free of persistence-layer wiring details.
"""

from collections.abc import Generator

from sqlalchemy.orm import Session

from app.data.base import SessionLocal


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
