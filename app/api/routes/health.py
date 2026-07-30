"""Liveness and readiness endpoints.

Readiness will grow a model-artifact-loaded check once inference
exists (per the EDD's error handling: the API should refuse
predictions with a 503 if the registered model failed to load rather
than serving from a stale/partial model). For now it only verifies
the database is reachable, since that's the one dependency this phase
actually wires up.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def liveness() -> dict:
    """Process is up. Does not touch any dependency."""
    return {"status": "ok"}


@router.get("/health/ready")
def readiness(db: Session = Depends(get_db)) -> dict:
    """Process is up AND its dependencies are reachable."""
    db.execute(text("SELECT 1"))
    return {"status": "ready"}
