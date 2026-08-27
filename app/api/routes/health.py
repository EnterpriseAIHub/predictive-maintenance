from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.ml.inference import ModelNotFoundError, get_model

router = APIRouter(tags=["health"])


@router.get("/health")
def liveness() -> dict:
    """Process is up. Does not touch any dependency."""
    return {"status": "ok"}


@router.get("/health/ready")
def readiness(db: Session = Depends(get_db)) -> dict:
    """Process is up AND its dependencies are reachable AND a model is
    loaded and ready to serve predictions.
    """
    db.execute(text("SELECT 1"))

    try:
        get_model()
    except ModelNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return {"status": "ready"}
