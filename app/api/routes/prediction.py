"""Real-time prediction endpoints.

Thin adapter over app.services.prediction_service — every route here does request/response translation only. No business logic lives in this file (see the architecture doc, §3: business rules belong in the service layer, not the API layer).
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.data.repositories import equipment_repository, risk_score_repository
from app.services.errors import EquipmentNotFoundError
from app.services.prediction_service import run_prediction_for_equipment

router = APIRouter(tags=["prediction"])


class PredictRequest(BaseModel):
    equipment_id: str
    as_of: datetime | None = None  # defaults to now() in the service layer


class FeatureAttributionResponse(BaseModel):
    feature: str
    shap_value: float
    feature_value: float


class WorkOrderResponse(BaseModel):
    id: str
    priority: str
    recommended_priority: str | None
    status: str
    created_at: datetime


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    equipment_id: str
    probability: float
    model_version: str
    attributions: list[FeatureAttributionResponse]
    work_order: WorkOrderResponse | None


class RiskScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    equipment_id: str
    probability: float
    model_version: str
    source: str
    created_at: datetime


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, db: Session = Depends(get_db)) -> PredictResponse:
    """Runs a real-time prediction for one asset: fetches recent sensor data, scores it, persists the result, and opens a work order if the risk policy warrants one (see app.services.prediction_service).

    Raises (mapped to HTTP by app.api.error_handlers):
    - EquipmentNotFoundError -> 404
    - ModelNotFoundError -> 503 (no trained model registered yet)
    """
    outcome = run_prediction_for_equipment(db, request.equipment_id, as_of=request.as_of)

    return PredictResponse(
        equipment_id=outcome.equipment_id,
        probability=outcome.probability,
        model_version=outcome.model_version,
        attributions=[
            FeatureAttributionResponse(
                feature=a.feature, shap_value=a.shap_value, feature_value=a.feature_value
            )
            for a in outcome.attributions
        ],
        work_order=(
            WorkOrderResponse(
                id=outcome.work_order.id,
                priority=outcome.work_order.priority.value,
                recommended_priority=(
                    outcome.work_order.recommended_priority.value
                    if outcome.work_order.recommended_priority
                    else None
                ),
                status=outcome.work_order.status.value,
                created_at=outcome.work_order.created_at,
            )
            if outcome.work_order
            else None
        ),
    )


@router.get("/equipment/{equipment_id}/risk", response_model=RiskScoreResponse)
def get_latest_risk(equipment_id: str, db: Session = Depends(get_db)) -> RiskScoreResponse:
    """Returns the most recently persisted risk score for this asset —
    does NOT run a new prediction (use POST /predict for that). This
    is a cheap read of prediction history, e.g. for a dashboard.
    """
    equipment = equipment_repository.get_by_id(db, equipment_id)
    if equipment is None:
        raise EquipmentNotFoundError(f"No equipment with id '{equipment_id}'.")

    risk_score = risk_score_repository.get_latest(db, equipment_id)
    if risk_score is None:
        raise HTTPException(
            status_code=404, detail=f"No predictions have been recorded yet for '{equipment_id}'."
        )

    return RiskScoreResponse(
        equipment_id=risk_score.equipment_id,
        probability=risk_score.probability,
        model_version=risk_score.model_version,
        source=risk_score.source.value,
        created_at=risk_score.created_at,
    )
