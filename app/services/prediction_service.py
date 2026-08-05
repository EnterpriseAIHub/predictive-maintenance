"""Prediction orchestration — ties together the data layer, the ML
layer, and the work-order business rule into the one operation the
API layer (Phase 8) and the batch job (Phase 9) will both call.

This module owns the transaction boundary: repositories only flush
(see their docstrings from the data-layer milestone), and this
function commits once, at the end, so the risk score and any work
order it triggers are written atomically together.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd
from sqlalchemy.orm import Session

from app.config import settings
from app.data.models.risk_score import RiskScore, RiskScoreSource
from app.data.models.work_order import WorkOrder
from app.data.repositories import equipment_repository, risk_score_repository, sensor_reading_repository
from app.ml.explain import FeatureAttribution
from app.ml.features import build_feature_vector
from app.ml.inference import get_model
from app.services.errors import EquipmentNotFoundError
from app.services.work_order_service import create_work_order_for_prediction


@dataclass(frozen=True)
class PredictionOutcome:
    equipment_id: str
    probability: float
    model_version: str
    attributions: list[FeatureAttribution]
    work_order: WorkOrder | None


def _readings_to_dataframe(readings: list) -> pd.DataFrame:
    """ORM rows -> the plain DataFrame shape app.ml.features expects.
    Keeps the ML layer's "no ORM objects" boundary intact (see the
    architecture doc, §7) — this conversion is the one place that
    boundary is crossed.
    """
    return pd.DataFrame(
        [{"sensor_type": r.sensor_type, "timestamp": r.timestamp, "value": r.value} for r in readings]
    )


def run_prediction_for_equipment(
    db: Session, equipment_id: str, as_of: datetime | None = None
) -> PredictionOutcome:
    """The real-time prediction path: fetch data, build features,
    score, persist, and (if warranted) open a work order — one
    atomic operation.
    """
    equipment = equipment_repository.get_by_id(db, equipment_id)
    if equipment is None:
        raise EquipmentNotFoundError(f"No equipment with id '{equipment_id}'.")

    as_of = as_of or datetime.now(UTC)
    since = as_of - timedelta(hours=settings.feature_lookback_hours)
    readings = sensor_reading_repository.get_recent(db, equipment_id, since)
    readings_df = _readings_to_dataframe(readings)

    features = build_feature_vector(readings_df, equipment.install_date, equipment.criticality_tier, as_of)

    model = get_model()
    result = model.predict_with_explanation(features, top_n=3)

    risk_score_repository.create(
        db,
        RiskScore(
            equipment_id=equipment_id,
            probability=result.probability,
            model_version=result.model_version,
            source=RiskScoreSource.REAL_TIME,
            created_at=as_of,
        ),
    )

    work_order = create_work_order_for_prediction(db, equipment_id, result.probability, as_of)

    db.commit()

    return PredictionOutcome(
        equipment_id=equipment_id,
        probability=result.probability,
        model_version=result.model_version,
        attributions=result.attributions,
        work_order=work_order,
    )