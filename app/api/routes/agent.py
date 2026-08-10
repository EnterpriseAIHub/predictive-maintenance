"""The Agent Contract endpoint.

This is what lets a future platform-orchestrator repo treat this
service identically to a RAG-based repo (e.g. maintenance-copilot) —
same request/response shape regardless of what's happening internally
(a trained ML model here, a retrieval pipeline elsewhere). See
app/schemas/agent_contract.py for the shared shape.

For this repo, the "query" is expected to be about one asset's risk;
`context.equipment_id` is required. There's no natural-language query
parsing here — this repo's AI is a classifier, not an LLM, so the
"agent" framing is about the response shape, not about interpreting
free text.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.agent_contract import AgentRequest, AgentResponse
from app.services.prediction_service import run_prediction_for_equipment

router = APIRouter(tags=["agent"])


@router.post("/agent", response_model=AgentResponse)
def handle_agent_request(request: AgentRequest, db: Session = Depends(get_db)) -> AgentResponse:
    equipment_id = request.context.get("equipment_id")
    if not equipment_id:
        raise HTTPException(status_code=422, detail="context.equipment_id is required.")

    as_of_raw = request.context.get("as_of")
    as_of = datetime.fromisoformat(as_of_raw) if as_of_raw else None

    outcome = run_prediction_for_equipment(db, equipment_id, as_of=as_of)

    answer = (
        f"Equipment '{equipment_id}' has a {outcome.probability:.0%} predicted failure risk "
        f"(model {outcome.model_version})."
    )
    if outcome.work_order:
        answer += (
            f" Work order {outcome.work_order.id} was opened at "
            f"{outcome.work_order.priority.value} priority."
        )
    else:
        answer += " No work order was opened — risk is below the action threshold."

    # provenance mirrors how a RAG-based repo would cite source
    # documents — here it's the SHAP attributions that justify the
    # prediction, in the same "list of evidence strings" shape.
    provenance = [
        f"{a.feature}={a.feature_value:.2f} (shap={a.shap_value:+.3f})"
        for a in outcome.attributions
    ]

    return AgentResponse(
        answer=answer,
        confidence=outcome.probability,
        provenance=provenance,
        structured_data={
            "equipment_id": outcome.equipment_id,
            "probability": outcome.probability,
            "model_version": outcome.model_version,
            "work_order_id": outcome.work_order.id if outcome.work_order else None,
        },
    )
