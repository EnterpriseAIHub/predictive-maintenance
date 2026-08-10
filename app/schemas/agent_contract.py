"""The platform-wide Agent Contract.

Every domain repo exposes itself to `platform-orchestrator` through
this uniform shape (see the platform architecture doc, §6) so the
orchestrator can treat a trained ML model and a RAG pipeline
identically. This will move to `platform-agent-sdk` once that shared
package exists; kept here for now for the same standalone-runnability
reason as `equipment.py`.
"""

from typing import Any

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    query: str
    context: dict[str, Any] = {}


class AgentResponse(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: list[str]
    structured_data: dict[str, Any] | None = None
