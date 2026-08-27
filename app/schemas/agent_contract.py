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
