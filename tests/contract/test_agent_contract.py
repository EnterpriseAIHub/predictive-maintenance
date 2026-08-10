"""Validates the Agent Contract shapes directly — independent of any
one route's usage of them. This is what a future platform-orchestrator
repo's own tests would check before trusting this repo's responses.
"""

import pytest
from pydantic import ValidationError

from app.schemas.agent_contract import AgentRequest, AgentResponse


def test_agent_request_requires_query():
    with pytest.raises(ValidationError):
        AgentRequest(context={"equipment_id": "eq-1"})


def test_agent_request_context_defaults_to_empty_dict():
    request = AgentRequest(query="how risky is eq-1?")
    assert request.context == {}


def test_agent_request_accepts_arbitrary_context_keys():
    # Context is intentionally a free-form dict — different domain
    # repos will need different context shapes (equipment_id here,
    # something else in maintenance-copilot) without a contract change.
    request = AgentRequest(query="q", context={"equipment_id": "eq-1", "anything_else": 42})
    assert request.context["anything_else"] == 42


def test_agent_response_requires_answer_confidence_and_provenance():
    with pytest.raises(ValidationError):
        AgentResponse(confidence=0.5, provenance=[])  # missing answer


def test_agent_response_confidence_must_be_a_valid_probability():
    with pytest.raises(ValidationError):
        AgentResponse(answer="a", confidence=1.5, provenance=[])
    with pytest.raises(ValidationError):
        AgentResponse(answer="a", confidence=-0.1, provenance=[])


def test_agent_response_structured_data_is_optional():
    response = AgentResponse(answer="a", confidence=0.5, provenance=[])
    assert response.structured_data is None


def test_agent_response_is_json_serializable():
    # The whole point of the contract is that it crosses a process
    # boundary — if this doesn't round-trip through JSON cleanly, it
    # can't actually serve as an HTTP response body.
    response = AgentResponse(
        answer="a", confidence=0.9, provenance=["x=1"], structured_data={"k": "v"}
    )
    assert response.model_dump_json()  # raises if not serializable
