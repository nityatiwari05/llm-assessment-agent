import pytest
from pydantic import ValidationError

from app.schemas import ChatRequest, ChatResponse, Recommendation


def test_chat_request_requires_messages():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])


def test_chat_request_parses_valid_payload():
    req = ChatRequest(
        messages=[
            {"role": "user", "content": "Hiring a Java developer"},
            {"role": "assistant", "content": "What seniority level?"},
            {"role": "user", "content": "Mid-level"},
        ]
    )
    assert len(req.messages) == 3
    assert req.messages[0].role == "user"


def test_chat_response_default_empty_recommendations():
    resp = ChatResponse(reply="Can you tell me more about the role?")
    assert resp.recommendations == []
    assert resp.end_of_conversation is False


def test_chat_response_serializes_matching_spec_shape():
    resp = ChatResponse(
        reply="Here are 2 assessments.",
        recommendations=[
            Recommendation(name="Java 8 (New)", url="https://www.shl.com/x/", test_type="K"),
        ],
        end_of_conversation=False,
    )
    dumped = resp.model_dump()
    assert set(dumped.keys()) == {"reply", "recommendations", "end_of_conversation"}
    assert set(dumped["recommendations"][0].keys()) == {"name", "url", "test_type"}


def test_recommendation_count_bounds_are_not_enforced_at_schema_level():
    # The 1-10 bound is a business rule enforced in app/agent.py (0 is allowed
    # when still gathering context/refusing), not a hard Pydantic constraint —
    # this test documents that choice.
    resp = ChatResponse(reply="ok", recommendations=[])
    assert resp.recommendations == []
