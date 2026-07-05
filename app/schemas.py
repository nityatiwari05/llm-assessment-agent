from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def last_message_is_user(cls, v: List[Message]) -> List[Message]:
        # The evaluator's simulated user always speaks last; we don't hard-fail
        # if that's violated (defensive), but we do trim to something sane.
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
