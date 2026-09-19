from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal[
        "user",
        "assistant",
    ]
    content: str = Field(
        min_length=1,
        max_length=10000,
    )


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        min_length=1,
        max_length=50,
    )
    session_id: str = Field(
        min_length=1,
        max_length=100,
    )


class ChatResponse(BaseModel):

    request_id: str

    response: str

    model: str

    latency_seconds: float

    tools_used: list[str] = Field(default_factory=list)

    route: str | None = None

    route_confidence: float | None = None

    agent_action: Literal[
        "respond",
        "tool",
    ]
