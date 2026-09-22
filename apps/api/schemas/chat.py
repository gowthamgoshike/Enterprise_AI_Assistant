from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal[
        "user",
        "assistant",
    ]
    content: str = Field(
        min_length=1,
        max_length=8000,
    )


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(
        min_length=1,
        max_length=50,
    )
    session_id: str = Field(
        min_length=1,
        max_length=128,
    )


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(
        min_length=1,
        max_length=128,
    )
    approved: bool
    comment: str | None = Field(
        default=None,
        max_length=1000,
    )


class ChatResponse(BaseModel):

    request_id: str

    response: str | None = None

    status: Literal[
        "completed",
        "approval_required",
        "failed",
        "degraded",
    ] = "completed"

    approval_request: dict | None = None

    model: str

    latency_seconds: float

    tools_used: list[str] = Field(default_factory=list)

    route: str | None = None

    route_confidence: float | None = None

    agent_action: Literal[
        "respond",
        "tool",
        "approval",
        "error",
    ]

    error_code: str | None = None

    retryable: bool = False
