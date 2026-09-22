"""Stable, privacy-conscious events sent over the chat SSE endpoint."""

from typing import Any, Literal

from pydantic import BaseModel


class StreamEvent(BaseModel):
    type: Literal[
        "status",
        "token",
        "tool_start",
        "tool_end",
        "approval_required",
        "done",
        "error",
    ]
    data: dict[str, Any]
