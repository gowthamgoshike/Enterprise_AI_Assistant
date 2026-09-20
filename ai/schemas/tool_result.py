"""Sanitized envelope exchanged between tools and the agent."""

from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    ok: bool
    data: Any | None = None
    error_code: str | None = None
    message: str | None = None
    retryable: bool = False

    @classmethod
    def success(cls, data: Any) -> "ToolResult":
        return cls(ok=True, data=data)

    @classmethod
    def failure(
        cls,
        *,
        error_code: str,
        message: str,
        retryable: bool = False,
    ) -> "ToolResult":
        return cls(
            ok=False,
            error_code=error_code,
            message=message,
            retryable=retryable,
        )
