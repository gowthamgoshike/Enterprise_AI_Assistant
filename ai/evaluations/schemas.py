"""Validated contracts for golden cases, traces, and saved evaluation runs."""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


EvaluationCategory = Literal[
    "tool_selection",
    "no_tool",
    "security",
    "grounding",
    "conversation",
    "workflow",
    "error_handling",
]
EvaluationStatus = Literal[
    "completed",
    "approval_required",
    "degraded",
    "failed",
]


class EvaluationIdentity(BaseModel):
    user_id: str
    role: Literal["CUSTOMER", "ANALYST", "ADMIN"]
    customer_id: str | None = None

    @model_validator(mode="after")
    def require_customer_scope(self):
        if self.role == "CUSTOMER" and not self.customer_id:
            raise ValueError("CUSTOMER evaluation identities require customer_id")
        return self


class ExpectedFact(BaseModel):
    """A response fact rendered in a stable, human-readable form."""

    name: str
    value: str


class EvaluationExpectation(BaseModel):
    expected_tool: str | None = None
    expected_arguments: dict[str, Any] | None = None
    expected_status: EvaluationStatus | None = None
    expected_error_code: str | None = None
    expected_approval_status: Literal["pending", "approved", "rejected"] | None = None
    expected_executed_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_response_contains: list[str] = Field(default_factory=list)
    forbidden_response_contains: list[str] = Field(default_factory=list)
    expected_facts: list[ExpectedFact] = Field(default_factory=list)

    @model_validator(mode="after")
    def arguments_require_a_tool(self):
        if self.expected_arguments is not None and self.expected_tool is None:
            raise ValueError("expected_arguments requires expected_tool")
        return self


class EvaluationCase(EvaluationExpectation):
    id: str
    category: EvaluationCategory
    user_message: str
    identity: EvaluationIdentity
    description: str | None = None
    fixture: str | None = None
    approval: dict[str, Any] | None = None
    critical: bool = False
    tags: list[str] = Field(default_factory=lambda: ["full"])


class ConversationTurn(EvaluationExpectation):
    user_message: str
    thread_key: str = "primary"
    fixture_action: str | None = None


class ConversationEvaluationCase(BaseModel):
    id: str
    category: Literal["conversation"] = "conversation"
    identity: EvaluationIdentity
    turns: list[ConversationTurn] = Field(min_length=2)
    description: str | None = None
    fixture: str | None = None
    critical: bool = False
    tags: list[str] = Field(default_factory=lambda: ["full"])


class ToolCallTrace(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None
    executed: bool = False


class EvaluationResult(BaseModel):
    case_id: str
    category: EvaluationCategory
    passed: bool
    expected_tool: str | None = None
    actual_tools: list[str] = Field(default_factory=list)
    actual_executed_tools: list[str] = Field(default_factory=list)
    actual_tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    expected_status: EvaluationStatus | None = None
    actual_status: EvaluationStatus | None = None
    latency_ms: float | None = None
    response: str | None = None
    error_code: str | None = None
    approval_status: str | None = None
    failure_reasons: list[str] = Field(default_factory=list)
    turn_results: list["EvaluationResult"] = Field(default_factory=list)


class EvaluationRun(BaseModel):
    run_id: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    model: str
    prompt_version: str
    application_version: str
    suite: str
    total_cases: int
    passed: int
    failed: int
    pass_rate: float
    scores: dict[str, float]
    average_latency_ms: float | None = None
    release_allowed: bool
    gate_failures: list[str] = Field(default_factory=list)
    results: list[EvaluationResult]
