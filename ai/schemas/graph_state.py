from typing import TypedDict

from langgraph.graph import MessagesState


class EnterpriseState(TypedDict):
    user_message: str
    request_id: str
    route: str | None
    route_confidence: float | None
    response: str | None
    tools_used: list[str]
    error: str | None


class EnterpriseAgentState(MessagesState):
    request_id: str
    user_id: str
    user_role: str
    customer_id: str | None
    route: str | None
    current_turn_tools: list[str]
    conversation_summary: str | None
    summarized_message_count: int
    pending_action: dict | None
    pending_tool_call_id: str | None
    approval_status: str | None
    error: str | None
    error_code: str | None
    error_message: str | None
    retry_count: int
