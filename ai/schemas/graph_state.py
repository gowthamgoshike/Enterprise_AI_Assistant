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
    route: str | None
    current_turn_tools: list[str]
    error: str | None
