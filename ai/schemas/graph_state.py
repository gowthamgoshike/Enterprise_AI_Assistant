from typing import TypedDict


class EnterpriseState(TypedDict):
    user_message: str
    request_id: str
    route: str | None
    route_confidence: float | None
    response: str | None
    tools_used: list[str]
    error: str | None
