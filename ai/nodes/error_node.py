"""Explicit safe terminal path for agent failures."""

from langchain_core.messages import AIMessage

from ai.schemas.graph_state import EnterpriseAgentState
from core.observability.metrics import AGENT_ERRORS_TOTAL


DEFAULT_ERROR_MESSAGE = "The request could not be completed."


def error_node(state: EnterpriseAgentState) -> dict:
    AGENT_ERRORS_TOTAL.labels(
        error_code=state.get("error_code") or "internal_error"
    ).inc()
    return {
        "messages": [AIMessage(content=(
            state.get("error_message") or DEFAULT_ERROR_MESSAGE
        ))],
    }
