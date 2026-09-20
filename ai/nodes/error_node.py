"""Explicit safe terminal path for agent failures."""

from langchain_core.messages import AIMessage

from ai.schemas.graph_state import EnterpriseAgentState


DEFAULT_ERROR_MESSAGE = "The request could not be completed."


def error_node(state: EnterpriseAgentState) -> dict:
    return {
        "messages": [AIMessage(content=(
            state.get("error_message") or DEFAULT_ERROR_MESSAGE
        ))],
    }
