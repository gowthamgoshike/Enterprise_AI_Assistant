import json

from langchain_core.messages import ToolMessage

from ai.schemas.graph_state import EnterpriseAgentState


def action_approved_node(state: EnterpriseAgentState) -> dict:
    """Complete approval bookkeeping without executing the proposed refund."""
    result = {
        "ok": True,
        "status": "approved_for_execution",
        "action": state["pending_action"],
    }
    return {
        "messages": [ToolMessage(
            content=json.dumps(result),
            tool_call_id=state["pending_tool_call_id"],
            name="propose_refund",
            status="success",
        )],
        "current_turn_tools": [
            *state["current_turn_tools"],
            "propose_refund",
        ],
        "pending_action": None,
        "pending_tool_call_id": None,
    }


def action_rejected_node(state: EnterpriseAgentState) -> dict:
    """Report rejection to the model without changing transaction data."""
    result = {
        "ok": False,
        "status": "rejected_by_human",
    }
    return {
        "messages": [ToolMessage(
            content=json.dumps(result),
            tool_call_id=state["pending_tool_call_id"],
            name="propose_refund",
            status="error",
        )],
        "current_turn_tools": [
            *state["current_turn_tools"],
            "propose_refund",
        ],
        "pending_action": None,
        "pending_tool_call_id": None,
    }
