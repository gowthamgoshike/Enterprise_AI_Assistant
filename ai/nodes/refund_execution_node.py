import json

from langchain_core.messages import ToolMessage

from ai.schemas.graph_state import EnterpriseAgentState
from services.auth_service.authorization_service import require_permission
from services.auth_service.exceptions import AuthorizationError
from services.auth_service.models import UserRole
from services.auth_service.permissions import Permission
from services.transaction_service.exceptions import (
    RefundError,
    RefundNotAllowedError,
    TransactionNotFoundError,
)
from services.transaction_service.refund_execution_service import execute_refund


def _failed_execution(
    state: EnterpriseAgentState,
    *,
    error: str,
    message: str,
) -> dict:
    return {
        "messages": [ToolMessage(
            content=json.dumps({
                "ok": False,
                "error": error,
                "message": message,
            }),
            tool_call_id=state["pending_tool_call_id"],
            name="propose_refund",
            status="error",
        )],
        "current_turn_tools": [
            *state["current_turn_tools"],
            "propose_refund",
            "execute_refund",
        ],
        "pending_action": None,
        "pending_tool_call_id": None,
        "approval_status": "approved",
    }


async def execute_refund_node(state: EnterpriseAgentState) -> dict:
    """Execute an approved refund using the checkpointed proposal identity."""
    try:
        require_permission(
            UserRole(state["user_role"]),
            Permission.EXECUTE_REFUND,
        )
        proposal = state["pending_action"]
        result = execute_refund(
            action_id=proposal["action_id"],
            transaction_id=proposal["transaction_id"],
            reason=proposal["reason"],
            requested_by=proposal["requested_by"],
            approved_by=state["user_id"],
            request_id=state["request_id"],
        )
    except AuthorizationError:
        return _failed_execution(
            state,
            error="permission_denied",
            message="Refund execution permission denied",
        )
    except TransactionNotFoundError:
        return _failed_execution(
            state,
            error="transaction_unavailable",
            message="Transaction unavailable",
        )
    except RefundNotAllowedError as exc:
        return _failed_execution(
            state,
            error="refund_not_allowed",
            message=str(exc),
        )
    except RefundError:
        return _failed_execution(
            state,
            error="refund_execution_failed",
            message="Refund execution failed",
        )

    tool_result = {
        "ok": True,
        "refund": result.model_dump(mode="json"),
    }
    return {
        "messages": [ToolMessage(
            content=json.dumps(tool_result, default=str),
            tool_call_id=state["pending_tool_call_id"],
            name="propose_refund",
            status="success",
        )],
        "current_turn_tools": [
            *state["current_turn_tools"],
            "propose_refund",
            "execute_refund",
        ],
        "pending_action": None,
        "pending_tool_call_id": None,
        "approval_status": "approved",
    }
