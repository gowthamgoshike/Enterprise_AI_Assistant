import inspect
import json
import logging
import time

from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from ai.schemas.graph_state import EnterpriseAgentState
from ai.tools.context import ToolExecutionContext
from ai.tools.langchain_registry import TOOL_MAP
from ai.tools.secure_registry import SECURE_TOOL_REGISTRY
from services.auth_service.authorization_service import require_permission
from services.auth_service.exceptions import AuthorizationError
from services.auth_service.models import UserRole


logger = logging.getLogger(__name__)
TRUSTED_CONTEXT_FIELDS = {
    "context",
    "user_id",
    "role",
    "customer_id",
    "permissions",
    "request_id",
}


def _validate_model_arguments(
    tool_name: str,
    arguments: dict,
) -> dict:
    schema_tool = TOOL_MAP.get(tool_name)
    if schema_tool is None:
        raise ValueError("Tool schema unavailable")

    untrusted_arguments = {
        key: value
        for key, value in arguments.items()
        if key not in TRUSTED_CONTEXT_FIELDS
    }
    return schema_tool.args_schema.model_validate(
        untrusted_arguments
    ).model_dump()


def _log_authorization(
    context: ToolExecutionContext,
    tool_name: str,
    result: str,
    started_at: float,
) -> None:
    latency = time.perf_counter() - started_at
    log = logger.warning if result == "DENIED" else logger.info
    event = (
        "tool_authorization_denied"
        if result == "DENIED"
        else "tool_authorization_allowed"
    )
    log(
        "%s request_id=%s user_id=%s role=%s tool=%s "
        "authorization=%s latency=%.3f",
        event,
        context.request_id,
        context.user_id,
        context.role.value,
        tool_name,
        result,
        latency,
        extra={
            "request_id": context.request_id,
            "user_id": context.user_id,
            "role": context.role.value,
            "tool": tool_name,
            "authorization_result": result,
            "authorization_latency": latency,
        },
    )


async def secure_tool_node(
    state: EnterpriseAgentState,
) -> dict:
    last_message = state["messages"][-1]
    context = ToolExecutionContext(
        user_id=state["user_id"],
        role=UserRole(state["user_role"]),
        customer_id=state.get("customer_id"),
        request_id=state["request_id"],
    )
    tool_messages = []
    executed_tools = []

    for call in getattr(last_message, "tool_calls", None) or []:
        started_at = time.perf_counter()
        tool_name = call.get("name", "")
        call_id = call.get("id", "unknown-tool-call")
        arguments = call.get("args", {})
        definition = SECURE_TOOL_REGISTRY.get(tool_name)
        status = "success"

        if definition is None:
            result = {"ok": False, "error": "unknown_tool"}
            status = "error"
            _log_authorization(
                context,
                tool_name or "unknown_tool",
                "DENIED",
                started_at,
            )
        elif not isinstance(arguments, dict):
            result = {"ok": False, "error": "invalid_arguments"}
            status = "error"
        else:
            try:
                validated_arguments = _validate_model_arguments(
                    tool_name,
                    arguments,
                )
            except (ValidationError, ValueError, TypeError):
                result = {"ok": False, "error": "invalid_arguments"}
                status = "error"
            else:
                try:
                    if definition.permission is not None:
                        require_permission(
                            context.role,
                            definition.permission,
                        )
                    result = definition.function(
                        context=context,
                        **validated_arguments,
                    )
                    if inspect.isawaitable(result):
                        result = await result
                except AuthorizationError:
                    result = {"ok": False, "error": "permission_denied"}
                    status = "error"
                    _log_authorization(
                        context,
                        tool_name,
                        "DENIED",
                        started_at,
                    )
                except Exception as exc:
                    _log_authorization(
                        context,
                        tool_name,
                        "ALLOWED",
                        started_at,
                    )
                    logger.error(
                        "Secure tool execution failed request_id=%s tool=%s "
                        "error_type=%s",
                        context.request_id,
                        tool_name,
                        type(exc).__name__,
                    )
                    result = {"ok": False, "error": "tool_execution_failed"}
                    status = "error"
                else:
                    executed_tools.append(tool_name)
                    _log_authorization(
                        context,
                        tool_name,
                        "ALLOWED",
                        started_at,
                    )

        tool_messages.append(ToolMessage(
            content=json.dumps(result, default=str),
            tool_call_id=call_id,
            name=tool_name or "unknown_tool",
            status=status,
        ))

    return {
        "messages": tool_messages,
        "current_turn_tools": [
            *state["current_turn_tools"],
            *executed_tools,
        ],
    }
