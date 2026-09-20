import inspect
import json
import logging
import time
import asyncio

from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from ai.schemas.graph_state import EnterpriseAgentState
from ai.schemas.tool_result import ToolResult
from ai.tools.context import ToolExecutionContext
from ai.tools.langchain_registry import TOOL_MAP
from ai.tools.secure_registry import SECURE_TOOL_REGISTRY
from services.auth_service.authorization_service import require_permission
from services.auth_service.exceptions import AuthorizationError
from services.auth_service.models import UserRole
from core.config import (
    DATABASE_MAX_RETRIES,
    TOOL_MAX_RETRIES,
    TOOL_TIMEOUT_SECONDS,
)
from core.errors import (
    BusinessRuleError,
    DatabaseUnavailableError,
    PermissionDeniedError,
    ToolTimeoutError,
)
from core.retry import retry_async, run_with_timeout
from core.observability.metrics import (
    AUTHORIZATION_DENIALS_TOTAL,
    TOOL_CALLS_TOTAL,
    TOOL_LATENCY,
)


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
    if result == "DENIED":
        capability = (
            tool_name
            if tool_name in SECURE_TOOL_REGISTRY
            else "unknown_tool"
        )
        AUTHORIZATION_DENIALS_TOTAL.labels(
            role=context.role.value,
            capability=capability,
        ).inc()
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


def _record_tool_execution(
    *,
    tool_name: str,
    status: str,
    role: str,
    started_at: float,
) -> None:
    duration_seconds = time.perf_counter() - started_at
    metric_status = "success" if status == "success" else "failure"
    TOOL_CALLS_TOTAL.labels(tool=tool_name, status=metric_status).inc()
    TOOL_LATENCY.labels(tool=tool_name).observe(duration_seconds)
    logger.info(
        "tool_execution_completed",
        extra={
            "event": "tool_execution",
            "tool": tool_name,
            "status": metric_status,
            "latency_ms": round(duration_seconds * 1000, 2),
            "role": role,
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
    error_code = None
    error_message = None
    retry_count = state.get("retry_count", 0)

    def record_retry(_attempt: int, _delay: float, _error: Exception) -> None:
        nonlocal retry_count
        retry_count += 1

    def failure_result(error: Exception) -> ToolResult:
        return ToolResult.failure(
            error_code=getattr(error, "code", "internal_error"),
            message=getattr(
                error,
                "safe_message",
                "The operation could not be completed.",
            ),
            retryable=bool(getattr(error, "retryable", False)),
        )

    async def invoke(definition, validated_arguments, context):
        if inspect.iscoroutinefunction(definition.function):
            return await run_with_timeout(
                definition.function(
                    context=context,
                    **validated_arguments,
                ),
                TOOL_TIMEOUT_SECONDS,
            )
        # Keep synchronous database work off the event loop. A timeout is not
        # applied here because cancelling asyncio.to_thread cannot cancel the
        # underlying database operation safely.
        return await asyncio.to_thread(
            definition.function,
            context=context,
            **validated_arguments,
        )

    for call in getattr(last_message, "tool_calls", None) or []:
        started_at = time.perf_counter()
        tool_name = call.get("name", "")
        call_id = call.get("id", "unknown-tool-call")
        arguments = call.get("args", {})
        definition = SECURE_TOOL_REGISTRY.get(tool_name)
        status = "success"

        if definition is None:
            result = ToolResult.failure(
                error_code="unknown_tool",
                message="The requested tool is not available.",
            )
            error_code = result.error_code
            error_message = result.message
            status = "error"
            _log_authorization(
                context,
                tool_name or "unknown_tool",
                "DENIED",
                started_at,
            )
        elif not isinstance(arguments, dict):
            result = ToolResult.failure(
                error_code="invalid_tool_arguments",
                message="The tool arguments were invalid.",
            )
            error_code = result.error_code
            error_message = result.message
            status = "error"
        else:
            try:
                validated_arguments = _validate_model_arguments(
                    tool_name,
                    arguments,
                )
            except (ValidationError, ValueError, TypeError):
                result = ToolResult.failure(
                    error_code="invalid_tool_arguments",
                    message="The tool arguments were invalid.",
                )
                error_code = result.error_code
                error_message = result.message
                status = "error"
            else:
                try:
                    if definition.permission is not None:
                        require_permission(
                            context.role,
                            definition.permission,
                        )
                    operation = lambda: invoke(
                        definition,
                        validated_arguments,
                        context,
                    )
                    if definition.retry_safe:
                        max_retries = (
                            TOOL_MAX_RETRIES
                            if inspect.iscoroutinefunction(definition.function)
                            else DATABASE_MAX_RETRIES
                        )
                        raw_result = await retry_async(
                            operation,
                            max_retries=max_retries,
                            retry_on=(
                                DatabaseUnavailableError,
                                ToolTimeoutError,
                            ),
                            operation_name=f"tool:{tool_name}",
                            request_id=context.request_id,
                            on_retry=record_retry,
                        )
                    else:
                        raw_result = await operation()
                    if definition.requires_approval:
                        _log_authorization(
                            context,
                            tool_name,
                            "ALLOWED",
                            started_at,
                        )
                        logger.info(
                            "human_approval_requested request_id=%s "
                            "user_id=%s action_id=%s action_type=%s",
                            context.request_id,
                            context.user_id,
                            raw_result["action_id"],
                            raw_result["action_type"],
                            extra={
                                "request_id": context.request_id,
                                "user_id": context.user_id,
                                "action_id": raw_result["action_id"],
                                "action_type": raw_result["action_type"],
                            },
                        )
                        _record_tool_execution(
                            tool_name=tool_name,
                            status="success",
                            role=context.role.value,
                            started_at=started_at,
                        )
                        return {
                            "messages": tool_messages,
                            "current_turn_tools": [
                                *state["current_turn_tools"],
                                *executed_tools,
                            ],
                            "pending_action": raw_result,
                            "pending_tool_call_id": call_id,
                            "approval_status": "pending",
                        }
                    result = ToolResult.success(raw_result)
                except AuthorizationError:
                    error = PermissionDeniedError()
                    result = failure_result(error)
                    status = "error"
                    error_code = result.error_code
                    error_message = result.message
                    _log_authorization(
                        context,
                        tool_name,
                        "DENIED",
                        started_at,
                    )
                except (DatabaseUnavailableError, ToolTimeoutError) as exc:
                    result = failure_result(exc)
                    status = "error"
                    error_code = result.error_code
                    error_message = (
                        "I couldn't verify the current transaction because "
                        "the transaction service is temporarily unavailable."
                        if isinstance(exc, DatabaseUnavailableError)
                        and tool_name == "get_transaction"
                        else result.message
                    )
                    _log_authorization(
                        context,
                        tool_name,
                        "ALLOWED",
                        started_at,
                    )
                except BusinessRuleError as exc:
                    result = failure_result(exc)
                    status = "error"
                    error_code = result.error_code
                    error_message = result.message
                    _log_authorization(
                        context,
                        tool_name,
                        "ALLOWED",
                        started_at,
                    )
                except (ValueError, TypeError):
                    result = ToolResult.failure(
                        error_code="invalid_tool_arguments",
                        message="The tool arguments were invalid.",
                    )
                    status = "error"
                    error_code = result.error_code
                    error_message = result.message
                    _log_authorization(
                        context,
                        tool_name,
                        "ALLOWED",
                        started_at,
                    )
                except Exception:
                    _log_authorization(
                        context,
                        tool_name,
                        "ALLOWED",
                        started_at,
                    )
                    logger.exception(
                        "Unexpected tool failure request_id=%s tool=%s",
                        context.request_id,
                        tool_name,
                    )
                    result = ToolResult.failure(
                        error_code="internal_error",
                        message="The operation could not be completed.",
                    )
                    status = "error"
                    error_code = result.error_code
                    error_message = result.message
                else:
                    executed_tools.append(tool_name)
                    _log_authorization(
                        context,
                        tool_name,
                        "ALLOWED",
                        started_at,
                    )

        tool_messages.append(ToolMessage(
            content=json.dumps(
                result.model_dump(mode="json")
                if isinstance(result, ToolResult)
                else result,
                default=str,
            ),
            tool_call_id=call_id,
            name=tool_name or "unknown_tool",
            status=status,
        ))
        _record_tool_execution(
            tool_name=(
                tool_name
                if definition is not None
                else "unknown_tool"
            ),
            status=status,
            role=context.role.value,
            started_at=started_at,
        )

    response = {
        "messages": tool_messages,
        "current_turn_tools": [
            *state["current_turn_tools"],
            *executed_tools,
        ],
        "retry_count": retry_count,
    }
    if error_code is not None:
        response.update({
            "error_code": error_code,
            "error_message": error_message,
        })
    return response
