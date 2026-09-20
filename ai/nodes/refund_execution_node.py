import json
import asyncio
import logging

from langchain_core.messages import ToolMessage

from ai.schemas.graph_state import EnterpriseAgentState
from ai.schemas.tool_result import ToolResult
from core.config import DATABASE_MAX_RETRIES
from core.errors import DatabaseUnavailableError
from core.retry import retry_async
from core.observability.metrics import (
    AUTHORIZATION_DENIALS_TOTAL,
    REFUND_EXECUTIONS_TOTAL,
)
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


logger = logging.getLogger(__name__)


def _failed_execution(
    state: EnterpriseAgentState,
    *,
    error: str,
    message: str,
    retryable: bool = False,
    retry_count: int | None = None,
) -> dict:
    metric_result = "denied" if error == "permission_denied" else "failed"
    REFUND_EXECUTIONS_TOTAL.labels(result=metric_result).inc()
    log = logger.warning if metric_result == "denied" else logger.error
    log(
        "refund_execution_failed",
        extra={
            "event": "refund_execution",
            "status": "failure",
            "result": metric_result,
            "error_code": error,
        },
    )
    tool_result = ToolResult.failure(
        error_code=error,
        message=message,
        retryable=retryable,
    )
    return {
        "messages": [ToolMessage(
            content=json.dumps(tool_result.model_dump(mode="json")),
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
        "error_code": error,
        "error_message": message,
        "retry_count": (
            state.get("retry_count", 0)
            if retry_count is None
            else retry_count
        ),
    }


async def execute_refund_node(state: EnterpriseAgentState) -> dict:
    """Execute an approved refund using the checkpointed proposal identity."""
    retries = 0

    def record_retry(_attempt: int, _delay: float, _error: Exception) -> None:
        nonlocal retries
        retries += 1

    try:
        require_permission(
            UserRole(state["user_role"]),
            Permission.EXECUTE_REFUND,
        )
        proposal = state["pending_action"]
        # Every attempt reuses the exact checkpointed action_id. The database
        # uniqueness constraint therefore makes an uncertain post-commit retry
        # return already_completed instead of creating a second refund.
        result = await retry_async(
            lambda: asyncio.to_thread(
                execute_refund,
                action_id=proposal["action_id"],
                transaction_id=proposal["transaction_id"],
                reason=proposal["reason"],
                requested_by=proposal["requested_by"],
                approved_by=state["user_id"],
                request_id=state["request_id"],
            ),
            max_retries=DATABASE_MAX_RETRIES,
            retry_on=(DatabaseUnavailableError,),
            operation_name="execute_refund",
            request_id=state["request_id"],
            on_retry=record_retry,
        )
    except AuthorizationError:
        AUTHORIZATION_DENIALS_TOTAL.labels(
            role=state["user_role"],
            capability="execute_refund",
        ).inc()
        return _failed_execution(
            state,
            error="permission_denied",
            message="Refund execution permission denied",
        )
    except TransactionNotFoundError:
        return _failed_execution(
            state,
            error="transaction_not_found",
            message="The transaction was not found.",
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
    except DatabaseUnavailableError as exc:
        return _failed_execution(
            state,
            error=exc.code,
            message=(
                "The refund service is temporarily unavailable. "
                "The operation's current status could not be confirmed."
            ),
            retryable=True,
            retry_count=state.get("retry_count", 0) + retries,
        )

    tool_result = ToolResult.success({
        "refund": result.model_dump(mode="json"),
    })
    REFUND_EXECUTIONS_TOTAL.labels(result=result.status).inc()
    logger.info(
        "refund_execution_completed",
        extra={
            "event": "refund_execution",
            "status": "success",
            "result": result.status,
        },
    )
    return {
        "messages": [ToolMessage(
            content=json.dumps(tool_result.model_dump(mode="json"), default=str),
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
        "error_code": None,
        "error_message": None,
        "retry_count": state.get("retry_count", 0) + retries,
    }
