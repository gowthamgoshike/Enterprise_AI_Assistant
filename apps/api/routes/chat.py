import json
import logging
import re
import time
import uuid

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
import httpx
from langgraph.errors import GraphRecursionError
from fastapi.responses import (
    StreamingResponse,
)

from apps.api.schemas.chat import (
    ApprovalRequest,
    ChatRequest,
    ChatResponse,
)
from apps.api.dependencies.auth import get_current_user
from apps.api.schemas.auth import AuthenticatedUser

from ai.graphs.tool_agent_graph import (
    PendingApprovalNotFoundError,
    resume_agent_graph,
    run_agent_graph,
)

from apps.api.services.llm_service import (
    OLLAMA_MODEL,
)
from core.errors import ModelUnavailableError
from core.observability.context import (
    get_request_id,
    set_request_context,
    set_user_context,
)
from core.observability.metrics import AGENT_ERRORS_TOTAL
from data.database.reliability import is_database_unavailable_error


logger = logging.getLogger(
    __name__
)
RETRYABLE_ERROR_CODES = {
    "model_unavailable",
    "database_unavailable",
    "tool_timeout",
}
TRANSACTION_ID_PATTERN = re.compile(r"\bTX-[A-Za-z0-9][A-Za-z0-9-]*\b", re.I)


def _database_failure_response(
    *,
    request_id: str,
    started_at: float,
    user_message: str | None = None,
    refund_status_uncertain: bool = False,
) -> ChatResponse:
    if refund_status_uncertain:
        message = (
            "The refund service is temporarily unavailable, so I couldn't "
            "confirm the operation's current status."
        )
    elif (
        user_message
        and "status" in user_message.casefold()
        and TRANSACTION_ID_PATTERN.search(user_message)
    ):
        message = (
            "I couldn't verify the current transaction because the "
            "transaction service is temporarily unavailable."
        )
    else:
        message = "The data service is temporarily unavailable."
    return ChatResponse(
        request_id=request_id,
        response=message,
        status="degraded",
        model=OLLAMA_MODEL,
        latency_seconds=round(time.perf_counter() - started_at, 3),
        agent_action="error",
        error_code="database_unavailable",
        retryable=True,
    )


def _get_approval_request(result: dict) -> dict | None:
    interrupts = result.get("__interrupt__") or []
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", interrupts[0])
    return value if isinstance(value, dict) else None


def _chat_response_from_result(
    result: dict,
    *,
    request_id: str,
    latency_seconds: float,
) -> ChatResponse:
    tools_used = result.get("current_turn_tools", [])
    approval_request = _get_approval_request(result)
    if approval_request is not None:
        return ChatResponse(
            request_id=request_id,
            status="approval_required",
            approval_request=approval_request,
            model=OLLAMA_MODEL,
            latency_seconds=round(latency_seconds, 3),
            tools_used=tools_used,
            route=result.get("route"),
            route_confidence=result.get("route_confidence"),
            agent_action="approval",
        )

    messages = result["messages"]
    response_text = messages[-1].text
    error_code = result.get("error_code")
    retryable = error_code in RETRYABLE_ERROR_CODES
    return ChatResponse(
        request_id=request_id,
        response=response_text,
        model=OLLAMA_MODEL,
        latency_seconds=round(latency_seconds, 3),
        tools_used=tools_used,
        route=result.get("route"),
        route_confidence=result.get("route_confidence"),
        status=("degraded" if retryable else "failed") if error_code else "completed",
        agent_action=(
            "error" if error_code else ("tool" if tools_used else "respond")
        ),
        error_code=error_code,
        retryable=retryable,
    )


router = APIRouter(
    prefix="/api/v1/chat",
    tags=["chat"],
)


@router.get("/model/info")
async def model_info():

    return {
        "provider": "ollama",
        "model": OLLAMA_MODEL,
    }


@router.post("/stream")
async def stream_chat(
    payload: ChatRequest,
    request: Request,
    current_user: AuthenticatedUser = Depends(
        get_current_user
    ),
):

    result = await chat(
        payload,
        request,
        current_user,
    )

    async def events():
        yield json.dumps({
            "model": result.model,
            "message": {
                "role": "assistant",
                "content": result.response or "",
            },
            "done": False,
        }) + "\n"
        yield json.dumps({
            "model": result.model,
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "tools_used": result.tools_used,
            "route": result.route,
            "route_confidence": result.route_confidence,
            "status": result.status,
            "approval_request": result.approval_request,
            "error_code": result.error_code,
            "retryable": result.retryable,
        }) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@router.post(
    "",
    response_model=ChatResponse,
)
async def chat(
    payload: ChatRequest,
    request: Request,
    current_user: AuthenticatedUser = Depends(
        get_current_user
    ),
):
    context_request_id = get_request_id()
    request_id = context_request_id or f"REQ-{uuid.uuid4()}"
    if context_request_id is None:
        set_request_context(request_id, current_user.user_id)
    else:
        set_user_context(current_user.user_id)
    start = time.perf_counter()

    try:
        logger.info(
            "agent_request_started",
            extra={
                "event": "agent_request",
                "request_id": request_id,
                "thread_id": payload.session_id,
                "status": "started",
                "role": current_user.role.value,
                "message_length": len(payload.messages[-1].content),
            },
        )

        result = await run_agent_graph(
            user_message=payload.messages[-1].content,
            request_id=request_id,
            thread_id=(
                f"{current_user.user_id}:"
                f"{payload.session_id}"
            ),
            graph=request.app.state.enterprise_graph,
            user_id=current_user.user_id,
            user_role=current_user.role.value,
            customer_id=current_user.customer_id,
        )

        latency = (
            time.perf_counter()
            -
            start
        )

        logger.info(
            "agent_request_completed",
            extra={
                "event": "agent_request",
                "status": "success",
                "role": current_user.role.value,
                "latency_ms": round(latency * 1000, 2),
            },
        )

        return _chat_response_from_result(
            result,
            request_id=request_id,
            latency_seconds=latency,
        )

    except GraphRecursionError as exc:
        AGENT_ERRORS_TOTAL.labels(error_code="agent_step_limit").inc()
        logger.warning("request_id=%s error=agent_step_limit", request_id)
        raise HTTPException(
            status_code=503,
            detail="The assistant reached its execution limit. Please narrow your request.",
        ) from exc

    except httpx.TimeoutException as exc:

        raise HTTPException(
            status_code=504,
            detail="Model request timed out",
        ) from exc

    except ModelUnavailableError as exc:
        AGENT_ERRORS_TOTAL.labels(error_code=exc.code).inc()
        return ChatResponse(
            request_id=request_id,
            response=exc.safe_message,
            status="degraded",
            model=OLLAMA_MODEL,
            latency_seconds=round(time.perf_counter() - start, 3),
            agent_action="error",
            error_code=exc.code,
            retryable=True,
        )

    except httpx.HTTPError as exc:

        raise HTTPException(
            status_code=502,
            detail="Model service unavailable",
        ) from exc

    except Exception as exc:
        if is_database_unavailable_error(exc):
            AGENT_ERRORS_TOTAL.labels(error_code="database_unavailable").inc()
            logger.warning(
                "Agent database unavailable request_id=%s error_type=%s",
                request_id,
                type(exc).__name__,
                extra={
                    "request_id": request_id,
                    "error_code": "database_unavailable",
                },
            )
            return _database_failure_response(
                request_id=request_id,
                started_at=start,
                user_message=payload.messages[-1].content,
            )
        AGENT_ERRORS_TOTAL.labels(error_code="internal_error").inc()
        logger.exception(
            "Agent graph execution failed request_id=%s", request_id,
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=500,
            detail="Agent execution failed.",
        ) from exc


@router.post(
    "/approval",
    response_model=ChatResponse,
)
async def submit_approval(
    payload: ApprovalRequest,
    request: Request,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    context_request_id = get_request_id()
    request_id = context_request_id or f"REQ-{uuid.uuid4()}"
    if context_request_id is None:
        set_request_context(request_id, current_user.user_id)
    else:
        set_user_context(current_user.user_id)
    start = time.perf_counter()
    thread_id = f"{current_user.user_id}:{payload.session_id}"

    try:
        result = await resume_agent_graph(
            thread_id=thread_id,
            approval={
                "approved": payload.approved,
                "comment": payload.comment,
            },
            graph=request.app.state.enterprise_graph,
        )
        return _chat_response_from_result(
            result,
            request_id=result.get("request_id", request_id),
            latency_seconds=time.perf_counter() - start,
        )
    except GraphRecursionError as exc:
        AGENT_ERRORS_TOTAL.labels(error_code="agent_step_limit").inc()
        raise HTTPException(
            status_code=503,
            detail="The assistant reached its execution limit.",
        ) from exc
    except PendingApprovalNotFoundError as exc:
        raise HTTPException(
            status_code=409,
            detail="No pending approval exists for this session.",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Model request timed out",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="Model service unavailable",
        ) from exc
    except Exception as exc:
        if is_database_unavailable_error(exc):
            AGENT_ERRORS_TOTAL.labels(error_code="database_unavailable").inc()
            logger.warning(
                "Approval database unavailable request_id=%s error_type=%s",
                request_id,
                type(exc).__name__,
                extra={
                    "request_id": request_id,
                    "error_code": "database_unavailable",
                },
            )
            return _database_failure_response(
                request_id=request_id,
                started_at=start,
                refund_status_uncertain=True,
            )
        AGENT_ERRORS_TOTAL.labels(error_code="internal_error").inc()
        logger.exception(
            "Approval resume failed request_id=%s",
            request_id,
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=500,
            detail="Approval processing failed.",
        ) from exc
