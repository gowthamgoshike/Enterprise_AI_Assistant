import json
import logging
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


logger = logging.getLogger(
    __name__
)
RETRYABLE_ERROR_CODES = {
    "model_unavailable",
    "database_unavailable",
    "tool_timeout",
}


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

    try:

        request_id = str(
            uuid.uuid4()
        )

        logger.info(
            "agent_request request_id=%s thread_id=%s message_length=%d",
            request_id,
            payload.session_id,
            len(payload.messages[-1].content),
            extra={
                "request_id": request_id,
                "thread_id": payload.session_id,
                "message_length": len(payload.messages[-1].content),
            },
        )

        start = time.perf_counter()

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
            "chat_request_completed latency=%.3f",
            latency,
        )

        return _chat_response_from_result(
            result,
            request_id=request_id,
            latency_seconds=latency,
        )

    except GraphRecursionError as exc:
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
    request_id = str(uuid.uuid4())
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
        logger.exception(
            "Approval resume failed request_id=%s",
            request_id,
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=500,
            detail="Approval processing failed.",
        ) from exc
