"""Adapt LangGraph streaming into a small, safe client event protocol."""

import logging
import time
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessageChunk
from langgraph.errors import GraphRecursionError

from ai.graphs.tool_agent_graph import GRAPH_RECURSION_LIMIT, initial_agent_state
from apps.api.routes.chat import _chat_response_from_result
from apps.api.schemas.auth import AuthenticatedUser
from apps.api.schemas.streaming import StreamEvent
from core.errors import ModelUnavailableError
from data.database.reliability import is_database_unavailable_error


logger = logging.getLogger(__name__)

TOOL_DISPLAY_NAMES = {
    "get_transaction": "Checking transaction...",
    "get_top_products": "Analyzing top products...",
    "get_revenue_summary": "Calculating revenue...",
    "get_transaction_metrics": "Analyzing transaction metrics...",
    "get_my_transactions": "Loading your transactions...",
    "propose_refund": "Preparing refund request...",
}
APPROVAL_ACTION_FIELDS = (
    "action_id",
    "action_type",
    "transaction_id",
    "amount_cents",
    "currency",
    "reason",
)
SAFE_ERROR_MESSAGES = {
    "model_unavailable": "The AI service is temporarily unavailable.",
    "database_unavailable": "The data service is temporarily unavailable.",
    "tool_timeout": "The operation timed out.",
    "agent_step_limit": "The assistant reached its execution limit.",
    "permission_denied": "You are not authorized to perform this action.",
    "internal_error": "The request could not be completed.",
}


def translate_graph_event(event: object) -> StreamEvent | None:
    """Keep graph-specific event shapes and content filtering in one place."""
    if not isinstance(event, tuple) or len(event) != 2:
        return None
    mode, payload = event
    if mode == "messages":
        if not isinstance(payload, tuple) or len(payload) != 2:
            return None
        chunk, metadata = payload
        if not isinstance(chunk, AIMessageChunk) or not isinstance(metadata, dict):
            return None
        if metadata.get("langgraph_node") != "agent":
            return None
        if chunk.tool_call_chunks or chunk.tool_calls:
            return None
        if isinstance(chunk.content, str):
            text = chunk.content
        elif isinstance(chunk.content, list):
            text = "".join(
                part.get("text", "")
                for part in chunk.content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            return None
        return StreamEvent(type="token", data={"text": text}) if text else None
    if mode == "custom" and isinstance(payload, dict):
        event_type = payload.get("type")
        tool = payload.get("tool")
        if event_type in {"tool_start", "tool_end"} and tool in TOOL_DISPLAY_NAMES:
            return StreamEvent(
                type=event_type,
                data={"tool": tool, "message": TOOL_DISPLAY_NAMES[tool]},
            )
    if mode == "updates" and isinstance(payload, dict):
        if "secure_tools" in payload:
            return StreamEvent(
                type="status",
                data={"message": "Generating response..."},
            )
    return None


def _safe_approval(approval_request: dict) -> dict:
    action = approval_request.get("action")
    if not isinstance(action, dict):
        action = {}
    return {
        key: action[key]
        for key in APPROVAL_ACTION_FIELDS
        if key in action
    }


def _error_events(code: str) -> tuple[StreamEvent, StreamEvent]:
    safe_code = code if code in SAFE_ERROR_MESSAGES else "internal_error"
    return (
        StreamEvent(
            type="error",
            data={
                "code": safe_code,
                "message": SAFE_ERROR_MESSAGES[safe_code],
                "retryable": safe_code in {
                    "model_unavailable", "database_unavailable", "tool_timeout"
                },
            },
        ),
        StreamEvent(type="done", data={"status": "degraded" if safe_code != "internal_error" else "failed", "error_code": safe_code}),
    )


async def stream_agent_response(
    *,
    user_message: str,
    current_user: AuthenticatedUser,
    session_id: str,
    request_id: str,
    graph,
) -> AsyncIterator[StreamEvent]:
    """Run one trusted graph turn and emit only client-safe progress/output."""
    started = time.perf_counter()
    config = {
        "configurable": {"thread_id": f"{current_user.user_id}:{session_id}"},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }
    initial_state = initial_agent_state(
        user_message=user_message,
        request_id=request_id,
        user_id=current_user.user_id,
        user_role=current_user.role.value,
        customer_id=current_user.customer_id,
    )
    yield StreamEvent(
        type="status", data={"message": "Processing your request..."}
    )
    streamed_text = ""
    try:
        async for raw_event in graph.astream(
            initial_state,
            config=config,
            stream_mode=["messages", "updates", "custom"],
        ):
            event = translate_graph_event(raw_event)
            if event is None:
                continue
            if event.type == "token":
                streamed_text += event.data["text"]
            yield event

        snapshot = await graph.aget_state(config)
        result = dict(snapshot.values)
        interrupts = [
            interrupt
            for task in snapshot.tasks
            for interrupt in task.interrupts
        ]
        if interrupts:
            result["__interrupt__"] = interrupts
        response = _chat_response_from_result(
            result,
            request_id=request_id,
            latency_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        if is_database_unavailable_error(exc):
            code = "database_unavailable"
        elif isinstance(exc, ModelUnavailableError):
            code = "model_unavailable"
        elif isinstance(exc, GraphRecursionError):
            code = "agent_step_limit"
        else:
            code = "internal_error"
        logger.warning(
            "stream_agent_failed",
            extra={"event": "agent_request", "error_code": code, "status": "failure"},
        )
        for event in _error_events(code):
            yield event
        return

    if response.status == "approval_required":
        yield StreamEvent(
            type="approval_required",
            data=_safe_approval(response.approval_request or {}),
        )
    elif response.error_code:
        yield _error_events(response.error_code)[0]
    elif response.response and not streamed_text:
        # Some injected/custom models emit only a completed AIMessage.
        yield StreamEvent(type="token", data={"text": response.response})
    elif response.response and response.response.startswith(streamed_text):
        remainder = response.response[len(streamed_text):]
        if remainder:
            yield StreamEvent(type="token", data={"text": remainder})

    yield StreamEvent(
        type="done",
        data={
            "request_id": response.request_id,
            "status": response.status,
            "response": response.response,
            "model": response.model,
            "latency_seconds": response.latency_seconds,
            "tools_used": response.tools_used,
            "route": response.route,
            "route_confidence": response.route_confidence,
            "error_code": response.error_code,
            "retryable": response.retryable,
        },
    )
