"""Authenticated Server-Sent Events for one checkpointed chat turn."""

import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent

from apps.api.dependencies.rate_limit import get_rate_limited_chat_user
from apps.api.schemas.auth import AuthenticatedUser
from apps.api.schemas.chat import ChatRequest
from apps.api.services.streaming_service import stream_agent_response
from core.observability.context import (
    get_request_id,
    set_request_context,
    set_user_context,
)


router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("/stream", response_class=EventSourceResponse)
async def stream_chat(
    payload: ChatRequest,
    request: Request,
    current_user: AuthenticatedUser = Depends(get_rate_limited_chat_user),
) -> AsyncIterator[ServerSentEvent]:
    request_id = get_request_id()
    if request_id is None:
        request_id = f"REQ-{uuid.uuid4()}"
        set_request_context(request_id, current_user.user_id)
    else:
        set_user_context(current_user.user_id)

    async for event in stream_agent_response(
        user_message=payload.messages[-1].content,
        current_user=current_user,
        session_id=payload.session_id,
        request_id=request_id,
        graph=request.app.state.enterprise_graph,
    ):
        yield ServerSentEvent(event=event.type, data=event.model_dump(mode="json"))
