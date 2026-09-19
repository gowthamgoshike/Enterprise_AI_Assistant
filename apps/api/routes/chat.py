import json
import logging
import time
import uuid

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
)
import httpx
from langgraph.errors import GraphRecursionError
from fastapi.responses import (
    StreamingResponse,
)

from apps.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
)

from ai.graphs.tool_agent_graph import (
    run_agent_graph,
)

from apps.api.services.llm_service import (
    OLLAMA_MODEL,
)


logger = logging.getLogger(
    __name__
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
):

    result = await chat(payload, request)

    async def events():
        yield json.dumps({
            "model": result.model,
            "message": {"role": "assistant", "content": result.response},
            "done": False,
        }) + "\n"
        yield json.dumps({
            "model": result.model,
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "tools_used": result.tools_used,
            "route": result.route,
            "route_confidence": result.route_confidence,
        }) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@router.post(
    "",
    response_model=ChatResponse,
)
async def chat(
    payload: ChatRequest,
    request: Request,
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
            thread_id=payload.session_id,
            graph=request.app.state.enterprise_graph,
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

        messages = result["messages"]
        response_text = messages[-1].text
        tools_used = result["current_turn_tools"]

        return ChatResponse(
            request_id=request_id,
            response=response_text,
            model=OLLAMA_MODEL,
            latency_seconds=round(
                latency,
                3,
            ),
            tools_used=tools_used,
            route=result.get("route"),
            route_confidence=result.get("route_confidence"),
            agent_action=(
                "tool"
                if tools_used
                else "respond"
            ),
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
