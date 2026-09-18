import json
import logging
import time
import uuid

from fastapi import (
    APIRouter,
    HTTPException,
)
import httpx
from fastapi.responses import (
    StreamingResponse,
)

from apps.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
)

from ai.graphs.enterprise_graph import (
    run_enterprise_graph,
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
    request: ChatRequest,
):

    result = await chat(request)

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
    request: ChatRequest,
):

    try:

        request_id = str(
            uuid.uuid4()
        )

        logger.info(
            "request_id=%s chat_started",
            request_id,
        )

        start = time.perf_counter()

        result = await run_enterprise_graph(
            user_message=request.messages[-1].content,
            request_id=request_id,
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

        return ChatResponse(
            request_id=request_id,
            response=result["response"],
            model=OLLAMA_MODEL,
            latency_seconds=round(
                latency,
                3,
            ),
            tools_used=result.get("tools_used", []),
            route=result.get("route"),
            route_confidence=result.get("route_confidence"),
            agent_action=(
                "tool"
                if result.get("tools_used", [])
                else "respond"
            ),
        )

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

        raise HTTPException(
            status_code=500,
            detail="Unexpected server error",
        ) from exc
