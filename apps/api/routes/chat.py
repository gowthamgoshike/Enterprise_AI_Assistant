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

from apps.api.services.agent_service import (
    process_user_request,
)

from apps.api.services.llm_service import (
    OLLAMA_MODEL,
    stream_chat_with_model,
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

    messages = [
        message.model_dump()
        for message in request.messages
    ]

    return StreamingResponse(
        stream_chat_with_model(
            messages
        ),
        media_type="application/x-ndjson",
    )


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

        result = await process_user_request(
            request.messages[-1].content,
            request_id,
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
            tool_used=result.get("tool_used"),
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