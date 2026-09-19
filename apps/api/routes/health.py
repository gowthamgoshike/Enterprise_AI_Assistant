import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import os

import httpx

from dotenv import load_dotenv


load_dotenv()


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL"
)
router = APIRouter(
    prefix="/api/v1/health",
    tags=["health"],
)


@router.get("")
async def health():

    return {
        "status": "ok",
        "service": "enterprise-ai-api",
    }

@router.get("/model")
async def model_health():

    try:

        async with httpx.AsyncClient(
            timeout=5.0
        ) as client:

            response = await client.get(
                f"{OLLAMA_BASE_URL}/api/tags"
            )

            response.raise_for_status()

        return {
            "status": "ok",
            "model_service": "ollama",
        }

    except Exception:

        return {
            "status": "unavailable",
            "model_service": "ollama",
        }


@router.get("/checkpoint")
async def checkpoint_health(request: Request):
    """Probe checkpoint storage without returning conversation data or errors."""
    try:
        saver = request.app.state.checkpoint_store
        await asyncio.wait_for(
            saver.aget_tuple({"configurable": {"thread_id": "__health_probe__"}}),
            timeout=5.0,
        )
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"api": "healthy", "checkpoint_store": "unavailable"},
        )
    return {"api": "healthy", "checkpoint_store": "healthy"}
