import asyncio
import os
import time

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from apps.api.schemas.health import DependencyStatus, ReadinessResponse
from core.config import HEALTH_CHECK_TIMEOUT_SECONDS
from data.database.connection import engine


load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("")
async def health():
    """Backwards-compatible shallow health endpoint."""
    return {"status": "ok", "service": "enterprise-ai-api"}


@router.get("/live")
async def liveness():
    """Report process liveness without probing downstream dependencies."""
    return {"status": "alive"}


async def _database_status() -> DependencyStatus:
    started = time.perf_counter()

    def probe() -> None:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    try:
        await asyncio.wait_for(
            asyncio.to_thread(probe),
            timeout=HEALTH_CHECK_TIMEOUT_SECONDS,
        )
    except Exception:
        return DependencyStatus(status="unavailable")
    return DependencyStatus(
        status="healthy",
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )


async def _checkpoint_status(request: Request) -> DependencyStatus:
    started = time.perf_counter()
    try:
        saver = request.app.state.checkpoint_store
        await asyncio.wait_for(
            saver.aget_tuple({
                "configurable": {"thread_id": "__health_probe__"}
            }),
            timeout=HEALTH_CHECK_TIMEOUT_SECONDS,
        )
    except Exception:
        return DependencyStatus(status="unavailable")
    return DependencyStatus(
        status="healthy",
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )


async def _model_status() -> DependencyStatus:
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(
            timeout=HEALTH_CHECK_TIMEOUT_SECONDS
        ) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
    except Exception:
        return DependencyStatus(status="unavailable")
    return DependencyStatus(
        status="healthy",
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(request: Request):
    """Report sanitized dependency readiness and graceful degradation."""
    database, checkpoint_store, model = await asyncio.gather(
        _database_status(),
        _checkpoint_status(request),
        _model_status(),
    )
    ready = all(
        dependency.status == "healthy"
        for dependency in (database, checkpoint_store, model)
    )
    result = ReadinessResponse(
        status="ready" if ready else "degraded",
        api=DependencyStatus(status="healthy"),
        database=database,
        checkpoint_store=checkpoint_store,
        model=model,
    )
    if ready:
        return result
    return JSONResponse(status_code=503, content=result.model_dump(mode="json"))


@router.get("/model")
async def model_health():
    status = await _model_status()
    return {
        "status": "ok" if status.status == "healthy" else "unavailable",
        "model_service": "ollama",
        "latency_ms": status.latency_ms,
    }


@router.get("/checkpoint")
async def checkpoint_health(request: Request):
    """Probe checkpoint storage without returning conversation data or errors."""
    status = await _checkpoint_status(request)
    content = {
        "api": "healthy",
        "checkpoint_store": status.status,
    }
    if status.status == "unavailable":
        return JSONResponse(status_code=503, content=content)
    return content
