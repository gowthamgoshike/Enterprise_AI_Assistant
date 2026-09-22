import asyncio
import time

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from apps.api.schemas.health import DependencyStatus, ReadinessResponse
from core.config import settings
from core.redis import check_redis
from data.database.connection import engine


OLLAMA_BASE_URL = settings.ollama_base_url

router = APIRouter(prefix="/health", tags=["health"])


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
            timeout=settings.health_check_timeout_seconds,
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
            timeout=settings.health_check_timeout_seconds,
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
            timeout=settings.health_check_timeout_seconds
        ) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
    except Exception:
        return DependencyStatus(status="unavailable")
    return DependencyStatus(
        status="healthy",
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )


async def _redis_status() -> DependencyStatus:
    started = time.perf_counter()
    try:
        healthy = await asyncio.wait_for(
            check_redis(), timeout=settings.health_check_timeout_seconds,
        )
    except Exception:
        healthy = False
    return DependencyStatus(
        status="healthy" if healthy else "unavailable",
        latency_ms=(
            round((time.perf_counter() - started) * 1000, 2)
            if healthy else None
        ),
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(request: Request):
    """Report sanitized dependency readiness and graceful degradation."""
    database, checkpoint_store, model, redis = await asyncio.gather(
        _database_status(),
        _checkpoint_status(request),
        _model_status(),
        _redis_status(),
    )
    core_ready = all(
        dependency.status == "healthy"
        for dependency in (database, checkpoint_store, model)
    )
    result = ReadinessResponse(
        status=(
            "ready"
            if core_ready and redis.status == "healthy"
            else "degraded" if core_ready else "not_ready"
        ),
        api=DependencyStatus(status="healthy"),
        database=database,
        checkpoint_store=checkpoint_store,
        model=model,
        redis=redis,
    )
    if core_ready:
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
