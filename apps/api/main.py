from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from ai.graphs.tool_agent_graph import build_enterprise_tool_graph
from ai.memory.postgres_checkpointer import get_checkpoint_database_url
from core.async_database import async_engine
from core.config import settings
from core.redis import redis_client
from fastapi.middleware.cors import (
    CORSMiddleware,
)
from apps.api.logging_config import configure_logging
from apps.api.error_handlers import register_exception_handlers
from apps.api.middleware.request_id import RequestIDMiddleware
from apps.api.middleware.request_limits import RequestSizeLimitMiddleware
from apps.api.middleware.security_headers import SecurityHeadersMiddleware
from apps.api.routes.metrics import router as metrics_router
from apps.api.routes.router import api_router


configure_logging(level=getattr(logging, settings.log_level.upper(), logging.INFO))


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPostgresSaver.from_conn_string(
        get_checkpoint_database_url(),
        serde=JsonPlusSerializer(allowed_msgpack_modules=None),
    ) as checkpointer:
        await checkpointer.setup()
        app.state.enterprise_graph = build_enterprise_tool_graph(checkpointer)
        app.state.checkpoint_store = checkpointer
        try:
            yield
        finally:
            del app.state.enterprise_graph
            del app.state.checkpoint_store
            await async_engine.dispose()
            await redis_client.aclose()


app = FastAPI(
    title="Enterprise AI Assistant API",
    version="0.1.0",
    lifespan=lifespan,
    debug=False,
    docs_url=None if settings.app_env == "production" else "/docs",
    redoc_url=None if settings.app_env == "production" else "/redoc",
    openapi_url=None if settings.app_env == "production" else "/openapi.json",
)


register_exception_handlers(app)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_request_bytes=settings.max_request_bytes,
)
app.add_middleware(
    SecurityHeadersMiddleware,
    enable_hsts=settings.app_env == "production",
)
app.add_middleware(RequestIDMiddleware)


@app.get("/")
async def root():

    return {
        "service":
            "Enterprise AI Assistant API",
        "version":
            "0.1.0",
    }


app.include_router(api_router, prefix="/api/v1")
app.include_router(metrics_router)
