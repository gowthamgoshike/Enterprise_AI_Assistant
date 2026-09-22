from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Response
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
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from apps.api.logging_config import configure_logging
from apps.api.middleware.request_context import request_context_middleware
from apps.api.routes.auth import (
    router as auth_router,
)
from apps.api.routes.chat import (
    router as chat_router,
)
from apps.api.routes.chat_stream import router as chat_stream_router

from apps.api.routes.health import (
    router as health_router,
)


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
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(request_context_middleware)


@app.get("/")
async def root():

    return {
        "service":
            "Enterprise AI Assistant API",
        "version":
            "0.1.0",
    }


@app.get(
    "/metrics",
    tags=["Observability"],
    summary="Get Prometheus metrics",
)
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


app.include_router(
    health_router
)

app.include_router(
    auth_router,
    prefix="/api/v1",
)

app.include_router(
    chat_router
)
app.include_router(chat_stream_router)
