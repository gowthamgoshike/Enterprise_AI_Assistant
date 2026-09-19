from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from ai.graphs.tool_agent_graph import build_enterprise_tool_graph
from ai.memory.postgres_checkpointer import get_checkpoint_database_url
from fastapi.middleware.cors import (
    CORSMiddleware,
)
import apps.api.logging_config
from apps.api.routes.chat import (
    router as chat_router,
)

from apps.api.routes.health import (
    router as health_router,
)


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


@app.get("/")
async def root():

    return {
        "service":
            "Enterprise AI Assistant API",
        "version":
            "0.1.0",
    }


app.include_router(
    health_router
)

app.include_router(
    chat_router
)
