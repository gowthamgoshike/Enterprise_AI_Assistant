from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver

from ai.memory.postgres_checkpointer import get_checkpoint_database_url
from apps.api import main


def test_checkpoint_url_required(monkeypatch):
    monkeypatch.delenv("LANGGRAPH_DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="LANGGRAPH_DATABASE_URL is not configured"):
        get_checkpoint_database_url()


def test_lifespan_reuses_graph_and_closes_saver(monkeypatch):
    saver = InMemorySaver()
    saver.setup = AsyncMock()
    closed = []
    opened = []

    @asynccontextmanager
    async def connect(url, *, serde):
        assert serde._allowed_msgpack_modules is None
        opened.append(url)
        try:
            yield saver
        finally:
            closed.append(True)

    monkeypatch.setenv("LANGGRAPH_DATABASE_URL", "postgresql://test/checkpoints")
    monkeypatch.setattr(main.AsyncPostgresSaver, "from_conn_string", connect)
    app = FastAPI(lifespan=main.lifespan)
    with TestClient(app) as client:
        compiled = app.state.enterprise_graph
        assert client.get("/openapi.json").status_code == 200
        assert app.state.enterprise_graph is compiled
        saver.setup.assert_awaited_once()
        assert opened == ["postgresql://test/checkpoints"]
        assert not closed
    assert closed == [True]
    assert not hasattr(app.state, "enterprise_graph")
