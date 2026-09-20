"""Tests explicitly own their in-memory graph; the API owns PostgreSQL in production."""

from functools import partial

import pytest
from langgraph.checkpoint.memory import InMemorySaver


@pytest.fixture(autouse=True)
def isolated_agent_graph(monkeypatch):
    from ai.graphs import tool_agent_graph
    from apps.api.main import app

    compiled = tool_agent_graph.build_enterprise_tool_graph(InMemorySaver())
    monkeypatch.setattr(app.state, "enterprise_graph", compiled, raising=False)
    monkeypatch.setattr(tool_agent_graph, "enterprise_tool_graph", compiled, raising=False)
    monkeypatch.setattr(
        tool_agent_graph, "run_agent_graph",
        partial(
            tool_agent_graph.run_agent_graph,
            graph=compiled,
            user_id="USER-TEST",
            user_role="ANALYST",
            customer_id=None,
        ),
    )
    return compiled
