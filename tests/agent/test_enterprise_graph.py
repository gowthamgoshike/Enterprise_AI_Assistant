"""Live graph integration tests requiring Ollama and analytics data services."""

import uuid

import pytest

from ai.graphs.enterprise_graph import run_enterprise_graph


@pytest.mark.asyncio(loop_scope="module")
async def test_general_route():
    result = await run_enterprise_graph(
        user_message="Explain Kafka.",
        request_id=str(uuid.uuid4()),
    )

    assert result["route"] == "general"
    assert result["response"] is not None


@pytest.mark.asyncio(loop_scope="module")
async def test_analytics_route():
    result = await run_enterprise_graph(
        user_message="Show the top 5 products.",
        request_id=str(uuid.uuid4()),
    )

    assert result["route"] == "analytics"
