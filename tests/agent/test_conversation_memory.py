"""Live checks for checkpointed conversation continuity and isolation."""

import uuid

import pytest
from langchain_core.messages import HumanMessage

from ai.graphs import tool_agent_graph as graph


@pytest.mark.asyncio(loop_scope="module")
async def test_same_thread_keeps_context():
    thread_id = str(uuid.uuid4())

    await graph.run_agent_graph(
        user_message="Explain Kafka.",
        request_id=str(uuid.uuid4()),
        thread_id=thread_id,
    )

    result = await graph.run_agent_graph(
        user_message="What technology was I asking about?",
        request_id=str(uuid.uuid4()),
        thread_id=thread_id,
    )

    response = result["messages"][-1].content.lower()
    assert "kafka" in response


@pytest.mark.asyncio(loop_scope="module")
async def test_threads_are_isolated():
    thread_a = str(uuid.uuid4())
    thread_b = str(uuid.uuid4())

    await graph.run_agent_graph(
        user_message="Remember that we are discussing Kafka.",
        request_id=str(uuid.uuid4()),
        thread_id=thread_a,
    )

    result = await graph.run_agent_graph(
        user_message="What technology were we discussing?",
        request_id=str(uuid.uuid4()),
        thread_id=thread_b,
    )

    human_messages = [
        message.content
        for message in result["messages"]
        if isinstance(message, HumanMessage)
    ]
    assert human_messages == ["What technology were we discussing?"]
