"""Streaming events expose progress and answer text, never graph internals."""

import asyncio
from types import SimpleNamespace

import psycopg
from langchain_core.messages import AIMessage, AIMessageChunk

from apps.api.schemas.auth import AuthenticatedUser
from apps.api.services.streaming_service import (
    stream_agent_response,
    translate_graph_event,
)
from services.auth_service.models import UserRole


USER = AuthenticatedUser(
    user_id="USER-1",
    email="admin@example.com",
    role=UserRole.ADMIN,
    customer_id=None,
)


def test_adapter_filters_tool_calls_and_non_agent_messages():
    assert translate_graph_event((
        "messages",
        (AIMessageChunk(content="Hello"), {"langgraph_node": "agent"}),
    )).data == {"text": "Hello"}
    assert translate_graph_event((
        "messages",
        (AIMessageChunk(content="hidden"), {"langgraph_node": "secure_tools"}),
    )) is None
    assert translate_graph_event((
        "messages",
        (
            AIMessageChunk(content="", tool_call_chunks=[{
                "name": "get_transaction",
                "args": '{"transaction_id":"TX-SECRET"}',
                "id": "call-1",
                "index": 0,
            }]),
            {"langgraph_node": "agent"},
        ),
    )) is None
    assert translate_graph_event((
        "custom", {"type": "tool_start", "tool": "unknown_tool"}
    )) is None
    event = translate_graph_event((
        "custom", {"type": "tool_start", "tool": "get_transaction"}
    ))
    assert event.type == "tool_start"
    assert event.data["message"] == "Checking transaction..."


class FakeGraph:
    def __init__(self, *, events=(), values=None, interrupts=(), error=None):
        self.events = events
        self.values = values or {
            "messages": [AIMessage(content="Done.")],
            "current_turn_tools": [],
        }
        self.interrupts = interrupts
        self.error = error
        self.input_state = None
        self.config = None

    async def astream(self, input_state, *, config, stream_mode):
        self.input_state = input_state
        self.config = config
        assert stream_mode == ["messages", "updates", "custom"]
        for event in self.events:
            yield event
        if self.error:
            raise self.error

    async def aget_state(self, config):
        assert config == self.config
        return SimpleNamespace(
            values=self.values,
            tasks=[SimpleNamespace(interrupts=self.interrupts)],
        )


def collect(graph):
    async def run():
        return [event async for event in stream_agent_response(
            user_message="Check the transaction",
            current_user=USER,
            session_id="SESSION-1",
            request_id="REQ-1",
            graph=graph,
        )]
    return asyncio.run(run())


def test_streaming_service_uses_trusted_thread_and_safe_progress():
    graph = FakeGraph(
        events=[
            ("custom", {"type": "tool_start", "tool": "get_transaction"}),
            ("custom", {"type": "tool_end", "tool": "get_transaction"}),
            ("messages", (
                AIMessageChunk(content="Done."),
                {"langgraph_node": "agent"},
            )),
        ],
        values={
            "messages": [AIMessage(content="Done.")],
            "current_turn_tools": ["get_transaction"],
        },
    )

    events = collect(graph)

    assert [event.type for event in events] == [
        "status", "tool_start", "tool_end", "token", "done"
    ]
    assert graph.config["configurable"]["thread_id"] == "USER-1:SESSION-1"
    assert graph.input_state["user_role"] == "ADMIN"
    assert graph.input_state["user_id"] == "USER-1"
    assert events[-1].data["response"] == "Done."


def test_approval_payload_is_allowlisted():
    action = {
        "action_id": "ACTION-1",
        "action_type": "refund",
        "transaction_id": "TX-1",
        "amount_cents": 7500,
        "currency": "USD",
        "reason": "duplicate charge",
        "requested_by": "USER-SECRET",
    }
    graph = FakeGraph(
        values={
            "messages": [AIMessage(content="")],
            "current_turn_tools": ["propose_refund"],
        },
        interrupts=[SimpleNamespace(value={
            "type": "action_approval",
            "action": action,
        })],
    )

    events = collect(graph)

    assert events[-2].type == "approval_required"
    assert events[-2].data["transaction_id"] == "TX-1"
    assert "requested_by" not in events[-2].data
    assert events[-1].data["status"] == "approval_required"


def test_database_failure_emits_safe_error_then_done():
    graph = FakeGraph(error=psycopg.OperationalError("private database host"))

    events = collect(graph)

    assert [event.type for event in events] == ["status", "error", "done"]
    assert events[1].data == {
        "code": "database_unavailable",
        "message": "The data service is temporarily unavailable.",
        "retryable": True,
    }
    assert "private database host" not in str([event.model_dump() for event in events])
