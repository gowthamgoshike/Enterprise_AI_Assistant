import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import psycopg
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from apps.api.dependencies.auth import get_current_user
from apps.api.main import app
from apps.api.routes import chat as chat_routes
from apps.api.schemas.auth import AuthenticatedUser
from services.auth_service.models import UserRole


client = TestClient(app)


def parse_sse(response):
    events = []
    event_type = None
    for line in response.text.splitlines():
        if line.startswith("event:"):
            event_type = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            payload = json.loads(line.removeprefix("data:").strip())
            assert payload["type"] == event_type
            events.append(payload)
    return events


@pytest.fixture(autouse=True)
def authenticated_chat_user():
    user = AuthenticatedUser(
        user_id="USER-TEST",
        email="customer@example.com",
        role=UserRole.CUSTOMER,
        customer_id="CUST-001",
    )
    app.dependency_overrides[get_current_user] = lambda: user
    yield user
    app.dependency_overrides.pop(get_current_user, None)


def test_graph_failure_logs_metadata_and_returns_safe_error(monkeypatch, caplog):
    from unittest.mock import AsyncMock

    monkeypatch.setattr(chat_routes, "run_agent_graph", AsyncMock(
        side_effect=RuntimeError("checkpoint failure"),
    ))
    with caplog.at_level(logging.INFO, logger=chat_routes.__name__):
        response = client.post("/api/v1/chat", json={
            "messages": [{"role": "user", "content": "sensitive question"}],
            "session_id": "logging-test",
        })
    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "agent_execution_failed",
        "message": "The assistant could not complete the request.",
        "retryable": False,
    }
    record = next(r for r in caplog.records if r.msg.startswith("agent_request"))
    assert record.thread_id == "logging-test"
    assert record.message_length == len("sensitive question")
    assert record.request_id
    assert "sensitive question" not in caplog.text
    assert any(r.exc_info for r in caplog.records)


def test_checkpoint_database_outage_returns_degraded_response(monkeypatch, caplog):
    monkeypatch.setattr(chat_routes, "run_agent_graph", AsyncMock(
        side_effect=psycopg.OperationalError("private connection details"),
    ))

    with caplog.at_level(logging.WARNING, logger=chat_routes.__name__):
        response = client.post("/api/v1/chat", json={
            "messages": [{
                "role": "user",
                "content": "What's the status of TX-A?",
            }],
            "session_id": "database-outage",
        })

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["error_code"] == "database_unavailable"
    assert response.json()["retryable"] is True
    assert "couldn't verify the current transaction" in response.json()["response"]
    assert "private connection details" not in response.text
    assert "private connection details" not in caplog.text


def test_chat_endpoints_use_graph(monkeypatch):
    calls = []
    tool_names = ["get_revenue_summary", "get_transaction_metrics", "get_top_products"]

    async def fake_graph(
        *,
        user_message,
        request_id,
        thread_id,
        graph,
        user_id,
        user_role,
        customer_id,
    ):
        calls.append({
            "user_message": user_message,
            "request_id": request_id,
            "thread_id": thread_id,
            "graph": graph,
            "user_id": user_id,
            "user_role": user_role,
            "customer_id": customer_id,
        })
        return {"messages": [
            AIMessage(content="", tool_calls=[
                {"name": name, "args": {}, "id": f"call-{i}"}
                for i, name in enumerate(tool_names)
            ]),
            AIMessage(content="Revenue is 100."),
        ], "route": None, "current_turn_tools": tool_names}

    monkeypatch.setattr(chat_routes, "run_agent_graph", fake_graph)
    monkeypatch.setattr(chat_routes, "OLLAMA_MODEL", "test-model")
    messages = [
        {"role": "user", "content": "Show revenue."},
        {"role": "assistant", "content": "Revenue is 100."},
        {"role": "user", "content": "How is the business doing?"},
    ]
    response = client.post(
        "/api/v1/chat",
        json={"messages": messages, "session_id": "conversation-123"},
    )
    assert response.status_code == 200
    assert response.json()["request_id"] == calls[0]["request_id"]
    assert response.headers["X-Request-ID"] == response.json()["request_id"]
    assert response.json()["response"] == "Revenue is 100."
    assert response.json()["model"] == "test-model"
    assert response.json()["latency_seconds"] >= 0
    assert response.json()["tools_used"] == tool_names
    assert "tool_used" not in response.json()
    assert response.json()["agent_action"] == "tool"
    assert response.json()["route"] is None
    assert response.json()["route_confidence"] is None

    assert len(calls) == 1
    assert all(call["user_message"] == messages[-1]["content"] for call in calls)
    assert all(call["request_id"] for call in calls)
    assert all(call["thread_id"] == "USER-TEST:conversation-123" for call in calls)
    assert all(call["graph"] is app.state.enterprise_graph for call in calls)
    assert all(call["user_id"] == "USER-TEST" for call in calls)
    assert all(call["user_role"] == "CUSTOMER" for call in calls)
    assert all(call["customer_id"] == "CUST-001" for call in calls)


def test_empty_chat_history_is_rejected():
    for endpoint in ("/api/v1/chat", "/api/v1/chat/stream"):
        assert client.post(
            endpoint,
            json={"messages": [], "session_id": "conversation-123"},
        ).status_code == 422


def test_chat_general_answer_uses_agent_loop(monkeypatch):
    from unittest.mock import AsyncMock
    from ai.graphs import tool_agent_graph

    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(content="Please clarify your request.")
    monkeypatch.setattr(tool_agent_graph, "model_with_tools", model)
    for endpoint in ["/api/v1/chat", "/api/v1/chat/stream"]:
        response = client.post(endpoint, json={"messages": [
            {"role": "user", "content": "Show me what's going on."}
        ], "session_id": f"general-{endpoint}"})
        assert response.status_code == 200
        if endpoint.endswith("/stream"):
            assert response.headers["content-type"].startswith("text/event-stream")
            events = parse_sse(response)
            assert events[0]["type"] == "status"
            assert events[-1]["type"] == "done"
            assert "".join(
                event["data"]["text"]
                for event in events
                if event["type"] == "token"
            ) == "Please clarify your request."
            result = events[-1]["data"]
        else:
            result = response.json()
            assert result["agent_action"] == "respond"
            assert result["response"] == "Please clarify your request."
        assert result["route"] is None
        assert result["route_confidence"] is None
        assert result["tools_used"] == []


def test_stream_endpoint_emits_model_chunks_before_done(monkeypatch):
    from ai.graphs import tool_agent_graph

    monkeypatch.setattr(
        tool_agent_graph,
        "model_with_tools",
        FakeListChatModel(responses=["Kafka streams events."]),
    )

    response = client.post("/api/v1/chat/stream", json={
        "messages": [{"role": "user", "content": "Explain Kafka briefly."}],
        "session_id": "stream-tokens",
    })

    assert response.status_code == 200
    events = parse_sse(response)
    tokens = [event["data"]["text"] for event in events if event["type"] == "token"]
    assert "".join(tokens) == "Kafka streams events."
    assert len(tokens) > 1
    assert events[-1]["type"] == "done"


def test_stream_endpoint_emits_safe_tool_progress(monkeypatch):
    from ai.graphs import tool_agent_graph
    from ai.nodes import secure_tool_node
    from ai.tools.secure_registry import ToolDefinition
    from services.auth_service.permissions import Permission

    model = AsyncMock()
    model.ainvoke.side_effect = [
        AIMessage(content="", tool_calls=[{
            "name": "get_transaction",
            "args": {"transaction_id": "TX-PRIVATE"},
            "id": "call-1",
        }]),
        AIMessage(content="The transaction was not found."),
    ]
    monkeypatch.setattr(tool_agent_graph, "model_with_tools", model)

    async def lookup(*, context, transaction_id):
        assert context.customer_id == "CUST-001"
        assert transaction_id == "TX-PRIVATE"
        return {"found": False, "transaction": None}

    monkeypatch.setitem(
        secure_tool_node.SECURE_TOOL_REGISTRY,
        "get_transaction",
        ToolDefinition(
            name="get_transaction",
            function=lookup,
            permission=Permission.READ_OWN_TRANSACTIONS,
        ),
    )

    response = client.post("/api/v1/chat/stream", json={
        "messages": [{"role": "user", "content": "Find a transaction"}],
        "session_id": "stream-tool-progress",
    })

    assert response.status_code == 200
    events = parse_sse(response)
    assert [event["type"] for event in events].count("tool_start") == 1
    assert [event["type"] for event in events].count("tool_end") == 1
    assert next(
        event["data"]["message"]
        for event in events
        if event["type"] == "tool_start"
    ) == "Checking transaction..."
    assert "TX-PRIVATE" not in response.text
    assert events[-1]["data"]["tools_used"] == ["get_transaction"]


def test_stream_endpoint_preserves_refund_approval_interrupt(monkeypatch):
    from ai.graphs import tool_agent_graph
    from ai.nodes import secure_tool_node
    from ai.tools.secure_registry import ToolDefinition
    from services.auth_service.permissions import Permission

    admin = AuthenticatedUser(
        user_id="USER-ADMIN",
        email="admin@example.com",
        role=UserRole.ADMIN,
        customer_id=None,
    )
    app.dependency_overrides[get_current_user] = lambda: admin
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(content="", tool_calls=[{
        "name": "propose_refund",
        "args": {"transaction_id": "TX-123", "reason": "duplicate charge"},
        "id": "refund-call",
    }])
    monkeypatch.setattr(tool_agent_graph, "model_with_tools", model)

    async def propose(*, context, transaction_id, reason):
        return {
            "action_id": "ACTION-123",
            "action_type": "refund",
            "transaction_id": transaction_id,
            "amount_cents": 2500,
            "currency": "USD",
            "reason": reason,
            "requested_by": context.user_id,
            "internal_secret": "never-stream-this",
        }

    monkeypatch.setitem(
        secure_tool_node.SECURE_TOOL_REGISTRY,
        "propose_refund",
        ToolDefinition(
            name="propose_refund",
            function=propose,
            permission=Permission.PROPOSE_REFUND,
            requires_approval=True,
        ),
    )

    response = client.post("/api/v1/chat/stream", json={
        "messages": [{"role": "user", "content": "Refund TX-123"}],
        "session_id": "stream-approval",
    })

    assert response.status_code == 200
    events = parse_sse(response)
    assert any(event["type"] == "tool_start" for event in events)
    approval = next(
        event["data"] for event in events
        if event["type"] == "approval_required"
    )
    assert approval["action_id"] == "ACTION-123"
    assert approval["transaction_id"] == "TX-123"
    assert events[-1]["data"]["status"] == "approval_required"
    assert "never-stream-this" not in response.text
    assert "requested_by" not in response.text


def test_chat_step_limit_returns_safe_error(monkeypatch):
    from unittest.mock import AsyncMock
    from langgraph.errors import GraphRecursionError

    monkeypatch.setattr(chat_routes, "run_agent_graph", AsyncMock(
        side_effect=GraphRecursionError("internal details"),
    ))
    response = client.post("/api/v1/chat", json={"messages": [
        {"role": "user", "content": "Repeat forever"}
    ], "session_id": "conversation-limit"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "agent_recursion_limit"
    assert "execution limit" in response.json()["error"]["message"]
    assert "internal details" not in response.text


def test_chat_returns_structured_degraded_response(monkeypatch):
    monkeypatch.setattr(chat_routes, "run_agent_graph", AsyncMock(
        return_value={
            "messages": [AIMessage(
                content="The AI service is temporarily unavailable."
            )],
            "current_turn_tools": [],
            "error_code": "model_unavailable",
        },
    ))

    response = client.post("/api/v1/chat", json={
        "messages": [{"role": "user", "content": "Explain Kafka"}],
        "session_id": "degraded-session",
    })

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["agent_action"] == "error"
    assert response.json()["error_code"] == "model_unavailable"
    assert response.json()["retryable"] is True


def test_chat_returns_approval_request_for_interrupted_graph(monkeypatch):
    approval = {
        "type": "action_approval",
        "title": "Refund approval required",
        "action": {
            "transaction_id": "TX-123",
            "amount_cents": 8745,
            "currency": "USD",
            "reason": "duplicate charge",
        },
    }
    monkeypatch.setattr(chat_routes, "run_agent_graph", AsyncMock(
        return_value={
            "messages": [AIMessage(content="")],
            "current_turn_tools": [],
            "route": None,
            "__interrupt__": [SimpleNamespace(value=approval)],
        },
    ))

    response = client.post("/api/v1/chat", json={
        "messages": [{"role": "user", "content": "Refund TX-123"}],
        "session_id": "approval-session",
    })

    assert response.status_code == 200
    assert response.json()["status"] == "approval_required"
    assert response.json()["response"] is None
    assert response.json()["approval_request"] == approval
    assert response.json()["agent_action"] == "approval"


def test_approval_endpoint_resumes_same_authenticated_thread(monkeypatch):
    resume = AsyncMock(return_value={
        "request_id": "REQ-original",
        "messages": [AIMessage(content="The proposal was rejected.")],
        "current_turn_tools": ["propose_refund"],
        "route": None,
    })
    monkeypatch.setattr(chat_routes, "resume_agent_graph", resume)

    response = client.post("/api/v1/chat/approval", json={
        "session_id": "approval-session",
        "approved": False,
        "comment": "Wrong transaction",
    })

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["response"] == "The proposal was rejected."
    assert response.json()["request_id"] == "REQ-original"
    assert resume.await_args.kwargs["thread_id"] == (
        "USER-TEST:approval-session"
    )
    assert resume.await_args.kwargs["approval"] == {
        "approved": False,
        "comment": "Wrong transaction",
    }


def test_approval_endpoint_rejects_session_without_pending_action(monkeypatch):
    monkeypatch.setattr(chat_routes, "resume_agent_graph", AsyncMock(
        side_effect=chat_routes.PendingApprovalNotFoundError(
            "No pending approval"
        ),
    ))

    response = client.post("/api/v1/chat/approval", json={
        "session_id": "no-pending-action",
        "approved": True,
    })

    assert response.status_code == 409
    assert response.json()["error"] == {
        "code": "approval_not_pending",
        "message": "No pending approval exists for this session.",
        "retryable": False,
    }
