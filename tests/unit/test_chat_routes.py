import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from apps.api.main import app
from apps.api.routes import chat as chat_routes


client = TestClient(app)


def test_chat_endpoints_use_graph(monkeypatch):
    calls = []
    tool_names = ["get_revenue_summary", "get_transaction_metrics", "get_top_products"]

    async def fake_graph(*, user_message, request_id):
        calls.append((user_message, request_id))
        return {"messages": [
            AIMessage(content="", tool_calls=[
                {"name": name, "args": {}, "id": f"call-{i}"}
                for i, name in enumerate(tool_names)
            ]),
            AIMessage(content="Revenue is 100."),
        ], "route": None}

    monkeypatch.setattr(chat_routes, "run_agent_graph", fake_graph)
    monkeypatch.setattr(chat_routes, "OLLAMA_MODEL", "test-model")
    messages = [
        {"role": "user", "content": "Show revenue."},
        {"role": "assistant", "content": "Revenue is 100."},
        {"role": "user", "content": "How is the business doing?"},
    ]
    response = client.post("/api/v1/chat", json={"messages": messages})
    assert response.status_code == 200
    assert response.json()["request_id"] == calls[0][1]
    assert response.json()["response"] == "Revenue is 100."
    assert response.json()["model"] == "test-model"
    assert response.json()["latency_seconds"] >= 0
    assert response.json()["tools_used"] == tool_names
    assert "tool_used" not in response.json()
    assert response.json()["agent_action"] == "tool"
    assert response.json()["route"] is None
    assert response.json()["route_confidence"] is None

    response = client.post("/api/v1/chat/stream", json={"messages": messages})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["message"]["content"] == "Revenue is 100."
    assert events[-1]["done"] is True
    assert events[-1]["route"] is None
    assert events[-1]["route_confidence"] is None
    assert events[-1]["tools_used"] == tool_names
    assert len(calls) == 2
    assert all(call[0] == messages[-1]["content"] for call in calls)
    assert all(call[1] for call in calls)


def test_empty_chat_history_is_rejected():
    for endpoint in ("/api/v1/chat", "/api/v1/chat/stream"):
        assert client.post(endpoint, json={"messages": []}).status_code == 422


def test_chat_general_answer_uses_agent_loop(monkeypatch):
    from unittest.mock import AsyncMock
    from ai.graphs import tool_agent_graph

    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(content="Please clarify your request.")
    monkeypatch.setattr(tool_agent_graph, "model_with_tools", model)
    for endpoint in ["/api/v1/chat", "/api/v1/chat/stream"]:
        response = client.post(endpoint, json={"messages": [
            {"role": "user", "content": "Show me what's going on."}
        ]})
        assert response.status_code == 200
        if endpoint.endswith("/stream"):
            result = json.loads(response.text.splitlines()[-1])
        else:
            result = response.json()
            assert result["agent_action"] == "respond"
            assert result["response"] == "Please clarify your request."
        assert result["route"] is None
        assert result["route_confidence"] is None
        assert result["tools_used"] == []


def test_chat_step_limit_returns_safe_error(monkeypatch):
    from unittest.mock import AsyncMock
    from langgraph.errors import GraphRecursionError

    monkeypatch.setattr(chat_routes, "run_agent_graph", AsyncMock(
        side_effect=GraphRecursionError("internal details"),
    ))
    for endpoint in ["/api/v1/chat", "/api/v1/chat/stream"]:
        response = client.post(endpoint, json={"messages": [
            {"role": "user", "content": "Repeat forever"}
        ]})
        assert response.status_code == 503
        assert "execution limit" in response.json()["detail"]
        assert "internal details" not in response.text
