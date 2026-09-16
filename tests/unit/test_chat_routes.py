import json

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routes import chat as chat_routes


client = TestClient(app)


def test_chat_endpoints_use_analytics_agent(monkeypatch):
    calls = []
    tool_names = ["get_revenue_summary", "get_transaction_metrics", "get_top_products"]

    async def fake_agent(message, request_id, history):
        calls.append((message, request_id, history))
        return {"response": "Revenue is 100.", "tools_used": tool_names}

    monkeypatch.setattr(chat_routes, "process_with_langchain", fake_agent)
    monkeypatch.setattr(chat_routes, "OLLAMA_MODEL", "test-model")
    messages = [
        {"role": "user", "content": "Show revenue."},
        {"role": "assistant", "content": "Revenue is 100."},
        {"role": "user", "content": "How is the business doing?"},
    ]
    response = client.post("/api/v1/chat", json={"messages": messages})
    assert response.status_code == 200
    assert response.json()["tools_used"] == tool_names
    assert "tool_used" not in response.json()
    assert response.json()["agent_action"] == "tool"

    response = client.post("/api/v1/chat/stream", json={"messages": messages})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["message"]["content"] == "Revenue is 100."
    assert events[-1]["done"] is True
    assert events[-1]["tools_used"] == tool_names
    assert len(calls) == 2
    assert all(call[0] == messages[-1]["content"] for call in calls)
    assert all(call[1] for call in calls)
    assert all(call[2] == messages for call in calls)


def test_empty_chat_history_is_rejected():
    for endpoint in ("/api/v1/chat", "/api/v1/chat/stream"):
        assert client.post(endpoint, json={"messages": []}).status_code == 422
