import json

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routes import chat as chat_routes


client = TestClient(app)


def test_chat_endpoints_use_graph(monkeypatch):
    calls = []
    tool_names = ["get_revenue_summary", "get_transaction_metrics", "get_top_products"]

    async def fake_graph(*, user_message, request_id):
        calls.append((user_message, request_id))
        return {"response": "Revenue is 100.", "tools_used": tool_names, "route": "analytics", "route_confidence": 0.94}

    monkeypatch.setattr(chat_routes, "run_enterprise_graph", fake_graph)
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
    assert response.json()["route"] == "analytics"
    assert response.json()["route_confidence"] == 0.94

    response = client.post("/api/v1/chat/stream", json={"messages": messages})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["message"]["content"] == "Revenue is 100."
    assert events[-1]["done"] is True
    assert events[-1]["route"] == "analytics"
    assert events[-1]["route_confidence"] == 0.94
    assert events[-1]["tools_used"] == tool_names
    assert len(calls) == 2
    assert all(call[0] == messages[-1]["content"] for call in calls)
    assert all(call[1] for call in calls)


def test_empty_chat_history_is_rejected():
    for endpoint in ("/api/v1/chat", "/api/v1/chat/stream"):
        assert client.post(endpoint, json={"messages": []}).status_code == 422


def test_chat_uses_graph_confidence_fallback(monkeypatch):
    from unittest.mock import AsyncMock
    from ai.graphs import enterprise_graph
    from ai.schemas.agent import IntentRoute

    monkeypatch.setattr(enterprise_graph, 'route_request', AsyncMock(return_value=IntentRoute(
        route='analytics', confidence=0.42, reason='Ambiguous request'
    )))
    handlers = []
    for domain in ['transaction', 'analytics', 'general']:
        handler = AsyncMock()
        monkeypatch.setattr(enterprise_graph, f'handle_{domain}_request', handler)
        handlers.append(handler)
    for endpoint in ['/api/v1/chat', '/api/v1/chat/stream']:
        response = client.post(endpoint, json={'messages': [
            {'role': 'user', 'content': "Show me what's going on."}
        ]})
        assert response.status_code == 200
        if endpoint.endswith('/stream'):
            result = json.loads(response.text.splitlines()[-1])
        else:
            result = response.json()
            assert result['agent_action'] == 'respond'
            assert 'clarify' in result['response']
        assert result['route'] == 'analytics'
        assert result['route_confidence'] == 0.42
        assert result['tools_used'] == []
    for handler in handlers:
        handler.assert_not_awaited()
