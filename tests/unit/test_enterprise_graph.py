import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai.graphs import enterprise_graph as graph


@pytest.mark.parametrize("route", ["transaction", "analytics", "general"])
@pytest.mark.parametrize("confidence", [0.0, 0.6499, 0.65, 0.93])
def test_graph_confidence_routing(monkeypatch, caplog, route, confidence):
    monkeypatch.setattr(graph, "route_request", AsyncMock(
        return_value=SimpleNamespace(route=route, confidence=confidence),
    ))
    handlers = {}
    for domain in ("transaction", "analytics", "general"):
        result = "answer" if domain == "general" else {
            "response": "answer", "tools_used": ["example_tool"],
        }
        handlers[domain] = AsyncMock(return_value=result)
        monkeypatch.setattr(graph, f"handle_{domain}_request", handlers[domain])

    with caplog.at_level(logging.INFO, logger=graph.__name__):
        result = asyncio.run(graph.run_enterprise_graph("Private question", "REQ-777"))

    clarify = confidence < 0.65
    for domain, handler in handlers.items():
        if domain == route and not clarify:
            handler.assert_awaited_once_with("Private question")
        else:
            handler.assert_not_awaited()
    assert result["request_id"] == "REQ-777"
    assert result["route"] == route
    assert result["route_confidence"] == confidence
    assert result["error"] is None
    assert result["tools_used"] == (
        [] if clarify or route == "general" else ["example_tool"]
    )
    if clarify:
        assert "Could you clarify" in result["response"]
    else:
        assert result["response"] == "answer"
    assert "REQ-777 node=router" in caplog.text
    assert f"node={'clarify' if clarify else route}" in caplog.text
    assert "REQ-777 graph=end" in caplog.text
    assert "Private question" not in caplog.text


@pytest.mark.parametrize("state", [{"route": "analytics"}, {
    "route": "analytics", "route_confidence": None,
}])
def test_missing_confidence_requests_clarification(state):
    assert graph.choose_route(state) == "clarify"


def test_transaction_failure_returns_safe_state(monkeypatch, caplog):
    monkeypatch.setattr(graph, "route_request", AsyncMock(
        return_value=SimpleNamespace(route="transaction", confidence=0.9),
    ))
    monkeypatch.setattr(graph, "handle_transaction_request", AsyncMock(
        side_effect=RuntimeError("private database credentials"),
    ))
    result = asyncio.run(graph.run_enterprise_graph("question", "REQ-error"))
    assert result["error"] == "transaction_handler_failed"
    assert result["response"] == "Transaction data is temporarily unavailable."
    assert result["tools_used"] == []
    assert "private database credentials" not in str(result)
    assert "private database credentials" not in caplog.text


def test_error_node_returns_safe_response():
    result = asyncio.run(graph.error_node({"request_id": "REQ-error"}))
    assert result == {"response": "The requested service is temporarily unavailable."}
