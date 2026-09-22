import asyncio
import json
import logging

from fastapi.testclient import TestClient
from langgraph.errors import GraphInterrupt
import pytest

from apps.api.main import app
from core.observability.context import (
    get_request_id,
    get_user_id,
    reset_request_context,
    set_request_context,
)
from core.observability.logging import JsonFormatter
from core.observability.metrics import (
    GRAPH_NODE_DURATION,
    GRAPH_NODE_EXECUTIONS,
)
from core.observability.timing import Timer, observe_async_node, observe_node


client = TestClient(app)


def test_json_formatter_adds_context_and_allowlisted_fields():
    tokens = set_request_context("REQ-123", "USER-123")
    try:
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="tool completed",
            args=(),
            exc_info=None,
        )
        record.event = "tool_execution"
        record.tool = "get_transaction"
        record.status = "success"
        record.password = "must-not-appear"

        payload = json.loads(JsonFormatter().format(record))
    finally:
        reset_request_context(tokens)

    assert payload["request_id"] == "REQ-123"
    assert payload["user_id"] == "USER-123"
    assert payload["event"] == "tool_execution"
    assert payload["tool"] == "get_transaction"
    assert payload["status"] == "success"
    assert "password" not in payload
    assert "must-not-appear" not in json.dumps(payload)


def test_request_context_can_be_reset_without_leaking():
    before_request = get_request_id()
    before_user = get_user_id()
    tokens = set_request_context("REQ-isolated", "USER-isolated")
    assert get_request_id() == "REQ-isolated"
    assert get_user_id() == "USER-isolated"
    reset_request_context(tokens)
    assert get_request_id() == before_request
    assert get_user_id() == before_user


def test_timer_and_async_node_observer(caplog):
    async def node(state):
        await asyncio.sleep(0)
        return {"value": state["value"] + 1}

    with Timer() as timer:
        sum(range(100))
    assert timer.elapsed_ms >= 0

    executions = GRAPH_NODE_EXECUTIONS.labels(
        node="example",
        status="success",
    )
    durations = GRAPH_NODE_DURATION.labels(node="example")
    executions_before = executions._value.get()
    duration_count_before = next(
        sample.value
        for metric in durations.collect()
        for sample in metric.samples
        if sample.name == "graph_node_duration_seconds_count"
    )

    with caplog.at_level(logging.INFO, logger="core.observability.timing"):
        result = asyncio.run(observe_async_node("example", node, {"value": 1}))

    assert result == {"value": 2}
    records = [r for r in caplog.records if getattr(r, "node", None) == "example"]
    assert [record.status for record in records] == ["started", "success"]
    assert records[-1].latency_ms >= 0
    assert executions._value.get() == executions_before + 1
    duration_count_after = next(
        sample.value
        for metric in durations.collect()
        for sample in metric.samples
        if sample.name == "graph_node_duration_seconds_count"
    )
    assert duration_count_after == duration_count_before + 1


def test_sync_node_observer_records_graph_interrupt():
    def interrupted_node(_state):
        raise GraphInterrupt(())

    interruptions = GRAPH_NODE_EXECUTIONS.labels(
        node="human_approval",
        status="interrupted",
    )
    interruptions_before = interruptions._value.get()

    with pytest.raises(GraphInterrupt):
        observe_node("human_approval", interrupted_node, {})

    assert interruptions._value.get() == interruptions_before + 1


def test_http_request_id_and_metrics_endpoint():
    health = client.get("/api/v1/health/live")
    assert health.status_code == 200
    assert health.headers["X-Request-ID"].startswith("REQ-")

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain")
    assert "http_requests_total" in metrics.text
    assert "http_request_duration_seconds" in metrics.text
    assert "model_requests_total" in metrics.text
    assert "model_queue_wait_duration_seconds" in metrics.text
    assert "tool_calls_total" in metrics.text
    assert "agent_errors_total" in metrics.text
    assert "approval_decisions_total" in metrics.text
    assert "refund_executions_total" in metrics.text
    assert "graph_node_executions_total" in metrics.text
    assert "graph_node_duration_seconds" in metrics.text
    assert "authorization_denials_total" in metrics.text
    assert 'path="/api/v1/health/live"' in metrics.text

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/metrics" in openapi.json()["paths"]
