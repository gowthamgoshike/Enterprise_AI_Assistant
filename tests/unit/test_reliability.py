import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
import psycopg
from psycopg_pool import PoolTimeout
from langchain_core.messages import AIMessage
from sqlalchemy.exc import OperationalError

from ai.models import reliability as model_reliability
from ai.nodes import refund_execution_node, secure_tool_node
from ai.schemas.actions import RefundExecutionResult
from ai.tools.secure_registry import ToolDefinition
from core.errors import (
    DatabaseUnavailableError,
    ModelUnavailableError,
    PermissionDeniedError,
)
from core import retry
from services.auth_service.models import UserRole
from services.auth_service.permissions import Permission
from services.transaction_service import repository
from data.database.reliability import is_database_unavailable_error


def test_backoff_is_exponential_capped_and_jittered(monkeypatch):
    monkeypatch.setattr(retry.random, "uniform", lambda low, high: high)
    assert retry.calculate_backoff(0, 0.5, 4.0) == 0.6
    assert retry.calculate_backoff(1, 0.5, 4.0) == 1.2
    assert retry.calculate_backoff(10, 0.5, 4.0) == 4.8


def test_retry_async_retries_only_classified_errors(monkeypatch):
    sleeps = []
    monkeypatch.setattr(retry, "calculate_backoff", lambda _attempt: 0.01)
    monkeypatch.setattr(
        retry.asyncio,
        "sleep",
        AsyncMock(side_effect=lambda delay: sleeps.append(delay)),
    )
    operation = AsyncMock(side_effect=[DatabaseUnavailableError(), "ok"])

    result = asyncio.run(retry.retry_async(
        operation,
        max_retries=2,
        retry_on=(DatabaseUnavailableError,),
    ))

    assert result == "ok"
    assert operation.await_count == 2
    assert sleeps == [0.01]

    permanent = AsyncMock(side_effect=PermissionDeniedError())
    with pytest.raises(PermissionDeniedError):
        asyncio.run(retry.retry_async(
            permanent,
            max_retries=2,
            retry_on=(DatabaseUnavailableError,),
        ))
    assert permanent.await_count == 1


def test_model_timeout_is_retried_then_sanitized(monkeypatch):
    monkeypatch.setattr(model_reliability, "MODEL_MAX_RETRIES", 1)
    monkeypatch.setattr(model_reliability, "MODEL_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(retry, "calculate_backoff", lambda _attempt: 0)
    operation = AsyncMock(side_effect=[TimeoutError("provider detail"), "ok"])

    assert asyncio.run(model_reliability.invoke_model_with_retry(operation)) == "ok"
    assert operation.await_count == 2


def test_repository_translates_operational_error(monkeypatch):
    monkeypatch.setattr(
        repository.engine,
        "connect",
        Mock(side_effect=OperationalError("select", {}, RuntimeError("secret"))),
    )

    with pytest.raises(DatabaseUnavailableError) as error:
        repository.get_transaction("TX-1")

    assert error.value.code == "database_unavailable"
    assert "secret" not in error.value.safe_message


@pytest.mark.parametrize(
    "error",
    [
        psycopg.OperationalError("connection refused"),
        PoolTimeout("pool exhausted"),
        RuntimeError("programming bug"),
    ],
)
def test_checkpoint_database_error_classification(error):
    expected = not isinstance(error, RuntimeError)
    assert is_database_unavailable_error(error) is expected


def test_checkpoint_database_error_classification_follows_cause():
    wrapper = RuntimeError("graph failed")
    wrapper.__cause__ = psycopg.OperationalError("connection refused")
    assert is_database_unavailable_error(wrapper) is True


def _secure_state(tool_name: str) -> dict:
    return {
        "messages": [AIMessage(content="", tool_calls=[{
            "name": tool_name,
            "args": {},
            "id": "call-1",
        }])],
        "request_id": "REQ-1",
        "user_id": "USER-1",
        "user_role": UserRole.ANALYST.value,
        "customer_id": None,
        "current_turn_tools": [],
        "retry_count": 0,
    }


def test_read_tool_retries_database_failure_and_returns_envelope(monkeypatch):
    operation = Mock(side_effect=[
        DatabaseUnavailableError(),
        {"value": 42},
    ])
    monkeypatch.setitem(
        secure_tool_node.SECURE_TOOL_REGISTRY,
        "get_revenue_summary",
        ToolDefinition(
            name="get_revenue_summary",
            function=operation,
            permission=Permission.READ_REVENUE,
            retry_safe=True,
        ),
    )
    monkeypatch.setattr(retry, "calculate_backoff", lambda _attempt: 0)

    result = asyncio.run(secure_tool_node.secure_tool_node(
        _secure_state("get_revenue_summary")
    ))

    assert operation.call_count == 2
    assert json.loads(result["messages"][0].content) == {
        "ok": True,
        "data": {"value": 42},
        "error_code": None,
        "message": None,
        "retryable": False,
    }


def test_refund_retry_reuses_action_id(monkeypatch):
    calls = []

    def execute(**kwargs):
        calls.append(kwargs["action_id"])
        if len(calls) == 1:
            raise DatabaseUnavailableError()
        return RefundExecutionResult(
            action_id=kwargs["action_id"],
            refund_id="REFUND-1",
            transaction_id=kwargs["transaction_id"],
            status="already_completed",
            amount_cents=100,
            currency="USD",
        )

    monkeypatch.setattr(refund_execution_node, "execute_refund", execute)
    monkeypatch.setattr(retry, "calculate_backoff", lambda _attempt: 0)
    state = {
        "request_id": "REQ-1",
        "user_id": "USER-ADMIN",
        "user_role": UserRole.ADMIN.value,
        "current_turn_tools": [],
        "pending_action": {
            "action_id": "ACTION-STABLE",
            "transaction_id": "TX-1",
            "reason": "duplicate",
            "requested_by": "USER-ADMIN",
        },
        "pending_tool_call_id": "call-1",
        "retry_count": 0,
    }

    result = asyncio.run(refund_execution_node.execute_refund_node(state))

    assert calls == ["ACTION-STABLE", "ACTION-STABLE"]
    payload = json.loads(result["messages"][0].content)
    assert payload["data"]["refund"]["status"] == "already_completed"
    assert result["retry_count"] == 1
