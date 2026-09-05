import pytest
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from ai.tools.exceptions import (
    ToolExecutionError,
)
from ai.tools import transaction_tools


def test_get_transaction_normalizes_transaction_id(monkeypatch):

    seen = {}

    def fake_find_transaction(transaction_id):
        seen["transaction_id"] = transaction_id
        return {
            "found": True,
            "transaction": {
                "transaction_id": transaction_id,
                "customer_id": "CUST-1",
                "product_id": "PROD-1",
                "amount_cents": 1000,
                "currency": "USD",
                "status": "completed",
                "transaction_type": "purchase",
                "source": "api",
                "event_time": "2026-09-05T12:00:00Z",
            },
        }

    monkeypatch.setattr(
        transaction_tools,
        "find_transaction",
        fake_find_transaction,
    )

    result = transaction_tools.get_transaction_tool(
        "  tx-123  "
    )

    assert seen["transaction_id"] == "TX-123"
    assert result.success is True
    assert result.tool_name == "get_transaction"
    assert result.error is None
    assert result.data["found"] is True
    assert result.data["transaction"]["transaction_id"] == "TX-123"


def test_get_transaction_rejects_blank_normalized_id():

    with pytest.raises(ValueError):
        transaction_tools.get_transaction_tool(
            "   "
        )


def test_get_transaction_rejects_invalid_database_result(monkeypatch):

    monkeypatch.setattr(
        transaction_tools,
        "find_transaction",
        lambda transaction_id: {
            "found": True,
            "transaction": {
                "transaction_id": transaction_id,
            },
        },
    )

    with pytest.raises(ValidationError):
        transaction_tools.get_transaction_tool(
            "TX-123"
        )


def test_get_transaction_returns_structured_database_error(
    monkeypatch,
):

    def fail_find_transaction(transaction_id):
        raise SQLAlchemyError("connection failed")

    monkeypatch.setattr(
        transaction_tools,
        "find_transaction",
        fail_find_transaction,
    )

    with pytest.raises(ToolExecutionError) as error:
        transaction_tools.get_transaction_tool(
            "TX-123"
        )

    assert error.value.code == "DATABASE_UNAVAILABLE"
    assert error.value.retryable is True


def test_get_transaction_not_found_is_successful_business_result(
    monkeypatch,
):

    monkeypatch.setattr(
        transaction_tools,
        "find_transaction",
        lambda transaction_id: {
            "found": False,
            "transaction": None,
        },
    )

    result = transaction_tools.get_transaction_tool(
        "TX-404"
    )

    assert result.success is True
    assert result.error is None
    assert result.data == {
        "found": False,
        "transaction": None,
    }