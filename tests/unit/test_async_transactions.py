"""Async reads keep parameterized ownership checks and scoped connections."""

import asyncio
from unittest.mock import AsyncMock

from ai.tools import secure_executors
from ai.tools.context import ToolExecutionContext
from services.auth_service.models import UserRole
from services.transaction_service import async_repository


class FakeConnection:
    def __init__(self, row):
        self.row = row
        self.calls = []

    async def execute(self, statement, parameters):
        self.calls.append((statement, parameters))
        return self

    def mappings(self):
        return self

    def first(self):
        return self.row


class FakeEngine:
    def __init__(self, row):
        self.connection = FakeConnection(row)
        self.closed = False

    def connect(self):
        return self

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, *_args):
        self.closed = True


def test_async_customer_lookup_is_parameterized_and_scoped(monkeypatch):
    engine = FakeEngine({"transaction_id": "TX-1", "customer_id": "CUST-1"})
    monkeypatch.setattr(async_repository, "async_engine", engine)

    result = asyncio.run(async_repository.find_customer_transaction_async(
        transaction_id="TX-1",
        customer_id="CUST-1",
    ))

    assert result == {"transaction_id": "TX-1", "customer_id": "CUST-1"}
    statement, parameters = engine.connection.calls[0]
    assert "customer_id = :customer_id" in str(statement)
    assert parameters == {"transaction_id": "TX-1", "customer_id": "CUST-1"}
    assert engine.closed


def test_async_tool_uses_trusted_customer_identity(monkeypatch):
    lookup = AsyncMock(return_value={"found": False, "transaction": None})
    monkeypatch.setattr(
        secure_executors,
        "find_customer_transaction_async",
        lookup,
    )
    context = ToolExecutionContext(
        user_id="USER-1",
        role=UserRole.CUSTOMER,
        customer_id="CUST-TRUSTED",
        request_id="REQ-1",
    )

    result = asyncio.run(secure_executors.execute_get_transaction_async(
        context=context,
        transaction_id="TX-1",
    ))

    assert result == {"found": False, "transaction": None}
    lookup.assert_awaited_once_with(
        transaction_id="TX-1",
        customer_id="CUST-TRUSTED",
    )
