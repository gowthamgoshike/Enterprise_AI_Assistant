import asyncio
import json
import logging
from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from langchain_core.messages import AIMessage

from ai.graphs import tool_agent_graph
from ai.nodes import secure_tool_node as secure_node_module
from ai.tools import secure_executors
from ai.tools.context import ToolExecutionContext
from ai.tools.langchain_registry import get_tools_for_role
from ai.tools.secure_registry import ToolDefinition
from ai.tools.transaction_tools import get_my_transactions
from services.auth_service.exceptions import AuthorizationError
from services.auth_service.models import UserRole
from services.auth_service.permissions import Permission
from services.transaction_service import repository as transaction_repository
from core.observability.metrics import AUTHORIZATION_DENIALS_TOTAL


def state_for(role, tool_name, arguments=None):
    return {
        "messages": [AIMessage(content="", tool_calls=[{
            "name": tool_name,
            "args": arguments or {},
            "id": "call-1",
        }])],
        "request_id": "REQ-1",
        "user_id": "USER-1",
        "user_role": role.value,
        "customer_id": "CUST-TRUSTED",
        "route": None,
        "current_turn_tools": [],
        "conversation_summary": None,
        "error": None,
    }


def test_tool_execution_context_is_frozen():
    context = ToolExecutionContext(
        user_id="USER-1",
        role=UserRole.CUSTOMER,
        customer_id="CUST-1",
        request_id="REQ-1",
    )

    with pytest.raises(FrozenInstanceError):
        context.role = UserRole.ADMIN


def test_customer_tool_schema_does_not_accept_customer_identity():
    properties = get_my_transactions.args_schema.model_json_schema()["properties"]

    assert set(properties) == {"limit"}


@pytest.mark.parametrize(
    ("role", "expected_names"),
    [
        (
            UserRole.CUSTOMER,
            {"get_my_transactions", "get_transaction"},
        ),
        (
            UserRole.ANALYST,
            {
                "get_transaction",
                "get_top_products",
                "get_revenue_summary",
                "get_transaction_metrics",
            },
        ),
        (
            UserRole.ADMIN,
            {
                "get_transaction",
                "get_top_products",
                "get_revenue_summary",
                "get_transaction_metrics",
                "propose_refund",
            },
        ),
    ],
)
def test_role_specific_tool_exposure(role, expected_names):
    assert {tool.name for tool in get_tools_for_role(role)} == expected_names


def test_agent_binds_only_authenticated_roles_tools(monkeypatch):
    response_model = AsyncMock()
    response_model.ainvoke.return_value = AIMessage(content="General answer")
    base_model = Mock()
    base_model.bind_tools.return_value = response_model
    monkeypatch.setattr(tool_agent_graph, "model", base_model)
    monkeypatch.setattr(tool_agent_graph, "model_with_tools", None)
    state = state_for(UserRole.CUSTOMER, "unused")
    state["messages"] = [tool_agent_graph.HumanMessage(content="Explain Kafka")]

    asyncio.run(tool_agent_graph.agent_node(state))

    bound_tools = base_model.bind_tools.call_args.args[0]
    assert {tool.name for tool in bound_tools} == {
        "get_my_transactions",
        "get_transaction",
    }


def test_my_transactions_uses_trusted_customer_and_clamps_limit(monkeypatch):
    lookup = Mock(return_value={"count": 0, "transactions": []})
    monkeypatch.setattr(
        secure_executors,
        "fetch_customer_transactions",
        lookup,
    )
    context = ToolExecutionContext(
        user_id="USER-1",
        role=UserRole.CUSTOMER,
        customer_id="CUST-TRUSTED",
        request_id="REQ-1",
    )

    secure_executors.execute_get_my_transactions(context, limit=999)

    lookup.assert_called_once_with(
        customer_id="CUST-TRUSTED",
        limit=20,
    )


def test_my_transactions_requires_customer_identity():
    context = ToolExecutionContext(
        user_id="USER-1",
        role=UserRole.CUSTOMER,
        customer_id=None,
        request_id="REQ-1",
    )

    with pytest.raises(
        AuthorizationError,
        match="Customer identity unavailable",
    ):
        secure_executors.execute_get_my_transactions(context)


def test_secure_node_discards_model_supplied_identity(monkeypatch):
    function = Mock(return_value={"count": 0, "transactions": []})
    monkeypatch.setitem(
        secure_node_module.SECURE_TOOL_REGISTRY,
        "get_my_transactions",
        ToolDefinition(
            name="get_my_transactions",
            function=function,
            permission=Permission.READ_OWN_TRANSACTIONS,
        ),
    )

    result = asyncio.run(secure_node_module.secure_tool_node(state_for(
        UserRole.CUSTOMER,
        "get_my_transactions",
        {
            "context": {"role": "ADMIN"},
            "customer_id": "CUST-999",
            "user_id": "USER-ATTACKER",
            "role": "ADMIN",
            "limit": 10,
        },
    )))

    assert json.loads(result["messages"][0].content) == {
        "ok": True,
        "data": {"count": 0, "transactions": []},
        "error_code": None,
        "message": None,
        "retryable": False,
    }
    assert result["messages"][0].status == "success"
    assert result["current_turn_tools"] == ["get_my_transactions"]
    call_arguments = function.call_args.kwargs
    assert call_arguments["context"].customer_id == "CUST-TRUSTED"
    assert call_arguments["context"].role is UserRole.CUSTOMER
    assert call_arguments["limit"] == 10
    assert "customer_id" not in call_arguments


def test_spoofed_customer_id_still_queries_authenticated_customer(
    monkeypatch,
    caplog,
):
    connection = MagicMock()
    connection.execute.return_value.mappings.return_value = []
    connection_context = MagicMock()
    connection_context.__enter__.return_value = connection
    monkeypatch.setattr(
        transaction_repository.engine,
        "connect",
        lambda: connection_context,
    )

    with caplog.at_level(logging.INFO, logger=secure_node_module.__name__):
        result = asyncio.run(secure_node_module.secure_tool_node(state_for(
            UserRole.CUSTOMER,
            "get_my_transactions",
            {"customer_id": "CUST-999", "limit": 10},
        )))

    assert result["messages"][0].status == "success"
    query, parameters = connection.execute.call_args.args
    assert "WHERE customer_id = :customer_id" in str(query)
    assert parameters == {
        "customer_id": "CUST-TRUSTED",
        "limit": 10,
    }
    assert "CUST-999" not in str(parameters)
    audit_record = next(
        record
        for record in caplog.records
        if getattr(record, "authorization_result", None) == "ALLOWED"
    )
    assert audit_record.request_id == "REQ-1"
    assert audit_record.user_id == "USER-1"
    assert audit_record.role == "CUSTOMER"
    assert audit_record.tool == "get_my_transactions"
    assert audit_record.authorization_latency >= 0
    assert "CUST-999" not in caplog.text


def test_prompt_injection_cannot_bypass_customer_analytics_denial(
    monkeypatch,
    caplog,
):
    function = Mock()
    monkeypatch.setitem(
        secure_node_module.SECURE_TOOL_REGISTRY,
        "get_revenue_summary",
        ToolDefinition(
            name="get_revenue_summary",
            function=function,
            permission=Permission.READ_REVENUE,
        ),
    )

    denials = AUTHORIZATION_DENIALS_TOTAL.labels(
        role="CUSTOMER",
        capability="get_revenue_summary",
    )
    denials_before = denials._value.get()

    with caplog.at_level(logging.WARNING, logger=secure_node_module.__name__):
        result = asyncio.run(secure_node_module.secure_tool_node(state_for(
            UserRole.CUSTOMER,
            "get_revenue_summary",
        )))

    assert json.loads(result["messages"][0].content) == {
        "ok": False,
        "data": None,
        "error_code": "permission_denied",
        "message": "You are not authorized to perform this action.",
        "retryable": False,
    }
    assert result["messages"][0].status == "error"
    assert result["current_turn_tools"] == []
    function.assert_not_called()
    record = next(
        record
        for record in caplog.records
        if getattr(record, "authorization_result", None) == "DENIED"
    )
    assert record.request_id == "REQ-1"
    assert record.user_id == "USER-1"
    assert record.role == "CUSTOMER"
    assert record.tool == "get_revenue_summary"
    assert record.authorization_latency >= 0
    assert denials._value.get() == denials_before + 1


def test_customer_transaction_executor_enforces_ownership(monkeypatch):
    transactions = {
        "TX-A": {
            "transaction_id": "TX-A",
            "customer_id": "CUST-A",
            "product_id": "PROD-1",
            "amount_cents": 1000,
            "currency": "USD",
            "status": "completed",
            "transaction_type": "purchase",
            "source": "web",
            "event_time": "2026-09-19T12:00:00Z",
        },
        "TX-B": {
            "transaction_id": "TX-B",
            "customer_id": "CUST-B",
            "product_id": "PROD-2",
            "amount_cents": 2000,
            "currency": "USD",
            "status": "completed",
            "transaction_type": "purchase",
            "source": "web",
            "event_time": "2026-09-19T12:00:00Z",
        },
    }
    calls = []

    def scoped_lookup(transaction_id, customer_id):
        calls.append((transaction_id, customer_id))
        transaction = transactions.get(transaction_id)
        if transaction is None or transaction["customer_id"] != customer_id:
            return {"found": False, "transaction": None}
        return {"found": True, "transaction": transaction}

    monkeypatch.setattr(
        secure_executors,
        "find_customer_transaction",
        scoped_lookup,
    )
    context = ToolExecutionContext(
        user_id="USER-A",
        role=UserRole.CUSTOMER,
        customer_id="CUST-A",
        request_id="REQ-ownership",
    )

    own_result = secure_executors.execute_get_transaction(context, "TX-A")
    other_result = secure_executors.execute_get_transaction(context, "TX-B")

    assert own_result["found"] is True
    assert own_result["transaction"]["transaction_id"] == "TX-A"
    assert other_result == {"found": False, "transaction": None}
    assert calls == [
        ("TX-A", "CUST-A"),
        ("TX-B", "CUST-A"),
    ]


def test_secure_node_never_executes_unknown_tool():
    result = asyncio.run(secure_node_module.secure_tool_node(state_for(
        UserRole.ADMIN,
        "delete_database",
    )))

    assert json.loads(result["messages"][0].content) == {
        "ok": False,
        "data": None,
        "error_code": "unknown_tool",
        "message": "The requested tool is not available.",
        "retryable": False,
    }
    assert result["messages"][0].status == "error"


def test_secure_node_hides_raw_execution_errors(monkeypatch, caplog):
    def fail(**kwargs):
        raise RuntimeError("private database credentials")

    monkeypatch.setitem(
        secure_node_module.SECURE_TOOL_REGISTRY,
        "get_revenue_summary",
        ToolDefinition(
            name="get_revenue_summary",
            function=fail,
            permission=Permission.READ_REVENUE,
        ),
    )

    with caplog.at_level(logging.ERROR, logger=secure_node_module.__name__):
        result = asyncio.run(secure_node_module.secure_tool_node(state_for(
            UserRole.ANALYST,
            "get_revenue_summary",
        )))

    assert json.loads(result["messages"][0].content) == {
        "ok": False,
        "data": None,
        "error_code": "internal_error",
        "message": "The operation could not be completed.",
        "retryable": False,
    }
    assert "private database credentials" not in result["messages"][0].content
    assert "private database credentials" in caplog.text
