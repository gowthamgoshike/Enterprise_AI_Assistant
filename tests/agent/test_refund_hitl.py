import asyncio
from copy import deepcopy
import json
from unittest.mock import AsyncMock, Mock

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai.graphs import tool_agent_graph
from ai.nodes import refund_execution_node
from ai.schemas.actions import RefundExecutionResult
from services.transaction_service import refund_service
from services.transaction_service.exceptions import RefundNotAllowedError


def completed_transaction() -> dict:
    return {
        "transaction_id": "TX-HITL-TEST",
        "customer_id": "CUST-A",
        "product_id": "PROD-TEST",
        "amount_cents": 2500,
        "currency": "USD",
        "status": "completed",
        "transaction_type": "purchase",
        "source": "web",
        "event_time": "2026-09-20T12:00:00Z",
    }


def refund_tool_call() -> AIMessage:
    return AIMessage(content="", tool_calls=[{
        "name": "propose_refund",
        "args": {
            "transaction_id": "TX-HITL-TEST",
            "reason": "duplicate charge",
        },
        "id": "refund-hitl-call",
    }])


def run_refund_workflow(monkeypatch, *, approved: bool):
    transaction = completed_transaction()
    original_transaction = deepcopy(transaction)
    lookup = Mock(return_value={
        "found": True,
        "transaction": transaction,
    })
    monkeypatch.setattr(refund_service, "find_transaction", lookup)
    execution_calls = []

    def execute(**kwargs):
        execution_calls.append(kwargs)
        transaction["status"] = "refunded"
        return RefundExecutionResult(
            action_id=kwargs["action_id"],
            refund_id="00000000-0000-0000-0000-000000000002",
            transaction_id=kwargs["transaction_id"],
            status="completed",
            amount_cents=transaction["amount_cents"],
            currency=transaction["currency"],
        )

    monkeypatch.setattr(refund_execution_node, "execute_refund", execute)

    model = AsyncMock()
    model.ainvoke.side_effect = [
        refund_tool_call(),
        AIMessage(content="The approval decision was recorded."),
    ]
    monkeypatch.setattr(tool_agent_graph, "model_with_tools", model)
    graph = tool_agent_graph.build_enterprise_tool_graph(InMemorySaver())

    async def run():
        interrupted = await tool_agent_graph.run_agent_graph(
            user_message="Refund TX-HITL-TEST because it was duplicated.",
            request_id="REQ-HITL",
            thread_id="THREAD-HITL",
            graph=graph,
            user_id="USER-ADMIN-1",
            user_role="ADMIN",
            customer_id=None,
        )
        assert interrupted["__interrupt__"]
        assert transaction == original_transaction

        resumed = await tool_agent_graph.resume_agent_graph(
            thread_id="THREAD-HITL",
            approval={"approved": approved, "comment": "Reviewed"},
            graph=graph,
        )
        return interrupted, resumed

    interrupted, resumed = asyncio.run(run())
    if approved:
        assert transaction["status"] == "refunded"
        assert len(execution_calls) == 1
    else:
        assert transaction == original_transaction
        assert execution_calls == []
    lookup.assert_called_once_with("TX-HITL-TEST")
    return interrupted, resumed, execution_calls


def approval_result(resumed: dict) -> dict:
    message = next(
        message
        for message in resumed["messages"]
        if isinstance(message, ToolMessage)
        and message.tool_call_id == "refund-hitl-call"
    )
    return json.loads(message.content)


def test_approve_interrupt_executes_refund(monkeypatch):
    interrupted, resumed, execution_calls = run_refund_workflow(
        monkeypatch,
        approved=True,
    )

    proposal = interrupted["__interrupt__"][0].value["action"]
    assert proposal["transaction_id"] == "TX-HITL-TEST"
    assert proposal["amount_cents"] == 2500
    result = approval_result(resumed)
    assert result["data"]["refund"]["status"] == "completed"
    assert execution_calls[0]["action_id"] == proposal["action_id"]
    assert resumed["current_turn_tools"] == [
        "propose_refund",
        "execute_refund",
    ]
    assert resumed["approval_status"] == "approved"


def test_reject_interrupt_does_not_change_transaction(monkeypatch):
    interrupted, resumed, execution_calls = run_refund_workflow(
        monkeypatch,
        approved=False,
    )

    assert interrupted["__interrupt__"][0].value["type"] == (
        "action_approval"
    )
    assert approval_result(resumed)["status"] == "rejected_by_human"
    assert resumed["approval_status"] == "rejected"
    assert execution_calls == []


def test_state_change_during_approval_is_rejected(monkeypatch):
    transaction = completed_transaction()
    monkeypatch.setattr(refund_service, "find_transaction", Mock(
        return_value={"found": True, "transaction": transaction},
    ))

    def execute(**_kwargs):
        if transaction["status"] != "completed":
            raise RefundNotAllowedError(
                "Transaction is no longer eligible for refund"
            )
        raise AssertionError("The changed transaction must not be refunded")

    monkeypatch.setattr(refund_execution_node, "execute_refund", execute)
    model = AsyncMock()
    model.ainvoke.side_effect = [
        refund_tool_call(),
        AIMessage(content="The transaction is no longer eligible for refund."),
    ]
    monkeypatch.setattr(tool_agent_graph, "model_with_tools", model)
    graph = tool_agent_graph.build_enterprise_tool_graph(InMemorySaver())

    async def run():
        interrupted = await tool_agent_graph.run_agent_graph(
            user_message="Refund TX-HITL-TEST because it was duplicated.",
            request_id="REQ-TOCTOU",
            thread_id="THREAD-TOCTOU",
            graph=graph,
            user_id="USER-ADMIN-1",
            user_role="ADMIN",
            customer_id=None,
        )
        assert interrupted["__interrupt__"]
        transaction["status"] = "refunded"
        return await tool_agent_graph.resume_agent_graph(
            thread_id="THREAD-TOCTOU",
            approval={"approved": True, "comment": "Reviewed"},
            graph=graph,
        )

    resumed = asyncio.run(run())
    result = approval_result(resumed)
    assert result == {
        "ok": False,
        "data": None,
        "error_code": "refund_not_allowed",
        "message": "Transaction is no longer eligible for refund",
        "retryable": False,
    }
    assert transaction["status"] == "refunded"
    assert resumed["current_turn_tools"] == [
        "propose_refund",
        "execute_refund",
    ]
