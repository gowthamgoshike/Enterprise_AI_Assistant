import asyncio
from copy import deepcopy
import json
from unittest.mock import AsyncMock, Mock

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai.graphs import tool_agent_graph
from services.transaction_service import refund_service
from services.transaction_service import repository as transaction_repository


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
    database_write = Mock()
    monkeypatch.setattr(refund_service, "find_transaction", lookup)
    monkeypatch.setattr(
        transaction_repository,
        "upsert_transaction",
        database_write,
    )

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
        database_write.assert_not_called()

        resumed = await tool_agent_graph.resume_agent_graph(
            thread_id="THREAD-HITL",
            approval={"approved": approved, "comment": "Reviewed"},
            graph=graph,
        )
        return interrupted, resumed

    interrupted, resumed = asyncio.run(run())
    assert transaction == original_transaction
    database_write.assert_not_called()
    lookup.assert_called_once_with("TX-HITL-TEST")
    return interrupted, resumed


def approval_result(resumed: dict) -> dict:
    message = next(
        message
        for message in resumed["messages"]
        if isinstance(message, ToolMessage)
        and message.tool_call_id == "refund-hitl-call"
    )
    return json.loads(message.content)


def test_approve_interrupt_does_not_change_transaction(monkeypatch):
    interrupted, resumed = run_refund_workflow(
        monkeypatch,
        approved=True,
    )

    proposal = interrupted["__interrupt__"][0].value["action"]
    assert proposal["transaction_id"] == "TX-HITL-TEST"
    assert proposal["amount_cents"] == 2500
    assert approval_result(resumed)["status"] == "approved_for_execution"
    assert resumed["approval_status"] == "approved"


def test_reject_interrupt_does_not_change_transaction(monkeypatch):
    interrupted, resumed = run_refund_workflow(
        monkeypatch,
        approved=False,
    )

    assert interrupted["__interrupt__"][0].value["type"] == (
        "action_approval"
    )
    assert approval_result(resumed)["status"] == "rejected_by_human"
    assert resumed["approval_status"] == "rejected"
