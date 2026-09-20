import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai.graphs import tool_agent_graph
from ai.nodes import secure_tool_node as secure_node_module
from ai.nodes import refund_execution_node
from ai.schemas.actions import RefundExecutionResult
from ai.tools.secure_registry import ToolDefinition
from services.auth_service.models import UserRole
from services.auth_service.permissions import Permission
from services.transaction_service import refund_service


def secure_state(role: UserRole) -> dict:
    return {
        "messages": [AIMessage(content="", tool_calls=[{
            "name": "propose_refund",
            "args": {
                "transaction_id": "TX-123",
                "reason": "duplicate charge",
                "amount_cents": 99999999,
                "currency": "ATTACKER-CURRENCY",
                "requested_by": "USER-ATTACKER",
            },
            "id": "refund-call-1",
        }])],
        "request_id": "REQ-refund",
        "user_id": "USER-ADMIN",
        "user_role": role.value,
        "customer_id": None,
        "route": None,
        "current_turn_tools": [],
        "conversation_summary": None,
        "pending_action": None,
        "pending_tool_call_id": None,
        "approval_status": None,
        "error": None,
    }


def trusted_proposal() -> dict:
    return {
        "action_id": "ACTION-1",
        "action_type": "refund",
        "transaction_id": "TX-123",
        "amount_cents": 8745,
        "currency": "USD",
        "reason": "duplicate charge",
        "requested_by": "USER-ADMIN",
        "created_at": "2026-09-20T12:00:00Z",
    }


def test_refund_proposal_uses_trusted_transaction_fields(monkeypatch):
    lookup = Mock(return_value={
        "found": True,
        "transaction": {
            "transaction_id": "TX-123",
            "amount_cents": 8745,
            "currency": "USD",
            "status": "completed",
        },
    })
    monkeypatch.setattr(refund_service, "find_transaction", lookup)

    proposal = refund_service.prepare_refund_proposal(
        transaction_id="tx-123",
        reason="duplicate charge",
        requested_by="USER-ADMIN",
    )

    lookup.assert_called_once_with("TX-123")
    assert proposal.amount_cents == 8745
    assert proposal.currency == "USD"
    assert proposal.requested_by == "USER-ADMIN"
    assert proposal.reason == "duplicate charge"


@pytest.mark.parametrize(
    ("status", "message"),
    [
        ("refunded", "already refunded"),
        ("pending", "not eligible"),
        ("failed", "not eligible"),
    ],
)
def test_refund_proposal_rejects_ineligible_transactions(
    monkeypatch,
    status,
    message,
):
    monkeypatch.setattr(refund_service, "find_transaction", lambda _: {
        "found": True,
        "transaction": {
            "transaction_id": "TX-123",
            "amount_cents": 8745,
            "currency": "USD",
            "status": status,
        },
    })

    with pytest.raises(ValueError, match=message):
        refund_service.prepare_refund_proposal(
            "TX-123",
            "duplicate charge",
            "USER-ADMIN",
        )


def test_secure_node_prepares_admin_action_without_completing_tool(
    monkeypatch,
    caplog,
):
    executor = Mock(return_value=trusted_proposal())
    monkeypatch.setitem(
        secure_node_module.SECURE_TOOL_REGISTRY,
        "propose_refund",
        ToolDefinition(
            name="propose_refund",
            function=executor,
            permission=Permission.PROPOSE_REFUND,
            requires_approval=True,
        ),
    )

    with caplog.at_level(logging.INFO, logger=secure_node_module.__name__):
        result = asyncio.run(
            secure_node_module.secure_tool_node(secure_state(UserRole.ADMIN))
        )

    assert result["messages"] == []
    assert result["current_turn_tools"] == []
    assert result["pending_action"] == trusted_proposal()
    assert result["pending_tool_call_id"] == "refund-call-1"
    assert result["approval_status"] == "pending"
    executor.assert_called_once()
    call_arguments = executor.call_args.kwargs
    assert set(call_arguments) == {"context", "transaction_id", "reason"}
    assert call_arguments["context"].user_id == "USER-ADMIN"
    approval_record = next(
        record for record in caplog.records
        if record.msg.startswith("human_approval_requested")
    )
    assert approval_record.request_id == "REQ-refund"
    assert approval_record.user_id == "USER-ADMIN"
    assert approval_record.action_id == "ACTION-1"
    assert approval_record.action_type == "refund"
    assert "duplicate charge" not in caplog.text
    assert "8745" not in caplog.text


def test_secure_node_denies_customer_refund_before_executor(monkeypatch):
    executor = Mock()
    monkeypatch.setitem(
        secure_node_module.SECURE_TOOL_REGISTRY,
        "propose_refund",
        ToolDefinition(
            name="propose_refund",
            function=executor,
            permission=Permission.PROPOSE_REFUND,
            requires_approval=True,
        ),
    )

    result = asyncio.run(
        secure_node_module.secure_tool_node(secure_state(UserRole.CUSTOMER))
    )

    assert json.loads(result["messages"][0].content) == {
        "ok": False,
        "data": None,
        "error_code": "permission_denied",
        "message": "You are not authorized to perform this action.",
        "retryable": False,
    }
    assert result["messages"][0].status == "error"
    assert result.get("pending_action") is None
    executor.assert_not_called()


@pytest.mark.parametrize(
    ("approved", "expected_status", "tool_status", "expected_tools"),
    [
        (
            True,
            "completed",
            "success",
            ["propose_refund", "execute_refund"],
        ),
        (False, "rejected_by_human", "error", ["propose_refund"]),
    ],
)
def test_graph_interrupts_and_resumes_approval(
    monkeypatch,
    caplog,
    approved,
    expected_status,
    tool_status,
    expected_tools,
):
    executor = Mock(return_value=trusted_proposal())
    monkeypatch.setitem(
        secure_node_module.SECURE_TOOL_REGISTRY,
        "propose_refund",
        ToolDefinition(
            name="propose_refund",
            function=executor,
            permission=Permission.PROPOSE_REFUND,
            requires_approval=True,
        ),
    )
    model = AsyncMock()
    model.ainvoke.side_effect = [
        secure_state(UserRole.ADMIN)["messages"][0],
        AIMessage(content="The approval decision was recorded."),
    ]
    monkeypatch.setattr(tool_agent_graph, "model_with_tools", model)
    execution = Mock(return_value=RefundExecutionResult(
        action_id="ACTION-1",
        refund_id="REFUND-1",
        transaction_id="TX-123",
        status="completed",
        amount_cents=8745,
        currency="USD",
    ))
    monkeypatch.setattr(refund_execution_node, "execute_refund", execution)
    compiled = tool_agent_graph.build_enterprise_tool_graph(InMemorySaver())

    async def run_workflow():
        interrupted = await tool_agent_graph.run_agent_graph(
            "Refund TX-123 because it was duplicated.",
            "REQ-refund",
            "THREAD-refund",
            graph=compiled,
            user_id="USER-ADMIN",
            user_role="ADMIN",
            customer_id=None,
        )
        resumed = await tool_agent_graph.resume_agent_graph(
            "THREAD-refund",
            {"approved": approved, "comment": "Reviewed"},
            graph=compiled,
        )
        return interrupted, resumed

    with caplog.at_level(logging.INFO):
        interrupted, resumed = asyncio.run(run_workflow())

    interrupt_payload = interrupted["__interrupt__"][0].value
    assert interrupt_payload["type"] == "action_approval"
    assert interrupt_payload["action"] == trusted_proposal()
    tool_message = next(
        message for message in resumed["messages"]
        if isinstance(message, ToolMessage)
    )
    tool_payload = json.loads(tool_message.content)
    actual_status = (
            tool_payload["data"]["refund"]["status"]
        if approved
        else tool_payload["status"]
    )
    assert actual_status == expected_status
    assert tool_message.status == tool_status
    assert resumed["approval_status"] == (
        "approved" if approved else "rejected"
    )
    assert resumed["pending_action"] is None
    assert resumed["current_turn_tools"] == expected_tools
    executor.assert_called_once()
    if approved:
        execution.assert_called_once()
        assert execution.call_args.kwargs["action_id"] == "ACTION-1"
    else:
        execution.assert_not_called()
    decision_record = next(
        record for record in caplog.records
        if record.msg.startswith("human_approval_resolved")
    )
    assert decision_record.request_id == "REQ-refund"
    assert decision_record.user_id == "USER-ADMIN"
    assert decision_record.action_id == "ACTION-1"
    assert decision_record.action_type == "refund"
    assert decision_record.approval == (
        "APPROVED" if approved else "REJECTED"
    )
    assert f"approval={decision_record.approval}" in caplog.text
    assert "Reviewed" not in caplog.text
    assert "duplicate charge" not in caplog.text
    assert "8745" not in caplog.text


def test_resume_rejects_session_without_pending_interrupt():
    compiled = AsyncMock()
    compiled.aget_state.return_value = SimpleNamespace(tasks=())

    with pytest.raises(
        tool_agent_graph.PendingApprovalNotFoundError,
        match="No pending approval",
    ):
        asyncio.run(tool_agent_graph.resume_agent_graph(
            "THREAD-without-approval",
            {"approved": True},
            graph=compiled,
        ))

    compiled.ainvoke.assert_not_awaited()


def test_resume_retries_failed_approved_execution():
    failed_task = SimpleNamespace(
        name="execute_refund",
        error="database unavailable",
        interrupts=(),
    )
    compiled = AsyncMock()
    compiled.aget_state.return_value = SimpleNamespace(
        tasks=(failed_task,),
        next=("execute_refund",),
        values={"approval_status": "approved"},
    )
    compiled.ainvoke.return_value = {"messages": []}

    result = asyncio.run(tool_agent_graph.resume_agent_graph(
        "THREAD-failed-execution",
        {"approved": True},
        graph=compiled,
    ))

    assert result == {"messages": []}
    assert compiled.ainvoke.await_args.args[0] is None
