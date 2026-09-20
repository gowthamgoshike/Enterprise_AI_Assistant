import asyncio
import json
from unittest.mock import Mock

from ai.nodes import refund_execution_node
from ai.schemas.actions import RefundExecutionResult
from services.transaction_service.exceptions import RefundNotAllowedError


def state(role: str) -> dict:
    return {
        "messages": [],
        "request_id": "REQ-1",
        "user_id": "USER-ADMIN-1",
        "user_role": role,
        "customer_id": None,
        "route": None,
        "current_turn_tools": [],
        "conversation_summary": None,
        "pending_action": {
            "action_id": "00000000-0000-0000-0000-000000000001",
            "action_type": "refund",
            "transaction_id": "TX-123",
            "amount_cents": 99999999,
            "currency": "UNTRUSTED",
            "reason": "duplicate charge",
            "requested_by": "USER-REQUESTER",
            "created_at": "2026-09-20T12:00:00Z",
        },
        "pending_tool_call_id": "refund-call-1",
        "approval_status": "approved",
        "error": None,
    }


def test_execution_node_rechecks_execute_permission(monkeypatch):
    execution = Mock()
    monkeypatch.setattr(refund_execution_node, "execute_refund", execution)

    result = asyncio.run(refund_execution_node.execute_refund_node(
        state("ANALYST")
    ))

    execution.assert_not_called()
    assert json.loads(result["messages"][0].content) == {
        "ok": False,
        "error": "permission_denied",
        "message": "Refund execution permission denied",
    }
    assert result["messages"][0].status == "error"


def test_execution_node_preserves_action_id_and_uses_structured_result(
    monkeypatch,
):
    execution = Mock(return_value=RefundExecutionResult(
        action_id="00000000-0000-0000-0000-000000000001",
        refund_id="00000000-0000-0000-0000-000000000002",
        transaction_id="TX-123",
        status="completed",
        amount_cents=8745,
        currency="USD",
    ))
    monkeypatch.setattr(refund_execution_node, "execute_refund", execution)

    result = asyncio.run(refund_execution_node.execute_refund_node(
        state("ADMIN")
    ))

    assert execution.call_args.kwargs == {
        "action_id": "00000000-0000-0000-0000-000000000001",
        "transaction_id": "TX-123",
        "reason": "duplicate charge",
        "requested_by": "USER-REQUESTER",
        "approved_by": "USER-ADMIN-1",
        "request_id": "REQ-1",
    }
    payload = json.loads(result["messages"][0].content)
    assert payload["refund"]["status"] == "completed"
    assert payload["refund"]["amount_cents"] == 8745
    assert result["current_turn_tools"] == [
        "propose_refund",
        "execute_refund",
    ]
    assert result["pending_action"] is None


def test_execution_node_reports_toctou_eligibility_failure(monkeypatch):
    execution = Mock(side_effect=RefundNotAllowedError(
        "Transaction is no longer eligible for refund"
    ))
    monkeypatch.setattr(refund_execution_node, "execute_refund", execution)

    result = asyncio.run(refund_execution_node.execute_refund_node(
        state("ADMIN")
    ))

    assert json.loads(result["messages"][0].content) == {
        "ok": False,
        "error": "refund_not_allowed",
        "message": "Transaction is no longer eligible for refund",
    }
    assert result["messages"][0].status == "error"
    assert result["pending_action"] is None
