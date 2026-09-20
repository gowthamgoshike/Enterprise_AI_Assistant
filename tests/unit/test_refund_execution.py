from contextlib import nullcontext
from inspect import signature
from unittest.mock import Mock

import pytest

from services.transaction_service import refund_execution_service as service
from services.transaction_service.exceptions import (
    RefundNotAllowedError,
    TransactionNotFoundError,
)


def configure_transaction(monkeypatch):
    connection = Mock(name="connection")
    fake_engine = Mock()
    fake_engine.begin.return_value = nullcontext(connection)
    monkeypatch.setattr(service, "engine", fake_engine)
    return connection, fake_engine


def existing_refund() -> dict:
    return {
        "refund_id": "00000000-0000-0000-0000-000000000002",
        "action_id": "00000000-0000-0000-0000-000000000001",
        "transaction_id": "TX-123",
        "amount_cents": 8745,
        "currency": "USD",
        "status": "completed",
    }


def execute() -> object:
    return service.execute_refund(
        action_id="00000000-0000-0000-0000-000000000001",
        transaction_id="tx-123",
        reason="duplicate charge",
        requested_by="USER-REQUESTER",
        approved_by="USER-APPROVER",
        request_id="REQ-1",
    )


def test_execute_refund_is_atomic_and_uses_current_transaction(monkeypatch):
    connection, fake_engine = configure_transaction(monkeypatch)
    find_existing = Mock(return_value=None)
    transaction = {
        "transaction_id": "TX-123",
        "amount_cents": 4321,
        "currency": "EUR",
        "status": "completed",
        "transaction_type": "purchase",
    }
    lock_transaction = Mock(return_value=transaction)
    create_refund = Mock(return_value=True)
    update_transaction = Mock(return_value=1)
    create_audit = Mock()
    monkeypatch.setattr(service, "find_refund_by_action_id", find_existing)
    monkeypatch.setattr(service, "find_transaction_for_update", lock_transaction)
    monkeypatch.setattr(service, "insert_refund", create_refund)
    monkeypatch.setattr(
        service,
        "mark_transaction_refunded",
        update_transaction,
    )
    monkeypatch.setattr(service, "insert_audit_log", create_audit)

    result = execute()

    fake_engine.begin.assert_called_once_with()
    lock_transaction.assert_called_once_with(connection, "TX-123")
    refund_values = create_refund.call_args.args[1]
    assert refund_values["amount_cents"] == 4321
    assert refund_values["currency"] == "EUR"
    assert refund_values["approved_by"] == "USER-APPROVER"
    update_transaction.assert_called_once_with(connection, "TX-123")
    audit_values = create_audit.call_args.args[1]
    assert audit_values["action_id"] == result.action_id
    assert audit_values["user_id"] == "USER-APPROVER"
    assert result.status == "completed"
    assert result.amount_cents == 4321
    assert result.currency == "EUR"
    assert "amount_cents" not in signature(service.execute_refund).parameters


def test_retry_returns_existing_refund_without_writes(monkeypatch):
    connection, _ = configure_transaction(monkeypatch)
    find_existing = Mock(return_value=existing_refund())
    lock_transaction = Mock()
    create_refund = Mock()
    update_transaction = Mock()
    create_audit = Mock()
    monkeypatch.setattr(service, "find_refund_by_action_id", find_existing)
    monkeypatch.setattr(service, "find_transaction_for_update", lock_transaction)
    monkeypatch.setattr(service, "insert_refund", create_refund)
    monkeypatch.setattr(
        service,
        "mark_transaction_refunded",
        update_transaction,
    )
    monkeypatch.setattr(service, "insert_audit_log", create_audit)

    result = execute()

    find_existing.assert_called_once_with(
        connection,
        "00000000-0000-0000-0000-000000000001",
    )
    assert result.status == "already_completed"
    lock_transaction.assert_not_called()
    create_refund.assert_not_called()
    update_transaction.assert_not_called()
    create_audit.assert_not_called()


def test_missing_transaction_rolls_out_without_writes(monkeypatch):
    _, _ = configure_transaction(monkeypatch)
    monkeypatch.setattr(service, "find_refund_by_action_id", Mock(return_value=None))
    monkeypatch.setattr(
        service,
        "find_transaction_for_update",
        Mock(return_value=None),
    )
    create_refund = Mock()
    create_audit = Mock()
    monkeypatch.setattr(service, "insert_refund", create_refund)
    monkeypatch.setattr(service, "insert_audit_log", create_audit)

    with pytest.raises(TransactionNotFoundError, match="unavailable"):
        execute()

    create_refund.assert_not_called()
    create_audit.assert_not_called()


def test_non_purchase_or_stale_transaction_is_rejected(
    monkeypatch,
):
    configure_transaction(monkeypatch)
    find_existing = Mock(return_value=None)
    monkeypatch.setattr(service, "find_refund_by_action_id", find_existing)
    monkeypatch.setattr(
        service,
        "find_transaction_for_update",
        Mock(return_value={
            "transaction_id": "TX-123",
            "amount_cents": 100,
            "currency": "USD",
            "status": "completed",
            "transaction_type": "refund",
        }),
    )

    with pytest.raises(RefundNotAllowedError, match="Only purchase"):
        execute()


def test_concurrent_retry_after_row_lock_returns_existing(monkeypatch):
    connection, _ = configure_transaction(monkeypatch)
    find_existing = Mock(side_effect=[None, existing_refund()])
    monkeypatch.setattr(service, "find_refund_by_action_id", find_existing)
    monkeypatch.setattr(
        service,
        "find_transaction_for_update",
        Mock(return_value={
            "transaction_id": "TX-123",
            "amount_cents": 8745,
            "currency": "USD",
            "status": "refunded",
            "transaction_type": "purchase",
        }),
    )

    result = execute()

    assert result.status == "already_completed"
    assert find_existing.call_count == 2
    assert all(call.args[0] is connection for call in find_existing.call_args_list)


def test_unique_action_conflict_returns_existing_without_update(monkeypatch):
    configure_transaction(monkeypatch)
    find_existing = Mock(side_effect=[None, existing_refund()])
    monkeypatch.setattr(service, "find_refund_by_action_id", find_existing)
    monkeypatch.setattr(
        service,
        "find_transaction_for_update",
        Mock(return_value={
            "transaction_id": "TX-123",
            "amount_cents": 8745,
            "currency": "USD",
            "status": "completed",
            "transaction_type": "purchase",
        }),
    )
    monkeypatch.setattr(service, "insert_refund", Mock(return_value=False))
    update_transaction = Mock()
    create_audit = Mock()
    monkeypatch.setattr(
        service,
        "mark_transaction_refunded",
        update_transaction,
    )
    monkeypatch.setattr(service, "insert_audit_log", create_audit)

    result = execute()

    assert result.status == "already_completed"
    update_transaction.assert_not_called()
    create_audit.assert_not_called()


def test_failed_conditional_update_prevents_audit(monkeypatch):
    configure_transaction(monkeypatch)
    monkeypatch.setattr(service, "find_refund_by_action_id", Mock(return_value=None))
    monkeypatch.setattr(
        service,
        "find_transaction_for_update",
        Mock(return_value={
            "transaction_id": "TX-123",
            "amount_cents": 8745,
            "currency": "USD",
            "status": "completed",
            "transaction_type": "purchase",
        }),
    )
    monkeypatch.setattr(service, "insert_refund", Mock(return_value=True))
    monkeypatch.setattr(
        service,
        "mark_transaction_refunded",
        Mock(return_value=0),
    )
    create_audit = Mock()
    monkeypatch.setattr(service, "insert_audit_log", create_audit)

    with pytest.raises(RefundNotAllowedError, match="state changed"):
        execute()

    create_audit.assert_not_called()
