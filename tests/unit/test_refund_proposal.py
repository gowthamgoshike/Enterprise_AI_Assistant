from inspect import signature
from unittest.mock import Mock

import pytest

from services.transaction_service import refund_service


def completed_transaction(*, amount_cents: int = 8745) -> dict:
    return {
        "transaction_id": "TX-REFUND-TEST",
        "amount_cents": amount_cents,
        "currency": "USD",
        "status": "completed",
    }


def test_completed_transaction_creates_valid_proposal(monkeypatch):
    lookup = Mock(return_value={
        "found": True,
        "transaction": completed_transaction(),
    })
    monkeypatch.setattr(refund_service, "find_transaction", lookup)

    proposal = refund_service.prepare_refund_proposal(
        transaction_id="tx-refund-test",
        reason="duplicate charge",
        requested_by="USER-ADMIN-1",
    )

    lookup.assert_called_once_with("TX-REFUND-TEST")
    assert proposal.action_type == "refund"
    assert proposal.transaction_id == "TX-REFUND-TEST"
    assert proposal.amount_cents == 8745
    assert proposal.currency == "USD"
    assert proposal.reason == "duplicate charge"
    assert proposal.requested_by == "USER-ADMIN-1"


def test_missing_transaction_fails(monkeypatch):
    monkeypatch.setattr(
        refund_service,
        "find_transaction",
        lambda _: {"found": False, "transaction": None},
    )

    with pytest.raises(ValueError, match="Transaction unavailable"):
        refund_service.prepare_refund_proposal(
            "TX-MISSING",
            "duplicate charge",
            "USER-ADMIN-1",
        )


def test_already_refunded_transaction_fails(monkeypatch):
    transaction = completed_transaction()
    transaction["status"] = "refunded"
    monkeypatch.setattr(refund_service, "find_transaction", lambda _: {
        "found": True,
        "transaction": transaction,
    })

    with pytest.raises(ValueError, match="already refunded"):
        refund_service.prepare_refund_proposal(
            "TX-REFUND-TEST",
            "duplicate charge",
            "USER-ADMIN-1",
        )


def test_amount_comes_from_repository_not_user_or_model(monkeypatch):
    repository_amount = 4321
    monkeypatch.setattr(refund_service, "find_transaction", lambda _: {
        "found": True,
        "transaction": completed_transaction(amount_cents=repository_amount),
    })

    proposal = refund_service.prepare_refund_proposal(
        transaction_id="TX-REFUND-TEST",
        reason="duplicate charge",
        requested_by="USER-ADMIN-1",
    )

    assert "amount_cents" not in signature(
        refund_service.prepare_refund_proposal
    ).parameters
    assert proposal.amount_cents == repository_amount
