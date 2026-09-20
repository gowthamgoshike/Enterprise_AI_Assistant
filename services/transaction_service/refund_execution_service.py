import json
from uuid import uuid4

from ai.schemas.actions import RefundExecutionResult
from data.database.connection import engine
from services.transaction_service.exceptions import (
    RefundError,
    RefundNotAllowedError,
    TransactionNotFoundError,
)
from services.transaction_service.identifiers import normalize_transaction_id
from services.transaction_service.refund_repository import (
    find_refund_by_action_id,
    find_transaction_for_update,
    insert_audit_log,
    insert_refund,
    mark_transaction_refunded,
)


def _existing_result(existing) -> RefundExecutionResult:
    return RefundExecutionResult(
        action_id=str(existing["action_id"]),
        refund_id=str(existing["refund_id"]),
        transaction_id=existing["transaction_id"],
        status="already_completed",
        amount_cents=existing["amount_cents"],
        currency=existing["currency"],
    )


def execute_refund(
    action_id: str,
    transaction_id: str,
    reason: str,
    requested_by: str,
    approved_by: str,
    request_id: str,
) -> RefundExecutionResult:
    transaction_id = normalize_transaction_id(transaction_id)

    with engine.begin() as connection:
        existing = find_refund_by_action_id(connection, action_id)
        if existing:
            return _existing_result(existing)

        transaction = find_transaction_for_update(connection, transaction_id)
        if not transaction:
            raise TransactionNotFoundError("Transaction unavailable")
        if transaction["status"] != "completed":
            # A concurrent request with this action_id may have completed while
            # this request waited for the transaction row lock.
            existing = find_refund_by_action_id(connection, action_id)
            if existing:
                return _existing_result(existing)
            raise RefundNotAllowedError(
                "Transaction is no longer eligible for refund"
            )
        if transaction["transaction_type"] != "purchase":
            raise RefundNotAllowedError(
                "Only purchase transactions can be refunded"
            )

        amount_cents = transaction["amount_cents"]
        currency = transaction["currency"]
        refund_id = str(uuid4())
        inserted = insert_refund(connection, {
            "refund_id": refund_id,
            "action_id": action_id,
            "transaction_id": transaction_id,
            "amount_cents": amount_cents,
            "currency": currency,
            "reason": reason,
            "requested_by": requested_by,
            "approved_by": approved_by,
        })
        if not inserted:
            existing = find_refund_by_action_id(connection, action_id)
            if not existing:
                raise RefundError("Refund idempotency conflict")
            return _existing_result(existing)

        if mark_transaction_refunded(connection, transaction_id) != 1:
            raise RefundNotAllowedError(
                "Transaction state changed before refund execution"
            )

        insert_audit_log(connection, {
            "audit_id": str(uuid4()),
            "request_id": request_id,
            "action_id": action_id,
            "user_id": approved_by,
            "transaction_id": transaction_id,
            "metadata": json.dumps({
                "amount_cents": amount_cents,
                "currency": currency,
            }),
        })

        return RefundExecutionResult(
            action_id=action_id,
            refund_id=refund_id,
            transaction_id=transaction_id,
            status="completed",
            amount_cents=amount_cents,
            currency=currency,
        )
