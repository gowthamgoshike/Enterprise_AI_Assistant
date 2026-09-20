from datetime import datetime, timezone
from uuid import uuid4

from ai.schemas.actions import RefundProposal
from services.transaction_service.identifiers import normalize_transaction_id
from services.transaction_service.service import find_transaction


def prepare_refund_proposal(
    transaction_id: str,
    reason: str,
    requested_by: str,
) -> RefundProposal:
    """Build a proposal from trusted transaction data without changing it."""
    result = find_transaction(normalize_transaction_id(transaction_id))

    if not result["found"]:
        raise ValueError("Transaction unavailable")

    transaction = result["transaction"]
    if transaction["status"] == "refunded":
        raise ValueError("Transaction is already refunded")
    if transaction["status"] != "completed":
        raise ValueError("Transaction is not eligible for refund")

    return RefundProposal(
        action_id=str(uuid4()),
        transaction_id=transaction["transaction_id"],
        amount_cents=transaction["amount_cents"],
        currency=transaction["currency"],
        reason=reason,
        requested_by=requested_by,
        created_at=datetime.now(timezone.utc),
    )
