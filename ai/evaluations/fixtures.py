"""Stable, unmistakably synthetic records for live evaluation runs."""

from copy import deepcopy
import os

from services.transaction_service.repository import upsert_transaction


EVALUATION_TRANSACTIONS = [
    {
        "transaction_id": "TX-EVAL-001",
        "latest_event_id": "EVENT-EVAL-001",
        "customer_id": "CUST-EVAL-001",
        "product_id": "PROD-EVAL-001",
        "amount_cents": 10000,
        "currency": "USD",
        "status": "completed",
        "transaction_type": "purchase",
        "source": "evaluation",
        "event_time": "2026-09-20T12:00:00+00:00",
    },
    {
        "transaction_id": "TX-EVAL-002",
        "latest_event_id": "EVENT-EVAL-002",
        "customer_id": "CUST-EVAL-002",
        "product_id": "PROD-EVAL-002",
        "amount_cents": 20000,
        "currency": "USD",
        "status": "failed",
        "transaction_type": "purchase",
        "source": "evaluation",
        "event_time": "2026-09-20T12:01:00+00:00",
    },
    {
        "transaction_id": "TX-EVAL-REFUND-001",
        "latest_event_id": "EVENT-EVAL-REFUND-001",
        "customer_id": "CUST-EVAL-001",
        "product_id": "PROD-EVAL-REFUND",
        "amount_cents": 4500,
        "currency": "USD",
        "status": "completed",
        "transaction_type": "purchase",
        "source": "evaluation",
        "event_time": "2026-09-20T12:02:00+00:00",
    },
]


def _require_safe_fixture_environment() -> None:
    environment = os.getenv("APP_ENV", "development").casefold()
    allowed = os.getenv("ALLOW_EVALUATION_FIXTURES", "false").casefold()
    if environment in {"production", "prod"}:
        raise RuntimeError("Evaluation fixtures are forbidden in production")
    if allowed not in {"1", "true", "yes"}:
        raise RuntimeError(
            "Set ALLOW_EVALUATION_FIXTURES=true only for a disposable test "
            "or staging database"
        )


def load_evaluation_fixtures() -> int:
    """Idempotently upsert the controlled fixture set into a non-production DB."""
    _require_safe_fixture_environment()
    for transaction in EVALUATION_TRANSACTIONS:
        upsert_transaction(deepcopy(transaction))
    return len(EVALUATION_TRANSACTIONS)


def fixture_by_id(transaction_id: str) -> dict:
    return deepcopy(next(
        transaction
        for transaction in EVALUATION_TRANSACTIONS
        if transaction["transaction_id"] == transaction_id
    ))


def apply_evaluation_fixture(action: str) -> None:
    """Apply a named deterministic setup/mutation requested by a golden case."""
    _require_safe_fixture_environment()
    if action == "status_completed":
        transaction = fixture_by_id("TX-EVAL-001")
    elif action == "refund_completed":
        transaction = fixture_by_id("TX-EVAL-REFUND-001")
    elif action == "eval_transactions":
        load_evaluation_fixtures()
        return
    elif action == "status_refunded":
        transaction = fixture_by_id("TX-EVAL-001")
        transaction.update({
            "latest_event_id": "EVENT-EVAL-001-REFUNDED",
            "status": "refunded",
            "event_time": "2026-09-20T12:05:00+00:00",
        })
    else:
        # Outage scenarios are intentionally supplied by integration-test
        # hooks; mutating global production dependencies here would be unsafe.
        return
    upsert_transaction(transaction)
