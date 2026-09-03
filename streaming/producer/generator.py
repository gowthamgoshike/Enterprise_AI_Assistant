import random
import uuid

from datetime import datetime, timezone

from data.schemas.transaction import (
    Currency,
    TransactionEvent,
    TransactionSource,
    TransactionStatus,
    TransactionType,
)


def generate_transaction() -> TransactionEvent:

    now = datetime.now(timezone.utc)

    event = TransactionEvent(
        event_id=f"EVT-{uuid.uuid4()}",
        transaction_id=f"TX-{uuid.uuid4()}",
        customer_id=f"C-{random.randint(1, 1000)}",
        product_id=f"P-{random.randint(1, 100)}",
        amount_cents=random.randint(100, 100000),
        currency=Currency.USD,
        status=random.choice(
            list(TransactionStatus)
        ),
        transaction_type=TransactionType.PURCHASE,
        source=random.choice(
            list(TransactionSource)
        ),
        event_time=now,
        ingestion_time=now,
    )

    return event

if __name__ == "__main__":

    transaction = generate_transaction()

    print(
        transaction.model_dump_json(
            indent=2
        )
    )