from datetime import datetime, timezone

from data.schemas.transaction import (
    Currency,
    TransactionEvent,
    TransactionSource,
    TransactionStatus,
    TransactionType,
)


now = datetime.now(timezone.utc)


transaction = TransactionEvent(
    event_id="EVT-1001",
    transaction_id="TX-1001",
    customer_id="C-101",
    product_id="P-501",
    amount_cents=2599,
    currency=Currency.USD,
    status=TransactionStatus.COMPLETED,
    transaction_type=TransactionType.PURCHASE,
    source=TransactionSource.WEB,
    event_time=now,
    ingestion_time=now,
)


print(transaction)

print(
    transaction.model_dump_json(
        indent=2
    )
)