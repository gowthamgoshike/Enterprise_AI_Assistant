from enum import Enum
from datetime import datetime, timezone

from pydantic import BaseModel, Field

class TransactionStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


class TransactionType(str, Enum):
    PURCHASE = "purchase"
    REFUND = "refund"


class TransactionSource(str, Enum):
    WEB = "web"
    MOBILE = "mobile"
    POS = "pos"

class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"

class TransactionEvent(BaseModel):

    event_id: str

    schema_version: str = "1.0"

    transaction_id: str

    customer_id: str

    product_id: str

    amount_cents: int = Field(gt=0)

    currency: Currency = Currency.USD

    status: TransactionStatus

    transaction_type: TransactionType

    source: TransactionSource

    event_time: datetime

    ingestion_time: datetime