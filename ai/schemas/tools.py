from datetime import datetime

from pydantic import BaseModel


class TransactionData(BaseModel):

    transaction_id: str

    customer_id: str

    product_id: str

    amount_cents: int

    currency: str

    status: str

    transaction_type: str

    source: str

    event_time: datetime


class TransactionLookupResult(BaseModel):

    found: bool

    transaction: TransactionData | None


class ToolError(BaseModel):

    code: str

    message: str

    retryable: bool = False


class ToolResult(BaseModel):

    success: bool

    tool_name: str

    data: dict | None = None

    error: ToolError | None = None
