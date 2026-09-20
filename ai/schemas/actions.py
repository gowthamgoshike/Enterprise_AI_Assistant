from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class RefundProposal(BaseModel):
    action_id: str
    action_type: Literal["refund"] = "refund"
    transaction_id: str
    amount_cents: int
    currency: str
    reason: str
    requested_by: str
    created_at: datetime


class ApprovalDecision(BaseModel):
    approved: bool
    comment: str | None = None


class RefundExecutionResult(BaseModel):
    action_id: str
    refund_id: str
    transaction_id: str
    status: Literal[
        "completed",
        "already_completed",
    ]
    amount_cents: int
    currency: str
