from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(StrEnum):
    FAILED_TRANSACTION_REPORT = "failed_transaction_report"


class JobStage(StrEnum):
    QUEUED = "queued"
    QUERYING_TRANSACTIONS = "querying_transactions"
    CALCULATING_METRICS = "calculating_metrics"
    GENERATING_SUMMARY = "generating_summary"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FailedTransactionReportParams(BaseModel):
    days: int = Field(default=30, ge=1, le=90)


class JobResponse(BaseModel):
    job_id: UUID
    job_type: str
    status: JobStatus
    stage: JobStage
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error_code: str | None = None


class JobCreateResponse(BaseModel):
    job_id: UUID
    status: JobStatus
