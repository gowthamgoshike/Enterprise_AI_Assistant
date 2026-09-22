from typing import Protocol
from uuid import UUID, uuid4

from repositories import job_repository
from services.job_service.models import (
    FailedTransactionReportParams,
    JobResponse,
    JobType,
)


class RequestingUser(Protocol):
    user_id: str


def create_failed_report_job(
    *,
    user: RequestingUser,
    request_id: str | None,
    days: int,
) -> UUID:
    params = FailedTransactionReportParams(days=days)
    job_id = uuid4()

    job_repository.create_job(
        job_id=job_id,
        job_type=JobType.FAILED_TRANSACTION_REPORT,
        requested_by=user.user_id,
        request_id=request_id,
        parameters=params.model_dump(),
    )
    return job_id


def get_job_for_user(
    job_id: UUID,
    user: RequestingUser,
) -> JobResponse | None:
    row = job_repository.get_job_for_user(
        job_id,
        requested_by=user.user_id,
    )
    return JobResponse.model_validate(row) if row else None


def cancel_job_for_user(
    job_id: UUID,
    user: RequestingUser,
) -> JobResponse | None:
    row = job_repository.get_job_for_user(
        job_id,
        requested_by=user.user_id,
    )
    if row is None:
        return None

    job_repository.cancel_queued_job_for_user(
        job_id,
        requested_by=user.user_id,
    )
    row = job_repository.get_job_for_user(
        job_id,
        requested_by=user.user_id,
    )
    return JobResponse.model_validate(row)
