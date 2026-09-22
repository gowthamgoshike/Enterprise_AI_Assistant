import asyncio
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    status,
)

from apps.api.dependencies.auth import get_current_user
from apps.api.schemas.auth import AuthenticatedUser
from apps.api.schemas.jobs import FailedReportRequest
from services.auth_service.authorization_service import require_permission
from services.auth_service.exceptions import AuthorizationError
from services.auth_service.permissions import Permission
from services.job_service.executor import execute_job
from services.job_service.job_service import (
    cancel_job_for_user,
    create_failed_report_job,
    get_job_for_user,
)
from services.job_service.models import JobCreateResponse, JobResponse, JobStatus


router = APIRouter(prefix="/jobs", tags=["background jobs"])


def _require_report_permission(user: AuthenticatedUser) -> None:
    try:
        require_permission(user.role, Permission.GENERATE_ANALYTICS_REPORT)
    except AuthorizationError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied",
        ) from exc


@router.post(
    "/failed-transactions",
    response_model=JobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_failed_report(
    payload: FailedReportRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> JobCreateResponse:
    _require_report_permission(current_user)
    job_id = await asyncio.to_thread(
        create_failed_report_job,
        user=current_user,
        request_id=request.state.request_id,
        days=payload.days,
    )
    background_tasks.add_task(execute_job, job_id)
    return JobCreateResponse(job_id=job_id, status=JobStatus.QUEUED)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> JobResponse:
    job = await asyncio.to_thread(get_job_for_user, job_id, current_user)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    return job


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    job_id: UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> JobResponse:
    job = await asyncio.to_thread(cancel_job_for_user, job_id, current_user)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    if job.status is not JobStatus.CANCELLED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only queued jobs can be cancelled",
        )
    return job
