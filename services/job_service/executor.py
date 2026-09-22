from collections import defaultdict
import logging
import time
from typing import Any
from uuid import UUID

from core.errors import AppError, ModelUnavailableError, ToolTimeoutError
from data.database.reliability import is_database_unavailable_error
from repositories import job_repository
from services.job_service.exceptions import UnknownJobTypeError
from services.job_service.models import (
    FailedTransactionReportParams,
    JobStage,
    JobStatus,
    JobType,
)
from services.transaction_service import repository as transaction_repository


logger = logging.getLogger(__name__)
FAILED_REPORT_TIMEOUT_SECONDS = 120


def calculate_failure_metrics(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    products: dict[str, dict[str, int]] = defaultdict(
        lambda: {"failed_count": 0, "total_failed_amount_cents": 0}
    )
    total_amount_cents = 0

    for transaction in transactions:
        amount_cents = int(transaction["amount_cents"])
        total_amount_cents += amount_cents
        product = products[str(transaction["product_id"])]
        product["failed_count"] += 1
        product["total_failed_amount_cents"] += amount_cents

    top_products = [
        {"product_id": product_id, **metrics}
        for product_id, metrics in products.items()
    ]
    top_products.sort(
        key=lambda item: (
            -item["failed_count"],
            -item["total_failed_amount_cents"],
            item["product_id"],
        )
    )

    return {
        "failed_count": len(transactions),
        "total_failed_amount_cents": total_amount_cents,
        "top_failure_products": top_products[:5],
    }


def _ensure_before_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("Background job exceeded its runtime limit")


def execute_failed_report(
    job: dict[str, Any],
    *,
    deadline: float,
) -> dict[str, Any]:
    params = FailedTransactionReportParams.model_validate(job["parameters"])
    _ensure_before_deadline(deadline)
    job_repository.update_stage(
        job["job_id"],
        JobStage.QUERYING_TRANSACTIONS,
    )
    transactions = transaction_repository.find_failed_transactions(
        days=params.days,
        timeout_ms=FAILED_REPORT_TIMEOUT_SECONDS * 1000,
    )
    _ensure_before_deadline(deadline)
    job_repository.update_stage(
        job["job_id"],
        JobStage.CALCULATING_METRICS,
    )
    metrics = calculate_failure_metrics(transactions)
    _ensure_before_deadline(deadline)
    return {
        "days": params.days,
        **metrics,
    }


def execute_by_type(job: dict[str, Any]) -> dict[str, Any]:
    try:
        job_type = JobType(job["job_type"])
    except ValueError as exc:
        raise UnknownJobTypeError() from exc

    if job_type is JobType.FAILED_TRANSACTION_REPORT:
        return execute_failed_report(
            job,
            deadline=time.monotonic() + FAILED_REPORT_TIMEOUT_SECONDS,
        )
    raise UnknownJobTypeError()


def classify_job_error(error: BaseException) -> str:
    if is_database_unavailable_error(error):
        return "database_unavailable"
    if isinstance(error, ModelUnavailableError):
        return "model_unavailable"
    if isinstance(error, (TimeoutError, ToolTimeoutError)):
        return "job_timeout"
    if isinstance(error, AppError):
        return error.code
    return "report_generation_failed"


def execute_job(job_id: UUID) -> None:
    job = job_repository.get_job(job_id)
    if job is None or job["status"] != JobStatus.QUEUED:
        return

    # The conditional update is the claim: only one runner can transition a
    # queued job to running and proceed with execution.
    if not job_repository.mark_running(job_id):
        return

    started = time.monotonic()
    log_context = {
        "request_id": job.get("request_id"),
        "user_id": job.get("requested_by"),
        "job_id": str(job_id),
        "job_type": str(job["job_type"]),
    }
    logger.info(
        "background_job_started",
        extra={**log_context, "event": "job_started", "status": "running"},
    )
    try:
        result = execute_by_type(job)
        job_repository.mark_completed(job_id, result=result)
        logger.info(
            "background_job_completed",
            extra={
                **log_context,
                "event": "job_completed",
                "status": "completed",
                "latency_ms": round((time.monotonic() - started) * 1000, 2),
            },
        )
    except Exception as exc:
        error_code = classify_job_error(exc)
        logger.exception(
            "background_job_failed",
            extra={
                **log_context,
                "event": "job_failed",
                "status": "failed",
                "error_code": error_code,
                "latency_ms": round((time.monotonic() - started) * 1000, 2),
            },
        )
        job_repository.mark_failed(
            job_id,
            error_code=error_code,
        )
