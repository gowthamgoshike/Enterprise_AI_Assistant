from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from apps.api.schemas.auth import AuthenticatedUser
from repositories import job_repository
from services.job_service import executor, job_service
from services.job_service.models import (
    FailedTransactionReportParams,
    JobStatus,
    JobType,
)
from services.transaction_service import repository as transaction_repository


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_failed_report_parameters_are_bounded():
    assert FailedTransactionReportParams().days == 30
    assert FailedTransactionReportParams(days=90).days == 90

    with pytest.raises(ValidationError):
        FailedTransactionReportParams(days=0)
    with pytest.raises(ValidationError):
        FailedTransactionReportParams(days=91)


def test_create_failed_report_job_uses_authenticated_user(monkeypatch):
    persisted = {}
    monkeypatch.setattr(
        job_repository,
        "create_job",
        lambda **values: persisted.update(values),
    )
    user = AuthenticatedUser(
        user_id="USER-ANALYST",
        email="analyst@example.com",
        role="ANALYST",
    )

    job_id = job_service.create_failed_report_job(
        user=user,
        request_id="REQ-123",
        days=30,
    )

    assert isinstance(job_id, UUID)
    assert persisted == {
        "job_id": job_id,
        "job_type": JobType.FAILED_TRANSACTION_REPORT,
        "requested_by": "USER-ANALYST",
        "request_id": "REQ-123",
        "parameters": {"days": 30},
    }


def test_job_lookup_sql_enforces_ownership():
    query = str(job_repository.GET_JOB_FOR_USER)
    assert "job_id = :job_id" in query
    assert "requested_by = :requested_by" in query


def test_job_schema_has_real_stage_and_creation_is_audited():
    migration = (
        PROJECT_ROOT / "data/migrations/007_add_background_job_stage.sql"
    ).read_text()
    audit_sql = str(job_repository.AUDIT_JOB_CREATION)

    assert "ADD COLUMN IF NOT EXISTS stage VARCHAR(100)" in migration
    assert "WHEN 'completed' THEN 'completed'" in migration
    assert "ALTER COLUMN stage SET NOT NULL" in migration
    assert "create_background_job" in audit_sql
    assert "CAST(:metadata AS JSONB)" in audit_sql


def test_duplicate_delivery_does_not_execute_twice(monkeypatch):
    job_id = uuid4()
    executed = []
    monkeypatch.setattr(
        job_repository,
        "get_job",
        lambda value: {
            "job_id": job_id,
            "job_type": JobType.FAILED_TRANSACTION_REPORT,
            "status": JobStatus.QUEUED,
            "parameters": {"days": 30},
        },
    )
    claims = iter([True, False])
    monkeypatch.setattr(job_repository, "mark_running", lambda value: next(claims))
    monkeypatch.setattr(
        executor,
        "execute_by_type",
        lambda value: executed.append(value) or {"failed_count": 0},
    )
    monkeypatch.setattr(
        job_repository,
        "mark_completed",
        lambda value, *, result: None,
    )

    executor.execute_job(job_id)
    executor.execute_job(job_id)

    assert len(executed) == 1


def test_failure_metrics_are_deterministic():
    metrics = executor.calculate_failure_metrics([
        {"product_id": "PROD-B", "amount_cents": 500},
        {"product_id": "PROD-A", "amount_cents": 300},
        {"product_id": "PROD-A", "amount_cents": 700},
    ])

    assert metrics == {
        "failed_count": 3,
        "total_failed_amount_cents": 1500,
        "top_failure_products": [
            {
                "product_id": "PROD-A",
                "failed_count": 2,
                "total_failed_amount_cents": 1000,
            },
            {
                "product_id": "PROD-B",
                "failed_count": 1,
                "total_failed_amount_cents": 500,
            },
        ],
    }


def test_failed_report_queries_transactions_then_calculates_metrics(monkeypatch):
    job_id = uuid4()
    stages = []
    monkeypatch.setattr(
        transaction_repository,
        "find_failed_transactions",
        lambda *, days, timeout_ms: [
            {"product_id": "PROD-A", "amount_cents": 250}
        ],
    )
    monkeypatch.setattr(
        job_repository,
        "update_stage",
        lambda value, stage: stages.append((value, stage)),
    )

    result = executor.execute_failed_report(
        {"job_id": job_id, "parameters": {"days": 14}},
        deadline=float("inf"),
    )

    assert result == {
        "days": 14,
        "failed_count": 1,
        "total_failed_amount_cents": 250,
        "top_failure_products": [
            {
                "product_id": "PROD-A",
                "failed_count": 1,
                "total_failed_amount_cents": 250,
            }
        ],
    }
    assert stages == [
        (job_id, executor.JobStage.QUERYING_TRANSACTIONS),
        (job_id, executor.JobStage.CALCULATING_METRICS),
    ]


def test_failed_report_enforces_runtime_deadline():
    with pytest.raises(TimeoutError, match="runtime limit"):
        executor.execute_failed_report(
            {"job_id": uuid4(), "parameters": {"days": 14}},
            deadline=0,
        )


def test_execute_job_completes_claimed_job(monkeypatch):
    job_id = uuid4()
    completed = {}
    job = {
        "job_id": job_id,
        "job_type": JobType.FAILED_TRANSACTION_REPORT,
        "status": JobStatus.QUEUED,
        "parameters": {"days": 7},
        "created_at": datetime.now(timezone.utc),
    }
    monkeypatch.setattr(job_repository, "get_job", lambda value: job)
    monkeypatch.setattr(job_repository, "mark_running", lambda value: True)
    monkeypatch.setattr(
        executor,
        "execute_by_type",
        lambda value: {"days": 7, "failed_count": 0},
    )
    monkeypatch.setattr(
        job_repository,
        "mark_completed",
        lambda value, *, result: completed.update(
            {"job_id": value, "result": result}
        ),
    )

    executor.execute_job(job_id)

    assert completed == {
        "job_id": job_id,
        "result": {"days": 7, "failed_count": 0},
    }


def test_execute_job_stores_only_safe_failure_code(monkeypatch):
    job_id = uuid4()
    failure = {}
    job = {
        "job_type": JobType.FAILED_TRANSACTION_REPORT,
        "status": JobStatus.QUEUED,
        "parameters": {"days": 30},
    }
    monkeypatch.setattr(job_repository, "get_job", lambda value: job)
    monkeypatch.setattr(job_repository, "mark_running", lambda value: True)
    monkeypatch.setattr(
        executor,
        "execute_by_type",
        lambda value: (_ for _ in ()).throw(RuntimeError("secret DB detail")),
    )
    monkeypatch.setattr(
        job_repository,
        "mark_failed",
        lambda value, *, error_code: failure.update(
            {"job_id": value, "error_code": error_code}
        ),
    )

    executor.execute_job(job_id)

    assert failure == {
        "job_id": job_id,
        "error_code": "report_generation_failed",
    }
