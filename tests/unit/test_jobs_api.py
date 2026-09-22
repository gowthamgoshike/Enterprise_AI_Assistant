from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from apps.api.dependencies.auth import get_current_user
from apps.api.main import app
from apps.api.routes import jobs as jobs_routes
from apps.api.schemas.auth import AuthenticatedUser
from services.job_service.models import JobResponse, JobStage, JobStatus


def _user(role: str) -> AuthenticatedUser:
    return AuthenticatedUser(
        user_id=f"USER-{role}",
        email=f"{role.lower()}@example.com",
        role=role,
    )


@pytest.mark.parametrize("role", ["ANALYST", "ADMIN"])
def test_authorized_role_can_queue_failed_transaction_report(monkeypatch, role):
    job_id = uuid4()
    app.dependency_overrides[get_current_user] = lambda: _user(role)
    monkeypatch.setattr(
        jobs_routes,
        "create_failed_report_job",
        lambda **values: job_id,
    )
    monkeypatch.setattr(jobs_routes, "execute_job", lambda value: None)
    try:
        response = TestClient(app).post(
            "/api/v1/jobs/failed-transactions",
            json={"days": 30},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 202
    assert response.json() == {"job_id": str(job_id), "status": "queued"}


def test_customer_cannot_create_report_job(monkeypatch):
    created = []
    app.dependency_overrides[get_current_user] = lambda: _user("CUSTOMER")
    monkeypatch.setattr(
        jobs_routes,
        "create_failed_report_job",
        lambda **values: created.append(values),
    )
    try:
        response = TestClient(app).post(
            "/api/v1/jobs/failed-transactions",
            json={"days": 30},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 403
    assert created == []


def test_report_days_are_validated_before_job_creation(monkeypatch):
    created = []
    app.dependency_overrides[get_current_user] = lambda: _user("ADMIN")
    monkeypatch.setattr(
        jobs_routes,
        "create_failed_report_job",
        lambda **values: created.append(values),
    )
    try:
        response = TestClient(app).post(
            "/api/v1/jobs/failed-transactions",
            json={"days": 91},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert created == []


def _job_response(job_id, status=JobStatus.COMPLETED):
    stage = {
        JobStatus.QUEUED: JobStage.QUEUED,
        JobStatus.RUNNING: JobStage.QUERYING_TRANSACTIONS,
        JobStatus.COMPLETED: JobStage.COMPLETED,
        JobStatus.FAILED: JobStage.FAILED,
        JobStatus.CANCELLED: JobStage.CANCELLED,
    }[status]
    return JobResponse(
        job_id=job_id,
        job_type="failed_transaction_report",
        status=status,
        stage=stage,
        created_at=datetime.now(timezone.utc),
        result={"days": 30, "failed_count": 2}
        if status is JobStatus.COMPLETED
        else None,
    )


def test_owner_can_poll_job_status(monkeypatch):
    job_id = uuid4()
    app.dependency_overrides[get_current_user] = lambda: _user("ANALYST")
    monkeypatch.setattr(
        jobs_routes,
        "get_job_for_user",
        lambda value, user: _job_response(value),
    )
    try:
        response = TestClient(app).get(f"/api/v1/jobs/{job_id}")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["job_id"] == str(job_id)
    assert response.json()["stage"] == "completed"
    assert response.json()["result"]["failed_count"] == 2


def test_non_owner_job_lookup_returns_404(monkeypatch):
    job_id = uuid4()
    app.dependency_overrides[get_current_user] = lambda: _user("ANALYST")
    monkeypatch.setattr(
        jobs_routes,
        "get_job_for_user",
        lambda value, user: None,
    )
    try:
        response = TestClient(app).get(f"/api/v1/jobs/{job_id}")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404


def test_owner_can_cancel_only_a_queued_job(monkeypatch):
    job_id = uuid4()
    app.dependency_overrides[get_current_user] = lambda: _user("ADMIN")
    monkeypatch.setattr(
        jobs_routes,
        "cancel_job_for_user",
        lambda value, user: _job_response(value, JobStatus.CANCELLED),
    )
    try:
        response = TestClient(app).post(f"/api/v1/jobs/{job_id}/cancel")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_running_job_cancellation_returns_conflict(monkeypatch):
    job_id = uuid4()
    app.dependency_overrides[get_current_user] = lambda: _user("ADMIN")
    monkeypatch.setattr(
        jobs_routes,
        "cancel_job_for_user",
        lambda value, user: _job_response(value, JobStatus.RUNNING),
    )
    try:
        response = TestClient(app).post(f"/api/v1/jobs/{job_id}/cancel")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 409
