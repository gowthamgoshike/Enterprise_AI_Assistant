from unittest.mock import Mock

from fastapi.testclient import TestClient

from apps.api.dependencies.auth import get_current_user
from apps.api.main import app
from apps.api.middleware.request_id import _metric_path
from apps.api.schemas.auth import AuthenticatedUser
from services.auth_service.models import UserRole


client = TestClient(app)


def _assert_error_contract(response, code: str) -> None:
    payload = response.json()
    assert payload["request_id"] == response.headers["X-Request-ID"]
    assert payload["error"]["code"] == code
    assert isinstance(payload["error"]["message"], str)
    assert isinstance(payload["error"]["retryable"], bool)


def test_client_request_id_is_validated_and_returned():
    accepted = client.get(
        "/api/v1/health/live",
        headers={"X-Request-ID": "client.request-123"},
    )
    rejected = client.get(
        "/api/v1/health/live",
        headers={"X-Request-ID": "invalid request id with spaces"},
    )

    assert accepted.headers["X-Request-ID"] == "client.request-123"
    assert rejected.headers["X-Request-ID"].startswith("REQ-")
    assert rejected.headers["X-Request-ID"] != "invalid request id with spaces"


def test_job_metric_paths_do_not_include_user_controlled_ids():
    request = Mock()
    request.url.path = "/api/v1/jobs/not-a-valid-uuid/cancel"
    assert _metric_path(request) == "/api/v1/jobs/{job_id}/cancel"

    request.url.path = "/api/v1/jobs/failed-transactions"
    assert _metric_path(request) == "/api/v1/jobs/failed-transactions"


def test_security_headers_are_on_normal_and_rejected_responses():
    normal = client.get("/api/v1/health/live")
    oversized = client.post(
        "/api/v1/auth/login",
        content=b"{}",
        headers={"Content-Length": "65537"},
    )

    assert oversized.status_code == 413
    _assert_error_contract(oversized, "request_too_large")
    for response in (normal, oversized):
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Referrer-Policy"] == "no-referrer"
        assert "Strict-Transport-Security" not in response.headers


def test_oversized_request_is_rejected_before_authentication(monkeypatch):
    authenticate = Mock()
    monkeypatch.setattr(
        "apps.api.routes.auth.authenticate_user",
        authenticate,
    )

    response = client.post(
        "/api/v1/auth/login",
        content=b"{}",
        headers={"Content-Length": "70000"},
    )

    assert response.status_code == 413
    authenticate.assert_not_called()


def test_chunked_body_without_content_length_is_also_bounded():
    response = client.post(
        "/api/v1/auth/login",
        content=(b"x" * 8192 for _ in range(9)),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413
    _assert_error_contract(response, "request_too_large")


def test_strict_schema_and_field_limits_reject_before_graph():
    user = AuthenticatedUser(
        user_id="USER-BOUNDARY",
        email="boundary@example.com",
        role=UserRole.ANALYST,
    )
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        extra_field = client.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "session_id": "SESSION-1",
                "role": "ADMIN",
            },
        )
        long_message = client.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "x" * 8001}],
                "session_id": "SESSION-1",
            },
        )
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert extra_field.status_code == 422
    assert long_message.status_code == 422
    _assert_error_contract(extra_field, "validation_error")
    _assert_error_contract(long_message, "validation_error")
    assert "ADMIN" not in extra_field.text


def test_authentication_and_not_found_use_standard_errors():
    unauthenticated = client.get("/api/v1/auth/me")
    missing = client.get("/api/v1/does-not-exist")

    assert unauthenticated.status_code == 401
    _assert_error_contract(unauthenticated, "authentication_required")
    assert unauthenticated.headers["WWW-Authenticate"] == "Bearer"
    assert missing.status_code == 404
    _assert_error_contract(missing, "resource_not_found")


def test_cors_is_narrow_and_versioned_routes_are_centralized():
    allowed = client.options(
        "/api/v1/chat",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization,Content-Type",
        },
    )
    denied = client.options(
        "/api/v1/chat",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert allowed.status_code == 200
    assert allowed.headers["Access-Control-Allow-Origin"] == (
        "http://localhost:8501"
    )
    assert "Access-Control-Allow-Origin" not in denied.headers

    paths = app.openapi()["paths"]
    assert "/api/v1/chat" in paths
    assert "/api/v1/chat/stream" in paths
    assert "/api/v1/jobs/failed-transactions" in paths
    assert "/api/v1/health/live" in paths
