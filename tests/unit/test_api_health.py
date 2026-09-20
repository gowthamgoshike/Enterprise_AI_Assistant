from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from apps.api.main import app
from apps.api.routes import health as health_routes
from apps.api.schemas.health import DependencyStatus


client = TestClient(app)


def test_health():

    response = client.get(
        "/api/v1/health"
    )

    assert response.status_code == 200

    assert (
        response.json()["status"]
        ==
        "ok"
    )


def test_liveness_does_not_depend_on_downstreams():
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_reports_sanitized_degradation(monkeypatch):
    monkeypatch.setattr(health_routes, "_database_status", AsyncMock(
        return_value=DependencyStatus(status="unavailable"),
    ))
    monkeypatch.setattr(health_routes, "_checkpoint_status", AsyncMock(
        return_value=DependencyStatus(status="healthy", latency_ms=1.0),
    ))
    monkeypatch.setattr(health_routes, "_model_status", AsyncMock(
        return_value=DependencyStatus(status="healthy", latency_ms=2.0),
    ))

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["database"] == {
        "status": "unavailable",
        "latency_ms": None,
    }
    assert "error" not in response.text

def test_checkpoint_health_is_safe_on_failure(monkeypatch):
    saver = AsyncMock()
    monkeypatch.setattr(app.state, "checkpoint_store", saver, raising=False)
    assert client.get("/api/v1/health/checkpoint").json() == {
        "api": "healthy", "checkpoint_store": "healthy",
    }
    saver.aget_tuple.side_effect = RuntimeError("private database connection string")
    response = client.get("/api/v1/health/checkpoint")
    assert response.status_code == 503
    assert response.json()["checkpoint_store"] == "unavailable"
    assert "private" not in response.text
