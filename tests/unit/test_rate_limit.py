"""Shared chat limiting and its authenticated API boundary."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from apps.api.dependencies.auth import get_current_user
from apps.api.dependencies import rate_limit as rate_limit_dependencies
from apps.api.main import app
from apps.api.routes import chat as chat_routes
from apps.api.schemas.auth import AuthenticatedUser
from core import rate_limit
from core.errors import RateLimitError
from services.auth_service.models import UserRole


class FakeRedis:
    def __init__(self):
        self.counts = {}
        self.expires_at = {}
        self.now = 0
        self.fail = False

    async def eval(self, script, numkeys, key, window):
        if self.fail:
            raise ConnectionError("Redis offline")
        assert numkeys == 1
        assert "INCR" in script and "EXPIRE" in script
        if self.expires_at.get(key, 0) <= self.now:
            self.counts.pop(key, None)
        count = self.counts.get(key, 0) + 1
        self.counts[key] = count
        if count == 1:
            self.expires_at[key] = self.now + window
        return count


@pytest.mark.asyncio
async def test_fixed_window_is_per_user_and_expires(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(rate_limit, "redis_client", redis)
    monkeypatch.setattr(rate_limit, "settings", SimpleNamespace(
        rate_limit_requests=2, rate_limit_window_seconds=60,
    ))

    assert await rate_limit.allow_request("USER-1")
    assert await rate_limit.allow_request("USER-1")
    assert not await rate_limit.allow_request("USER-1")
    assert await rate_limit.allow_request("USER-2")
    assert redis.counts["v1:rate:chat:USER-1"] == 3
    redis.now = 60
    assert await rate_limit.allow_request("USER-1")


@pytest.mark.asyncio
async def test_redis_failure_allows_request_and_logs(monkeypatch, caplog):
    redis = FakeRedis()
    redis.fail = True
    monkeypatch.setattr(rate_limit, "redis_client", redis)

    with caplog.at_level("WARNING"):
        assert await rate_limit.allow_request("USER-1")
    assert any(record.msg == "redis_rate_limit_unavailable"
               for record in caplog.records)


def test_chat_and_stream_share_authenticated_limit(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(rate_limit, "redis_client", redis)
    monkeypatch.setattr(rate_limit, "settings", SimpleNamespace(
        rate_limit_requests=1, rate_limit_window_seconds=60,
    ))
    user = AuthenticatedUser(
        user_id="TRUSTED-USER", email="test@example.com",
        role=UserRole.ANALYST, customer_id=None,
    )
    app.dependency_overrides[get_current_user] = lambda: user
    graph = AsyncMock(return_value={
        "messages": [AIMessage(content="Hello")],
        "current_turn_tools": [],
    })
    monkeypatch.setattr(chat_routes, "run_agent_graph", graph)
    client = TestClient(app)
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "session_id": "limit-test",
    }
    try:
        assert client.post("/api/v1/chat", json=payload).status_code == 200
        assert client.get("/api/v1/health/live").status_code == 200
        denied = client.post("/api/v1/chat/stream", json=payload)
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert denied.status_code == 429
    assert denied.json()["error"] == {
        "code": "rate_limit_exceeded",
        "message": "Too many requests. Please try again shortly.",
        "retryable": True,
    }
    assert denied.json()["request_id"] == denied.headers["X-Request-ID"]
    assert denied.headers["Retry-After"] == "60"
    assert redis.counts == {"v1:rate:chat:TRUSTED-USER": 2}
    assert graph.await_count == 1


def test_unauthenticated_request_does_not_touch_redis(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(rate_limit, "redis_client", redis)
    response = TestClient(app).post("/api/v1/chat", json={
        "messages": [{"role": "user", "content": "Hello"}],
        "session_id": "unauthenticated",
    })
    assert response.status_code in (401, 403)
    assert redis.counts == {}


@pytest.mark.asyncio
async def test_endpoint_specific_limits_use_separate_scopes(monkeypatch):
    calls = []

    async def allowed(subject, **options):
        calls.append((subject, options))
        return True

    monkeypatch.setattr(rate_limit_dependencies, "allow_request", allowed)
    monkeypatch.setattr(
        rate_limit_dependencies,
        "settings",
        SimpleNamespace(
            rate_limit_window_seconds=60,
            approval_rate_limit_requests=10,
            job_rate_limit_requests=5,
            login_rate_limit_requests=5,
        ),
    )
    user = AuthenticatedUser(
        user_id="USER-LIMITS",
        email="limits@example.com",
        role=UserRole.ADMIN,
    )

    await rate_limit_dependencies.get_rate_limited_chat_user(user)
    await rate_limit_dependencies.get_rate_limited_approval_user(user)
    await rate_limit_dependencies.get_rate_limited_job_user(user)
    await rate_limit_dependencies.rate_limit_login(
        SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    )

    assert calls == [
        ("USER-LIMITS", {"scope": "chat", "limit": None}),
        ("USER-LIMITS", {"scope": "approval", "limit": 10}),
        ("USER-LIMITS", {"scope": "jobs", "limit": 5}),
        ("127.0.0.1", {"scope": "login", "limit": 5}),
    ]


@pytest.mark.asyncio
async def test_endpoint_limit_raises_stable_rate_limit_error(monkeypatch):
    monkeypatch.setattr(
        rate_limit_dependencies,
        "allow_request",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        rate_limit_dependencies,
        "settings",
        SimpleNamespace(
            rate_limit_window_seconds=30,
            job_rate_limit_requests=5,
        ),
    )
    user = AuthenticatedUser(
        user_id="USER-LIMITED",
        email="limited@example.com",
        role=UserRole.ADMIN,
    )

    with pytest.raises(RateLimitError) as exc_info:
        await rate_limit_dependencies.get_rate_limited_job_user(user)

    assert exc_info.value.retry_after_seconds == 30
