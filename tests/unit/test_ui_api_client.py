import pytest

from apps.ui.services import api_client, chat_client


class FakeResponse:
    def __init__(
        self,
        *,
        status_code=200,
        payload=None,
        lines=None,
    ):
        self.status_code = status_code
        self.payload = payload or {}
        self.lines = lines or []
        self.raise_called = False

    def raise_for_status(self):
        self.raise_called = True

    def json(self):
        return self.payload

    def iter_lines(self):
        return iter(self.lines)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_login_posts_credentials_without_bearer_header(monkeypatch):
    calls = []
    response = FakeResponse(payload={
        "access_token": "jwt-token",
        "token_type": "bearer",
    })

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    result = api_client.login(
        "customer@example.com",
        "secret-password",
    )

    assert calls == [(
        f"{api_client.API_BASE_URL}/api/v1/auth/login",
        {
            "json": {
                "email": "customer@example.com",
                "password": "secret-password",
            },
            "timeout": 30,
        },
    )]
    assert response.raise_called
    assert result["access_token"] == "jwt-token"


def test_send_chat_uses_bearer_token(monkeypatch):
    calls = []
    response = FakeResponse(payload={"response": "Hello"})

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    result = api_client.send_chat(
        messages=[{"role": "user", "content": "Hello"}],
        session_id="SESSION-1",
        access_token="jwt-token",
    )

    assert calls[0][1]["headers"] == {
        "Authorization": "Bearer jwt-token",
    }
    assert calls[0][1]["timeout"] == 120
    assert response.raise_called
    assert result == {"response": "Hello"}


def test_send_chat_converts_401_to_session_expired(monkeypatch):
    response = FakeResponse(status_code=401)
    monkeypatch.setattr(
        api_client.requests,
        "post",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(
        api_client.SessionExpiredError,
        match="Session expired",
    ):
        api_client.send_chat(
            messages=[{"role": "user", "content": "Hello"}],
            session_id="SESSION-1",
            access_token="expired-token",
        )

    assert not response.raise_called


def test_submit_approval_uses_bearer_token(monkeypatch):
    calls = []
    response = FakeResponse(payload={"status": "completed"})

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(api_client.requests, "post", fake_post)
    result = api_client.submit_approval(
        session_id="SESSION-1",
        approved=True,
        comment="Reviewed",
        access_token="jwt-token",
    )

    assert calls == [(
        f"{api_client.API_BASE_URL}/api/v1/chat/approval",
        {
            "json": {
                "session_id": "SESSION-1",
                "approved": True,
                "comment": "Reviewed",
            },
            "headers": {"Authorization": "Bearer jwt-token"},
            "timeout": 120,
        },
    )]
    assert response.raise_called
    assert result == {"status": "completed"}


def test_report_job_client_supports_submit_poll_and_cancel(monkeypatch):
    calls = []
    responses = iter([
        FakeResponse(payload={"job_id": "JOB-1", "status": "queued"}),
        FakeResponse(payload={"job_id": "JOB-1", "status": "running"}),
        FakeResponse(payload={"job_id": "JOB-1", "status": "cancelled"}),
    ])

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        return next(responses)

    monkeypatch.setattr(api_client.requests, "post", fake_request)
    monkeypatch.setattr(api_client.requests, "get", fake_request)

    api_client.create_failed_transaction_report(
        days=30,
        access_token="jwt-token",
    )
    api_client.get_background_job(job_id="JOB-1", access_token="jwt-token")
    api_client.cancel_background_job(job_id="JOB-1", access_token="jwt-token")

    assert calls == [
        (
            f"{api_client.API_BASE_URL}/api/v1/jobs/failed-transactions",
            {
                "json": {"days": 30},
                "headers": {"Authorization": "Bearer jwt-token"},
                "timeout": 30,
            },
        ),
        (
            f"{api_client.API_BASE_URL}/api/v1/jobs/JOB-1",
            {
                "headers": {"Authorization": "Bearer jwt-token"},
                "timeout": 10,
            },
        ),
        (
            f"{api_client.API_BASE_URL}/api/v1/jobs/JOB-1/cancel",
            {
                "headers": {"Authorization": "Bearer jwt-token"},
                "timeout": 10,
            },
        ),
    ]


def test_stream_chat_uses_bearer_token_and_detects_401(monkeypatch):
    calls = []
    response = FakeResponse(
        lines=[
            "event: token",
            'data: {"type":"token","data":{"text":"Hello"}}',
            "",
            "event: done",
            'data: {"type":"done","data":{"status":"completed"}}',
        ],
    )

    def fake_stream(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return response

    monkeypatch.setattr(chat_client.httpx, "stream", fake_stream)

    assert list(chat_client.stream_chat(
        message="Hello",
        session_id="SESSION-1",
        access_token="jwt-token",
    )) == [
        {"type": "token", "data": {"text": "Hello"}},
        {"type": "done", "data": {"status": "completed"}},
    ]
    assert calls[0][0:2] == (
        "POST", f"{chat_client.API_BASE_URL}/api/v1/chat/stream"
    )
    assert calls[0][2]["headers"] == {
        "Authorization": "Bearer jwt-token",
        "Accept": "text/event-stream",
    }
    assert calls[0][2]["json"] == {
        "messages": [{"role": "user", "content": "Hello"}],
        "session_id": "SESSION-1",
    }

    monkeypatch.setattr(
        chat_client.httpx,
        "stream",
        lambda *args, **kwargs: FakeResponse(status_code=401),
    )

    with pytest.raises(api_client.SessionExpiredError):
        list(chat_client.stream_chat(
            message="Hello",
            session_id="SESSION-1",
            access_token="expired-token",
        ))
