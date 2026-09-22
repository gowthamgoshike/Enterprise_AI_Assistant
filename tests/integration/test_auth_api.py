from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from apps.api.dependencies import auth as auth_dependency
from apps.api.main import app
from apps.api.routes import chat as chat_routes
from services.auth_service import auth_service, jwt_service
from services.auth_service.password_service import hash_password


TEST_JWT_SECRET = "integration-test-signing-key-with-at-least-32-bytes"
TEST_PASSWORD = "correct-integration-password"


@pytest.fixture
def auth_users(monkeypatch):
    users = {
        "active@example.com": {
            "user_id": "USER-ACTIVE",
            "email": "active@example.com",
            "password_hash": hash_password(TEST_PASSWORD),
            "role": "CUSTOMER",
            "customer_id": "CUST-001",
            "is_active": True,
        },
        "inactive@example.com": {
            "user_id": "USER-INACTIVE",
            "email": "inactive@example.com",
            "password_hash": hash_password(TEST_PASSWORD),
            "role": "CUSTOMER",
            "customer_id": "CUST-002",
            "is_active": False,
        },
    }

    monkeypatch.setattr(
        auth_service,
        "find_user_by_email",
        lambda email: users.get(email.lower()),
    )
    monkeypatch.setattr(
        auth_dependency,
        "find_user_by_id",
        lambda user_id: next(
            (user for user in users.values() if user["user_id"] == user_id),
            None,
        ),
    )
    monkeypatch.setattr(jwt_service, "JWT_SECRET_KEY", TEST_JWT_SECRET)
    monkeypatch.setattr(jwt_service, "JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(jwt_service, "ACCESS_TOKEN_MINUTES", 30)
    return users


@pytest.fixture
def client():
    return TestClient(app)


def login(client, email="active@example.com", password=TEST_PASSWORD):
    return client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_login_success_returns_a_decodable_access_token(client, auth_users):
    response = login(client)

    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"
    payload = jwt_service.decode_access_token(response.json()["access_token"])
    assert payload["sub"] == auth_users["active@example.com"]["user_id"]


def test_bad_login_returns_401(client, auth_users):
    response = login(client, password="incorrect-password")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_credentials"


def test_chat_without_jwt_returns_401(client, auth_users):
    response = client.post(
        "/api/v1/chat",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "session_id": "missing-jwt",
        },
    )

    assert response.status_code == 401


def test_chat_with_valid_jwt_is_accepted(client, auth_users, monkeypatch):
    graph_run = AsyncMock(return_value={
        "messages": [AIMessage(content="Authenticated response")],
        "current_turn_tools": [],
        "route": None,
        "route_confidence": None,
    })
    monkeypatch.setattr(chat_routes, "run_agent_graph", graph_run)
    token = login(client).json()["access_token"]

    response = client.post(
        "/api/v1/chat",
        headers=bearer(token),
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "session_id": "valid-jwt",
        },
    )

    assert response.status_code == 200
    assert response.json()["response"] == "Authenticated response"
    assert graph_run.await_args.kwargs["user_id"] == "USER-ACTIVE"


def test_me_with_valid_jwt_returns_the_user(client, auth_users):
    token = login(client).json()["access_token"]

    response = client.get(
        "/api/v1/auth/me",
        headers=bearer(token),
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "USER-ACTIVE",
        "email": "active@example.com",
        "role": "CUSTOMER",
        "customer_id": "CUST-001",
    }
    assert "password" not in response.text


def test_inactive_user_returns_401(client, auth_users):
    login_response = login(client, email="inactive@example.com")
    inactive_token = jwt_service.create_access_token("USER-INACTIVE")
    me_response = client.get(
        "/api/v1/auth/me",
        headers=bearer(inactive_token),
    )

    assert login_response.status_code == 401
    assert me_response.status_code == 401
    assert me_response.json()["error"]["code"] == "invalid_token"
