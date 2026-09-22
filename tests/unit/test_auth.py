from datetime import datetime, timedelta, timezone
import asyncio
import logging

import jwt
import pytest
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from apps.api.dependencies import auth as auth_dependency
from apps.api.dependencies.auth import get_current_user
from apps.api.main import app
from apps.api.routes import auth as auth_routes
from apps.api.schemas.auth import AuthenticatedUser
from core.errors import InvalidTokenError
from services.auth_service import auth_service, jwt_service
from services.auth_service.models import UserRole
from services.auth_service.password_service import (
    hash_password,
    verify_password,
)


client = TestClient(app)


def test_user_roles_are_restricted():
    assert UserRole.CUSTOMER.value == "CUSTOMER"

    with pytest.raises(ValueError):
        AuthenticatedUser(
            user_id="USER-1",
            email="user@example.com",
            role="INVENTED_ROLE",
        )


@pytest.fixture
def hashed_test_password():
    return hash_password("correct horse battery staple")


def test_password_hashing_does_not_store_plaintext(hashed_test_password):
    hashed_password = hashed_test_password
    assert hashed_password.startswith("$argon2")
    assert hashed_password != "correct horse battery staple"


def test_correct_password_is_accepted(hashed_test_password):
    hashed_password = hashed_test_password
    assert verify_password("correct horse battery staple", hashed_password)


def test_incorrect_password_is_rejected(hashed_test_password):
    hashed_password = hashed_test_password
    assert not verify_password("wrong password", hashed_password)


def test_jwt_creation_and_decoding_contains_only_safe_claims(monkeypatch):
    monkeypatch.setattr(jwt_service, "JWT_SECRET_KEY", "a-secure-test-key-with-at-least-32-bytes")
    monkeypatch.setattr(jwt_service, "JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(jwt_service, "ACCESS_TOKEN_MINUTES", 30)

    token = jwt_service.create_access_token("USER-ABC-123")
    payload = jwt_service.decode_access_token(token)

    assert payload["sub"] == "USER-ABC-123"
    assert set(payload) == {"sub", "iat", "exp"}
    assert payload["exp"] > int(datetime.now(timezone.utc).timestamp())


def test_expired_access_token_is_rejected(monkeypatch):
    monkeypatch.setattr(
        jwt_service,
        "JWT_SECRET_KEY",
        "a-secure-test-key-with-at-least-32-bytes",
    )
    monkeypatch.setattr(jwt_service, "JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(jwt_service, "ACCESS_TOKEN_MINUTES", -1)

    token = jwt_service.create_access_token("USER-EXPIRED")

    with pytest.raises(jwt.ExpiredSignatureError):
        jwt_service.decode_access_token(token)


def test_modified_access_token_is_rejected(monkeypatch):
    monkeypatch.setattr(
        jwt_service,
        "JWT_SECRET_KEY",
        "a-secure-test-key-with-at-least-32-bytes",
    )
    monkeypatch.setattr(jwt_service, "JWT_ALGORITHM", "HS256")
    monkeypatch.setattr(jwt_service, "ACCESS_TOKEN_MINUTES", 30)

    token = jwt_service.create_access_token("USER-ABC-123")
    header, payload, signature = token.split(".")
    replacement = "A" if signature[0] != "A" else "B"
    modified_token = ".".join((
        header,
        payload,
        replacement + signature[1:],
    ))

    with pytest.raises(jwt.InvalidSignatureError):
        jwt_service.decode_access_token(modified_token)


def test_authenticate_user_requires_active_user_and_valid_password(monkeypatch):
    hashed_password = hash_password("correct-password")
    user = {
        "user_id": "USER-ABC-123",
        "email": "customer@example.com",
        "password_hash": hashed_password,
        "role": "CUSTOMER",
        "customer_id": "CUST-001",
        "is_active": True,
    }
    monkeypatch.setattr(auth_service, "find_user_by_email", lambda email: user)

    assert auth_service.authenticate_user(
        "customer@example.com",
        "correct-password",
    ) == user
    assert auth_service.authenticate_user(
        "customer@example.com",
        "wrong-password",
    ) is None

    user["is_active"] = False
    assert auth_service.authenticate_user(
        "customer@example.com",
        "correct-password",
    ) is None


def test_login_returns_token_without_exposing_user_details(monkeypatch):
    monkeypatch.setattr(
        auth_routes,
        "authenticate_user",
        lambda **credentials: {"user_id": "USER-ABC-123"},
    )
    monkeypatch.setattr(
        auth_routes,
        "create_access_token",
        lambda user_id: f"token-for-{user_id}",
    )

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "customer@example.com",
            "password": "correct-password",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "access_token": "token-for-USER-ABC-123",
        "token_type": "bearer",
    }


def test_login_uses_generic_error_for_invalid_credentials(monkeypatch):
    monkeypatch.setattr(
        auth_routes,
        "authenticate_user",
        lambda **credentials: None,
    )

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "unknown@example.com",
            "password": "wrong-password",
        },
    )

    assert response.status_code == 401
    assert response.json()["error"] == {
        "code": "invalid_credentials",
        "message": "The email or password is incorrect.",
        "retryable": False,
    }


@pytest.mark.parametrize("login_succeeds", [True, False])
def test_login_logs_only_operational_metadata(
    monkeypatch,
    caplog,
    login_succeeds,
):
    email = "private-person@example.com"
    password = "do-not-log-this-password"
    password_hash = "$argon2id$do-not-log-this-hash"
    jwt_secret = "do-not-log-this-jwt-secret"
    access_token = "do-not-log-this-full-access-token"
    user = {
        "user_id": "USER-SAFE-LOG",
        "email": email,
        "password_hash": password_hash,
    }
    monkeypatch.setattr(
        auth_routes,
        "authenticate_user",
        lambda **credentials: user if login_succeeds else None,
    )
    monkeypatch.setattr(
        auth_routes,
        "create_access_token",
        lambda user_id: access_token,
    )

    with caplog.at_level(logging.INFO, logger=auth_routes.__name__):
        response = client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": password},
        )

    assert response.status_code == (200 if login_succeeds else 401)
    assert "auth_login" in caplog.text
    assert "endpoint=/api/v1/auth/login" in caplog.text
    assert f"result={'success' if login_succeeds else 'failure'}" in caplog.text
    assert email not in caplog.text
    assert password not in caplog.text
    assert password_hash not in caplog.text
    assert jwt_secret not in caplog.text
    assert access_token not in caplog.text


def test_me_returns_authenticated_user_without_password_hash():
    user = AuthenticatedUser(
        user_id="USER-ABC-123",
        email="customer@example.com",
        role=UserRole.CUSTOMER,
        customer_id="CUST-001",
    )
    app.dependency_overrides[get_current_user] = lambda: user

    try:
        response = client.get(
            "/api/v1/auth/me"
        )
    finally:
        app.dependency_overrides.pop(
            get_current_user,
            None,
        )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "USER-ABC-123",
        "email": "customer@example.com",
        "role": "CUSTOMER",
        "customer_id": "CUST-001",
    }
    assert "password" not in response.text


def test_me_requires_bearer_token():
    app.dependency_overrides.pop(
        get_current_user,
        None,
    )

    response = client.get(
        "/api/v1/auth/me"
    )

    assert response.status_code == 401


def test_current_user_is_loaded_from_verified_token_subject(monkeypatch):
    monkeypatch.setattr(
        auth_dependency,
        "decode_access_token",
        lambda token: {"sub": "USER-ABC-123"},
    )
    monkeypatch.setattr(
        auth_dependency,
        "find_user_by_id",
        lambda user_id: {
            "user_id": user_id,
            "email": "customer@example.com",
            "role": "CUSTOMER",
            "customer_id": "CUST-001",
            "is_active": True,
        },
    )

    user = asyncio.run(get_current_user(
        HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="verified-token",
        )
    ))

    assert user == AuthenticatedUser(
        user_id="USER-ABC-123",
        email="customer@example.com",
        role=UserRole.CUSTOMER,
        customer_id="CUST-001",
    )


@pytest.mark.parametrize("payload", [{}, {"other": "claim"}])
def test_current_user_rejects_invalid_or_subjectless_token(
    monkeypatch,
    payload,
):
    if payload:
        monkeypatch.setattr(
            auth_dependency,
            "decode_access_token",
            lambda token: payload,
        )
    else:
        def reject_token(token):
            raise jwt.InvalidTokenError("invalid token details")

        monkeypatch.setattr(
            auth_dependency,
            "decode_access_token",
            reject_token,
        )

    with pytest.raises(InvalidTokenError) as exc_info:
        asyncio.run(get_current_user(
            HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials="invalid-token",
            )
        ))

    assert exc_info.value.code == "invalid_token"


def test_current_user_rejects_inactive_database_user(monkeypatch):
    monkeypatch.setattr(
        auth_dependency,
        "decode_access_token",
        lambda token: {"sub": "USER-INACTIVE"},
    )
    monkeypatch.setattr(
        auth_dependency,
        "find_user_by_id",
        lambda user_id: {
            "user_id": user_id,
            "email": "inactive@example.com",
            "role": "CUSTOMER",
            "customer_id": "CUST-002",
            "is_active": False,
        },
    )

    with pytest.raises(InvalidTokenError) as exc_info:
        asyncio.run(get_current_user(
            HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials="valid-token",
            )
        ))

    assert exc_info.value.code == "invalid_token"


def test_chat_requires_bearer_token():
    app.dependency_overrides.pop(get_current_user, None)
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "session_id": "auth-required",
    }

    response = client.post(
        "/api/v1/chat",
        json=payload,
    )
    stream_response = client.post(
        "/api/v1/chat/stream",
        json=payload,
    )

    assert response.status_code == 401
    assert stream_response.status_code == 401


def test_chat_rejects_invalid_bearer_token_before_graph_or_tools(monkeypatch):
    from unittest.mock import AsyncMock, Mock

    from ai.tools import transaction_tools
    from apps.api.routes import chat as chat_routes
    from services.transaction_service import repository as transaction_repository

    app.dependency_overrides.pop(get_current_user, None)
    user_lookup = Mock()
    graph_run = AsyncMock()
    transaction_tool_lookup = Mock()
    transaction_database_connect = Mock()
    monkeypatch.setattr(
        auth_dependency,
        "find_user_by_id",
        user_lookup,
    )
    monkeypatch.setattr(
        chat_routes,
        "run_agent_graph",
        graph_run,
    )
    monkeypatch.setattr(
        transaction_tools,
        "find_transaction",
        transaction_tool_lookup,
    )
    monkeypatch.setattr(
        transaction_repository.engine,
        "connect",
        transaction_database_connect,
    )

    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": "Bearer abc123"},
        json={
            "messages": [{
                "role": "user",
                "content": "What is the status of TX-123?",
            }],
            "session_id": "invalid-token",
        },
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_token"
    user_lookup.assert_not_called()
    graph_run.assert_not_awaited()
    transaction_tool_lookup.assert_not_called()
    transaction_database_connect.assert_not_called()


def test_chat_rejects_modified_signed_token_before_user_or_graph(monkeypatch):
    from unittest.mock import AsyncMock, Mock

    from apps.api.routes import chat as chat_routes

    app.dependency_overrides.pop(get_current_user, None)
    monkeypatch.setattr(
        jwt_service,
        "JWT_SECRET_KEY",
        "a-secure-test-key-with-at-least-32-bytes",
    )
    valid_token = jwt_service.create_access_token(
        "USER-ABC-123"
    )
    header, payload, signature = valid_token.split(".")
    replacement = "A" if signature[0] != "A" else "B"
    modified_token = ".".join((
        header,
        payload,
        replacement + signature[1:],
    ))
    user_lookup = Mock()
    graph_run = AsyncMock()
    monkeypatch.setattr(
        auth_dependency,
        "find_user_by_id",
        user_lookup,
    )
    monkeypatch.setattr(
        chat_routes,
        "run_agent_graph",
        graph_run,
    )

    response = client.post(
        "/api/v1/chat",
        headers={
            "Authorization": f"Bearer {modified_token}",
        },
        json={
            "messages": [{
                "role": "user",
                "content": "What is the status of TX-123?",
            }],
            "session_id": "modified-token",
        },
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_token"
    user_lookup.assert_not_called()
    graph_run.assert_not_awaited()
