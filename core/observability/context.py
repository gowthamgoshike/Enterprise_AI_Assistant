"""Async-safe request context propagated through application calls."""

from contextvars import ContextVar, Token


request_id_var: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)
user_id_var: ContextVar[str | None] = ContextVar(
    "user_id",
    default=None,
)


def set_request_context(
    request_id: str,
    user_id: str | None = None,
) -> tuple[Token, Token]:
    """Bind request identity and return tokens for guaranteed cleanup."""
    return (
        request_id_var.set(request_id),
        user_id_var.set(user_id),
    )


def set_user_context(user_id: str | None) -> Token:
    return user_id_var.set(user_id)


def reset_request_context(tokens: tuple[Token, Token]) -> None:
    request_token, user_token = tokens
    user_id_var.reset(user_token)
    request_id_var.reset(request_token)


def get_request_id() -> str | None:
    return request_id_var.get()


def get_user_id() -> str | None:
    return user_id_var.get()

