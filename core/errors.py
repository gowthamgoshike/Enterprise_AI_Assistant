"""Stable application error taxonomy used across dependency boundaries."""


class AppError(Exception):
    """Base application exception safe for deterministic classification."""

    code = "application_error"
    safe_message = "The operation could not be completed."
    retryable = False

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.safe_message)


class RetryableError(AppError):
    """Temporary failure that may succeed when retried."""

    retryable = True


class NonRetryableError(AppError):
    """Failure that retrying cannot safely correct."""


class SecurityError(NonRetryableError):
    """Authentication or authorization failure."""


class BusinessRuleError(NonRetryableError):
    """Valid request that violates a business rule."""


class ModelUnavailableError(RetryableError):
    code = "model_unavailable"
    safe_message = "The AI service is temporarily unavailable."


class DatabaseUnavailableError(RetryableError):
    code = "database_unavailable"
    safe_message = "The data service is temporarily unavailable."


class ToolTimeoutError(RetryableError):
    code = "tool_timeout"
    safe_message = "The operation timed out."


class InvalidModelOutputError(RetryableError):
    code = "invalid_model_output"
    safe_message = "The assistant could not produce a valid response."


class TransactionNotFoundError(BusinessRuleError, ValueError):
    code = "transaction_not_found"
    safe_message = "The transaction was not found."


class RefundNotAllowedError(BusinessRuleError, ValueError):
    code = "refund_not_allowed"
    safe_message = "The transaction is not eligible for a refund."


class PermissionDeniedError(SecurityError):
    code = "permission_denied"
    safe_message = "You are not authorized to perform this action."
