"""Provider-neutral model failures exposed to application code."""

from core.errors import (
    InvalidModelOutputError,
    ModelUnavailableError as CoreModelUnavailableError,
    RetryableError,
)


class ModelError(Exception):
    """Marker for failures originating at the model boundary."""


class ModelUnavailableError(CoreModelUnavailableError, ModelError):
    pass


class ModelTimeoutError(RetryableError, ModelError):
    code = "model_timeout"
    safe_message = "The AI service timed out."


class ModelResponseError(InvalidModelOutputError, ModelError):
    pass
