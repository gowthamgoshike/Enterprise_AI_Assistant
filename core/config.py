"""Environment-backed reliability configuration."""

import os

from dotenv import load_dotenv


load_dotenv()


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


MODEL_TIMEOUT_SECONDS = _positive_float(
    "MODEL_TIMEOUT_SECONDS",
    float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "60")),
)
MODEL_MAX_RETRIES = _positive_int("MODEL_MAX_RETRIES", 2)
TOOL_MAX_RETRIES = _positive_int("TOOL_MAX_RETRIES", 2)
DATABASE_MAX_RETRIES = _positive_int("DATABASE_MAX_RETRIES", 2)
TOOL_TIMEOUT_SECONDS = _positive_float("TOOL_TIMEOUT_SECONDS", 30.0)
RETRY_BASE_DELAY_SECONDS = _positive_float("RETRY_BASE_DELAY_SECONDS", 0.5)
RETRY_MAX_DELAY_SECONDS = _positive_float("RETRY_MAX_DELAY_SECONDS", 4.0)
HEALTH_CHECK_TIMEOUT_SECONDS = _positive_float(
    "HEALTH_CHECK_TIMEOUT_SECONDS", 5.0
)

if RETRY_MAX_DELAY_SECONDS < RETRY_BASE_DELAY_SECONDS:
    raise ValueError(
        "RETRY_MAX_DELAY_SECONDS must be greater than or equal to "
        "RETRY_BASE_DELAY_SECONDS"
    )
