"""Privacy-conscious JSON logging configuration."""

import json
import logging
from datetime import datetime, timezone

from core.observability.context import request_id_var, user_id_var


EXTRA_FIELDS = (
    "event",
    "tool",
    "model",
    "latency_ms",
    "model_queue_wait_ms",
    "status",
    "status_code",
    "error_code",
    "attempt",
    "max_attempts",
    "retry_delay",
    "operation",
    "node",
    "role",
    "method",
    "path",
    "message_length",
    "authorization_result",
    "authorization_latency",
    "action_id",
    "action_type",
    "job_id",
    "job_type",
    "stage",
    "thread_id",
    "route",
    "decision",
    "result",
    "cache_type",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(
                record,
                "request_id",
                request_id_var.get(),
            ),
            "user_id": getattr(record, "user_id", user_id_var.get()),
        }
        for field in EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler._enterprise_json_handler = True  # type: ignore[attr-defined]

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
