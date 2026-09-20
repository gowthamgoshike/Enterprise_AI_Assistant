from typing import Literal

from pydantic import BaseModel


class DependencyStatus(BaseModel):
    status: Literal["healthy", "unavailable"]
    latency_ms: float | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    api: DependencyStatus
    database: DependencyStatus
    checkpoint_store: DependencyStatus
    model: DependencyStatus
