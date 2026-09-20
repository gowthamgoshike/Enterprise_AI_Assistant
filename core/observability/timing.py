"""Reusable operation and LangGraph node timing helpers."""

import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from typing import TypeVar

from core.observability.metrics import (
    GRAPH_NODE_DURATION,
    GRAPH_NODE_EXECUTIONS,
)


logger = logging.getLogger(__name__)
T = TypeVar("T")


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000


@contextmanager
def observe_database_query(operation: str):
    status = "failure"
    started = time.perf_counter()
    try:
        yield
        status = "success"
    finally:
        logger.debug(
            "database_query_completed",
            extra={
                "event": "database_query",
                "operation": operation,
                "status": status,
                "latency_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
            },
        )


async def observe_async_node(
    node_name: str,
    function: Callable[[dict], Awaitable[T]],
    state: dict,
) -> T:
    logger.info(
        "graph_node_started",
        extra={"event": "graph_node", "node": node_name, "status": "started"},
    )
    status = "failure"
    started = time.perf_counter()
    try:
        result = await function(state)
        status = "success"
        return result
    finally:
        duration_seconds = time.perf_counter() - started
        GRAPH_NODE_EXECUTIONS.labels(node=node_name, status=status).inc()
        GRAPH_NODE_DURATION.labels(node=node_name).observe(duration_seconds)
        logger.info(
            "graph_node_completed",
            extra={
                "event": "graph_node",
                "node": node_name,
                "status": status,
                "latency_ms": round(duration_seconds * 1000, 2),
            },
        )


def observe_node(
    node_name: str,
    function: Callable[[dict], T],
    state: dict,
) -> T:
    logger.info(
        "graph_node_started",
        extra={"event": "graph_node", "node": node_name, "status": "started"},
    )
    status = "failure"
    started = time.perf_counter()
    try:
        result = function(state)
        status = "success"
        return result
    except Exception as exc:
        if type(exc).__name__ in {"GraphInterrupt", "GraphBubbleUp"}:
            status = "interrupted"
        raise
    finally:
        duration_seconds = time.perf_counter() - started
        GRAPH_NODE_EXECUTIONS.labels(node=node_name, status=status).inc()
        GRAPH_NODE_DURATION.labels(node=node_name).observe(duration_seconds)
        logger.info(
            "graph_node_completed",
            extra={
                "event": "graph_node",
                "node": node_name,
                "status": status,
                "latency_ms": round(duration_seconds * 1000, 2),
            },
        )
