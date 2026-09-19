import asyncio
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from ai.evaluations.context_benchmark import (
    ContextBenchmarkResult,
    build_benchmark_messages,
    format_results,
    measure_context_latency,
)


def test_build_benchmark_messages_creates_requested_valid_context() -> None:
    messages = build_benchmark_messages(10)

    assert len(messages) == 10
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[-1], HumanMessage)


@pytest.mark.parametrize("message_count", [0, 1, 3])
def test_build_benchmark_messages_rejects_invalid_sizes(message_count) -> None:
    with pytest.raises(ValueError):
        build_benchmark_messages(message_count)


def test_measure_context_latency_invokes_model_once() -> None:
    model = AsyncMock()

    result = asyncio.run(measure_context_latency(model, 2))

    model.ainvoke.assert_awaited_once()
    assert result.message_count == 2
    assert result.estimated_tokens > 0
    assert result.latency_seconds >= 0


def test_format_results_creates_markdown_table() -> None:
    result = ContextBenchmarkResult(
        message_count=2,
        estimated_tokens=18,
        latency_seconds=1.2345,
    )

    table = format_results([result])

    assert "| Messages | Est. Tokens | Latency |" in table
    assert "| 2 | 18 | 1.234s |" in table
