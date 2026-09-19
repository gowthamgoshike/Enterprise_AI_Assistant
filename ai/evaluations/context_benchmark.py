"""Measure how local-model latency changes as conversation context grows."""

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ai.context.manager import estimate_message_tokens
from ai.models.langchain_ollama import get_chat_model


DEFAULT_MESSAGE_COUNTS = (2, 10, 20, 40)


class AsyncChatModel(Protocol):
    async def ainvoke(self, messages: list[BaseMessage]): ...


@dataclass(frozen=True)
class ContextBenchmarkResult:
    message_count: int
    estimated_tokens: int
    latency_seconds: float


def build_benchmark_messages(message_count: int) -> list[BaseMessage]:
    """Build a valid synthetic conversation ending with a user request."""
    if message_count < 2 or message_count % 2:
        raise ValueError("message_count must be an even integer of at least 2")

    messages: list[BaseMessage] = [
        SystemMessage(content="You are a concise technical assistant."),
    ]
    for turn in range((message_count - 2) // 2):
        messages.extend([
            HumanMessage(
                content=f"Earlier question {turn}: explain a distributed system concept."
            ),
            AIMessage(
                content=f"Earlier answer {turn}: a concise synthetic benchmark response."
            ),
        ])
    messages.append(HumanMessage(content="Explain Kafka in one sentence."))
    return messages


async def measure_context_latency(
    model: AsyncChatModel,
    message_count: int,
) -> ContextBenchmarkResult:
    """Invoke the model once and capture context size and wall-clock latency."""
    messages = build_benchmark_messages(message_count)
    started_at = perf_counter()
    await model.ainvoke(messages)
    latency_seconds = perf_counter() - started_at

    return ContextBenchmarkResult(
        message_count=len(messages),
        estimated_tokens=estimate_message_tokens(messages),
        latency_seconds=latency_seconds,
    )


async def run_benchmark(
    message_counts: Sequence[int] = DEFAULT_MESSAGE_COUNTS,
) -> list[ContextBenchmarkResult]:
    """Run sequential measurements against the configured local model."""
    model = get_chat_model()
    return [
        await measure_context_latency(model, message_count)
        for message_count in message_counts
    ]


def format_results(results: Sequence[ContextBenchmarkResult]) -> str:
    """Format benchmark measurements as a Markdown table."""
    rows = [
        "| Messages | Est. Tokens | Latency |",
        "| ---: | ---: | ---: |",
    ]
    rows.extend(
        f"| {result.message_count} | {result.estimated_tokens} | "
        f"{result.latency_seconds:.3f}s |"
        for result in results
    )
    return "\n".join(rows)


async def main() -> None:
    print(format_results(await run_benchmark()))


if __name__ == "__main__":
    asyncio.run(main())
