import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from ai.models.base import BaseModelProvider
from ai.models.errors import ModelTimeoutError
from ai.models.reliability import invoke_model_with_retry
from core.config import Settings, settings
from core.observability.metrics import (
    MODEL_FAILURES_TOTAL,
    MODEL_LATENCY,
    MODEL_QUEUE_WAIT,
    MODEL_REQUESTS_TOTAL,
    MODEL_TIMEOUTS_TOTAL,
)


class BoundModel:
    """A LangChain runnable guarded by the same policies as the base model."""

    def __init__(self, runnable: Any, gateway: "ModelGateway") -> None:
        self.runnable = runnable
        self._gateway = gateway

    async def ainvoke(
        self,
        messages: Any,
        *,
        request_id: str | None = None,
        operation_name: str = "model_call",
        on_retry: Any = None,
        **kwargs: Any,
    ) -> Any:
        return await self._gateway._invoke(
            lambda: self.runnable.ainvoke(messages, **kwargs),
            request_id=request_id,
            operation_name=operation_name,
            on_retry=on_retry,
        )

    async def astream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async for chunk in self._gateway._stream(
            lambda: self.runnable.astream(messages, **kwargs)
        ):
            yield chunk

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "BoundModel":
        return BoundModel(self.runnable.bind_tools(tools, **kwargs), self._gateway)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "BoundModel":
        return BoundModel(
            self.runnable.with_structured_output(schema, **kwargs), self._gateway
        )


class ModelGateway:
    def __init__(
        self,
        provider: BaseModelProvider,
        config: Settings = settings,
    ) -> None:
        self.provider = provider
        self.config = config
        self._semaphore = asyncio.Semaphore(config.model_max_concurrency)

    @property
    def metadata(self) -> Any:
        return self.provider.metadata

    async def _invoke(
        self,
        operation: Any,
        *,
        request_id: str | None = None,
        operation_name: str = "model_call",
        on_retry: Any = None,
    ) -> Any:
        return await invoke_model_with_retry(
            operation,
            request_id=request_id,
            operation_name=operation_name,
            on_retry=on_retry,
            semaphore=self._semaphore,
            timeout_seconds=self.config.model_timeout_seconds,
            model_name=self.config.model_name,
            provider_name=self.config.model_provider,
        )

    async def ainvoke(
        self,
        messages: Any,
        *,
        request_id: str | None = None,
        operation_name: str = "model_call",
        on_retry: Any = None,
        **kwargs: Any,
    ) -> Any:
        return await self._invoke(
            lambda: self.provider.ainvoke(messages, **kwargs),
            request_id=request_id,
            operation_name=operation_name,
            on_retry=on_retry,
        )

    async def _stream(self, stream_factory: Any) -> AsyncIterator[Any]:
        # The slot remains held for the full stream lifetime. A per-item timeout
        # would allow an indefinitely long stream, so bound the whole iteration.
        queued_at = time.perf_counter()
        async with self._semaphore:
            labels = {
                "provider": self.config.model_provider,
                "model": self.config.model_name,
            }
            MODEL_QUEUE_WAIT.labels(**labels).observe(
                time.perf_counter() - queued_at
            )
            started = time.perf_counter()
            status = "failure"
            try:
                iterator = stream_factory().__aiter__()
                async with asyncio.timeout(self.config.model_timeout_seconds):
                    async for chunk in iterator:
                        yield chunk
                status = "success"
            except TimeoutError as exc:
                MODEL_TIMEOUTS_TOTAL.labels(**labels).inc()
                raise ModelTimeoutError() from exc
            finally:
                MODEL_REQUESTS_TOTAL.labels(**labels, status=status).inc()
                MODEL_LATENCY.labels(**labels).observe(
                    time.perf_counter() - started
                )
                if status == "failure":
                    MODEL_FAILURES_TOTAL.labels(**labels).inc()

    async def astream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async for chunk in self._stream(
            lambda: self.provider.astream(messages, **kwargs)
        ):
            yield chunk

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BoundModel:
        return BoundModel(self.provider.bind_tools(tools, **kwargs), self)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> BoundModel:
        return BoundModel(self.provider.with_structured_output(schema, **kwargs), self)

    async def health_check(self) -> bool:
        try:
            async with asyncio.timeout(self.config.health_check_timeout_seconds):
                return await self.provider.health_check()
        except TimeoutError:
            return False
