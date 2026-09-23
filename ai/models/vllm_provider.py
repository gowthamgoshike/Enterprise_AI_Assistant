from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError

from ai.models.base import BaseModelProvider
from ai.models.errors import (
    ModelResponseError,
    ModelTimeoutError,
    ModelUnavailableError,
)
from ai.models.types import (
    ModelProviderInfo,
    ModelProviderName,
    ProviderCapabilities,
)
from core.config import Settings, settings


def _normalized_error(exc: Exception) -> Exception:
    if isinstance(exc, (TimeoutError, httpx.TimeoutException, APITimeoutError)):
        return ModelTimeoutError()
    if isinstance(exc, (httpx.NetworkError, ConnectionError, APIConnectionError)):
        return ModelUnavailableError()
    if isinstance(exc, APIStatusError):
        if exc.status_code == 429 or exc.status_code >= 500:
            return ModelUnavailableError()
        return ModelResponseError()
    return exc


class VLLMProvider(BaseModelProvider):
    """OpenAI-compatible LangChain adapter for a vLLM server."""

    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self._model_options = {
            "model": config.model_name,
            "base_url": config.model_base_url.rstrip("/"),
            "api_key": config.model_api_key or "EMPTY",
            "temperature": 0.1,
            "timeout": config.model_timeout_seconds,
        }

    def _new_model(self) -> ChatOpenAI:
        # Construct the client in the event loop that uses it. This keeps the
        # cached provider safe across workers and multi-loop test processes.
        return ChatOpenAI(**self._model_options)

    @property
    def model(self) -> ChatOpenAI:
        """Expose a fresh native model for compatibility and inspection."""
        return self._new_model()

    def _configured_model(self, configure: Any) -> "_DeferredVLLMRunnable":
        return _DeferredVLLMRunnable(self._new_model, configure)

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        try:
            return await self._new_model().ainvoke(messages, **kwargs)
        except Exception as exc:
            normalized = _normalized_error(exc)
            if normalized is exc:
                raise
            raise normalized from exc

    async def astream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        try:
            async for chunk in self._new_model().astream(messages, **kwargs):
                yield chunk
        except Exception as exc:
            normalized = _normalized_error(exc)
            if normalized is exc:
                raise
            raise normalized from exc

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        return self._configured_model(
            lambda model: model.bind_tools(tools, **kwargs)
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        return self._configured_model(
            lambda model: model.with_structured_output(schema, **kwargs)
        )

    async def health_check(self) -> bool:
        headers = {"Authorization": f"Bearer {self.config.model_api_key or 'EMPTY'}"}
        try:
            async with httpx.AsyncClient(
                timeout=self.config.health_check_timeout_seconds
            ) as client:
                response = await client.get(
                    f"{self.config.model_base_url.rstrip('/')}/models",
                    headers=headers,
                )
                response.raise_for_status()
            return True
        except (httpx.HTTPError, OSError):
            return False

    @property
    def metadata(self) -> ModelProviderInfo:
        return ModelProviderInfo(
            provider=ModelProviderName.VLLM,
            model=self.config.model_name,
            base_url_type="local",
            capabilities=ProviderCapabilities(
                tool_calling=True,
                structured_output=True,
                streaming=True,
            ),
        )


class _DeferredVLLMRunnable:
    """Build provider-bound runnables in the event loop that invokes them."""

    def __init__(self, model_factory: Any, configure: Any) -> None:
        self._model_factory = model_factory
        self._configure = configure

    def _runnable(self) -> Any:
        return self._configure(self._model_factory())

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        try:
            return await self._runnable().ainvoke(messages, **kwargs)
        except Exception as exc:
            normalized = _normalized_error(exc)
            if normalized is exc:
                raise
            raise normalized from exc

    async def astream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        try:
            async for chunk in self._runnable().astream(messages, **kwargs):
                yield chunk
        except Exception as exc:
            normalized = _normalized_error(exc)
            if normalized is exc:
                raise
            raise normalized from exc

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        return self._runnable().invoke(messages, **kwargs)
