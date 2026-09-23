from collections.abc import AsyncIterator, Sequence
from typing import Any

from ai.models.base import BaseModelProvider
from ai.models.types import (
    ModelProviderInfo,
    ModelProviderName,
    ProviderCapabilities,
)
from core.config import Settings, settings


class VLLMProvider(BaseModelProvider):
    """Reserved provider skeleton for the Day 37 implementation."""

    def __init__(self, config: Settings = settings) -> None:
        raise NotImplementedError(
            "The vLLM provider is reserved for the Day 37 implementation"
        )

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def astream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        raise NotImplementedError
        yield  # pragma: no cover

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        raise NotImplementedError

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def health_check(self) -> bool:
        return False

    @property
    def metadata(self) -> ModelProviderInfo:
        return ModelProviderInfo(
            provider=ModelProviderName.VLLM,
            model="unconfigured",
            base_url_type="local",
            capabilities=ProviderCapabilities(),
        )
