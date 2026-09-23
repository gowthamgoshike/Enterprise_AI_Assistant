from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any


class BaseModelProvider(ABC):
    """Thin provider contract built on LangChain message and runnable types."""

    @abstractmethod
    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        ...

    @abstractmethod
    def astream(self, messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        ...

    @abstractmethod
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        ...

    @abstractmethod
    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        ...


# Kept as an import-compatible alias for the original abstraction.
LLMProvider = BaseModelProvider
