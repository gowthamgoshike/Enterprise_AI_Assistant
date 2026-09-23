from ai.models.base import BaseModelProvider
from ai.models.ollama_provider import OllamaProvider
from ai.models.types import ModelProviderName
from core.config import Settings, settings


def create_model_provider(
    provider_name: str | ModelProviderName,
    config: Settings = settings,
) -> BaseModelProvider:
    try:
        name = ModelProviderName(provider_name)
    except ValueError as exc:
        raise ValueError(f"Unsupported model provider: {provider_name}") from exc

    if name is ModelProviderName.OLLAMA:
        return OllamaProvider(config)
    if name is ModelProviderName.VLLM:
        from ai.models.vllm_provider import VLLMProvider

        return VLLMProvider(config)
    raise ValueError(f"Unsupported model provider: {provider_name}")
