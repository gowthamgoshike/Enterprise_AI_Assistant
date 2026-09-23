from enum import StrEnum

from pydantic import BaseModel, Field


class ModelProviderName(StrEnum):
    OLLAMA = "ollama"
    VLLM = "vllm"


class ProviderCapabilities(BaseModel):
    tool_calling: bool = False
    structured_output: bool = False
    streaming: bool = False


class ModelProviderInfo(BaseModel):
    provider: ModelProviderName
    model: str
    base_url_type: str = "local"
    capabilities: ProviderCapabilities


class ModelConfig(BaseModel):
    name: str
    temperature: float = Field(default=0.1, ge=0, le=2)
    timeout_seconds: float = Field(default=60, gt=0)
