from types import SimpleNamespace

import httpx
import pytest

from ai.models.factory import create_model_provider
from ai.models.types import ModelProviderName
from ai.models.vllm_provider import VLLMProvider


class FakeRunnable:
    async def ainvoke(self, messages, **kwargs):
        return {"messages": messages, "kwargs": kwargs}

    async def astream(self, messages, **kwargs):
        for message in messages:
            yield message

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        return messages


def config(**overrides):
    values = {
        "model_name": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "model_base_url": "http://host.docker.internal:8001/v1",
        "model_api_key": None,
        "model_timeout_seconds": 60,
        "health_check_timeout_seconds": 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_factory_creates_vllm_provider():
    provider = create_model_provider(ModelProviderName.VLLM, config())

    assert isinstance(provider, VLLMProvider)
    assert provider.metadata.provider is ModelProviderName.VLLM
    assert provider.metadata.capabilities.tool_calling is True


@pytest.mark.asyncio
async def test_vllm_invocation_streaming_and_bound_models(monkeypatch):
    options = []

    def fake_model(**kwargs):
        options.append(kwargs)
        return FakeRunnable()

    monkeypatch.setattr("ai.models.vllm_provider.ChatOpenAI", fake_model)
    provider = VLLMProvider(config())

    result = await provider.ainvoke(["hello"], max_tokens=10)
    chunks = [chunk async for chunk in provider.astream(["a", "b"])]
    bound_result = await provider.bind_tools([]).ainvoke(["tool"])
    structured_result = await provider.with_structured_output(dict).ainvoke(
        ["structured"]
    )

    assert result["messages"] == ["hello"]
    assert chunks == ["a", "b"]
    assert bound_result["messages"] == ["tool"]
    assert structured_result["messages"] == ["structured"]
    assert options[0] == {
        "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "base_url": "http://host.docker.internal:8001/v1",
        "api_key": "EMPTY",
        "temperature": 0.1,
        "timeout": 60,
    }


@pytest.mark.asyncio
async def test_vllm_health_check_uses_openai_models_endpoint(monkeypatch):
    request = httpx.Request("GET", "http://host.docker.internal:8001/v1/models")
    response = httpx.Response(200, request=request)
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, **kwargs):
            calls.append({"url": url, **kwargs})
            return response

    monkeypatch.setattr("ai.models.vllm_provider.httpx.AsyncClient", FakeClient)

    assert await VLLMProvider(config()).health_check() is True
    assert calls[1]["url"] == "http://host.docker.internal:8001/v1/models"
    assert calls[1]["headers"] == {"Authorization": "Bearer EMPTY"}


@pytest.mark.asyncio
async def test_vllm_health_check_returns_false_when_unavailable(monkeypatch):
    class FailingClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, **kwargs):
            raise httpx.ConnectError("unavailable")

    monkeypatch.setattr("ai.models.vllm_provider.httpx.AsyncClient", FailingClient)

    assert await VLLMProvider(config()).health_check() is False
