import asyncio
from types import SimpleNamespace

import pytest

from ai.models.factory import create_model_provider
from ai.models.gateway import BoundModel, ModelGateway


class FakeRunnable:
    def __init__(self, active=None):
        self.active = active or {"count": 0, "peak": 0}

    async def ainvoke(self, messages, **kwargs):
        self.active["count"] += 1
        self.active["peak"] = max(self.active["peak"], self.active["count"])
        await asyncio.sleep(0.01)
        self.active["count"] -= 1
        return messages

    async def astream(self, messages, **kwargs):
        for message in messages:
            yield message

    def bind_tools(self, tools, **kwargs):
        return FakeRunnable(self.active)

    def with_structured_output(self, schema, **kwargs):
        return FakeRunnable(self.active)


class FakeProvider(FakeRunnable):
    metadata = {"provider": "fake", "model": "test"}

    async def health_check(self):
        return True


def config(**overrides):
    values = {
        "model_max_concurrency": 1,
        "model_timeout_seconds": 1,
        "health_check_timeout_seconds": 1,
        "model_name": "test",
        "model_provider": "fake",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_gateway_bounds_base_and_bound_models():
    provider = FakeProvider()
    gateway = ModelGateway(provider, config())
    bound = gateway.bind_tools([])

    assert isinstance(bound, BoundModel)
    await asyncio.gather(gateway.ainvoke("base"), bound.ainvoke("bound"))
    assert provider.active["peak"] == 1


@pytest.mark.asyncio
async def test_structured_output_uses_gateway_policy():
    provider = FakeProvider()
    gateway = ModelGateway(provider, config())
    structured = gateway.with_structured_output(dict)

    assert isinstance(structured, BoundModel)
    assert await structured.ainvoke({"route": "general"}) == {
        "route": "general"
    }


@pytest.mark.asyncio
async def test_gateway_streaming_and_health_check():
    gateway = ModelGateway(FakeProvider(), config())
    assert [chunk async for chunk in gateway.astream(["a", "b"])] == ["a", "b"]
    assert await gateway.health_check() is True


@pytest.mark.asyncio
async def test_gateway_times_out():
    class SlowProvider(FakeProvider):
        async def ainvoke(self, messages, **kwargs):
            await asyncio.sleep(0.05)

    gateway = ModelGateway(SlowProvider(), config(model_timeout_seconds=0.001))
    with pytest.raises(Exception, match="timed out"):
        await gateway.ainvoke("hello")


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported model provider: unknown"):
        create_model_provider("unknown", config())
