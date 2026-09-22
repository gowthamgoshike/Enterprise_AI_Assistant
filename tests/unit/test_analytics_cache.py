"""Cache behavior at the analytics service boundary, without a Redis server."""

from decimal import Decimal

import pytest

from ai.tools.context import ToolExecutionContext
from ai.tools import secure_executors
from core import cache
from services.analytics_service import service
from services.auth_service.exceptions import AuthorizationError
from services.auth_service.models import UserRole


class MemoryRedis:
    def __init__(self):
        self.values = {}
        self.writes = []
        self.fail = False

    async def get(self, key):
        if self.fail:
            raise ConnectionError("Redis offline")
        return self.values.get(key)

    async def set(self, key, value, ex):
        if self.fail:
            raise ConnectionError("Redis offline")
        self.values[key] = value
        self.writes.append((key, ex))


@pytest.mark.asyncio
async def test_top_products_cache_is_scoped_by_limit_and_has_ttl(
    monkeypatch, caplog,
):
    redis = MemoryRedis()
    monkeypatch.setattr(cache, "redis_client", redis)
    calls = []

    def fetch_top_products(limit):
        calls.append(limit)
        return {"found": True, "products": [
            {"product_id": f"P-{limit}", "total_revenue": Decimal("12.50")},
        ]}

    monkeypatch.setattr(service, "fetch_top_products", fetch_top_products)
    with caplog.at_level("INFO"):
        first = await service.fetch_top_products_async(5)
        assert first == await service.fetch_top_products_async(5)
    assert [
        record.result for record in caplog.records
        if getattr(record, "event", None) == "cache_lookup"
    ] == ["miss", "hit"]
    assert sum(
        getattr(record, "event", None) == "analytics_db_query"
        for record in caplog.records
    ) == 1
    assert (await service.fetch_top_products_async(10))["products"][0][
        "product_id"
    ] == "P-10"
    assert calls == [5, 10]
    assert first["products"][0]["total_revenue"] == "12.50"
    assert redis.writes == [
        ("v1:analytics:top_products:5", service.settings.redis_cache_ttl_seconds),
        ("v1:analytics:top_products:10", service.settings.redis_cache_ttl_seconds),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "key", "result"),
    [
        ("revenue_summary", "v1:analytics:revenue_summary", {
            "found": True, "summary": {"total_revenue": Decimal("8.25")},
        }),
        ("transaction_metrics", "v1:analytics:transaction_metrics", {
            "found": True, "metrics": [{"status": "complete", "count": 2}],
        }),
    ],
)
async def test_other_aggregates_cache_and_fail_open(
    monkeypatch, name, key, result,
):
    redis = MemoryRedis()
    monkeypatch.setattr(cache, "redis_client", redis)
    calls = []

    def fetch():
        calls.append(True)
        return result

    monkeypatch.setattr(service, f"fetch_{name}", fetch)
    operation = getattr(service, f"fetch_{name}_async")
    first = await operation()
    assert await operation() == first
    assert len(calls) == 1
    assert redis.writes[0][0] == key

    redis.fail = True
    assert await operation() == first
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_analytics_permission_is_checked_before_cache(monkeypatch):
    async def unexpected_cache_lookup(*, limit):
        raise AssertionError("cache must not be read")

    monkeypatch.setattr(
        secure_executors, "fetch_top_products_async", unexpected_cache_lookup,
    )
    context = ToolExecutionContext(
        user_id="USER-1", role=UserRole.CUSTOMER,
        customer_id="CUST-1", request_id="REQ-1",
    )
    with pytest.raises(AuthorizationError):
        await secure_executors.execute_get_top_products(context)
