import asyncio
import json
import logging

from core.cache import cache_get, cache_set
from core.config import settings
from services.analytics_service.repository import (
    get_revenue_summary,
    get_top_products,
    get_transaction_metrics,
)


logger = logging.getLogger(__name__)


def fetch_top_products(
    limit: int = 5,
) -> dict:

    products = get_top_products(
        limit=limit
    )

    return {
        "found": len(products) > 0,
        "products": products,
    }


def fetch_revenue_summary() -> dict:

    summary = get_revenue_summary()

    return {
        "found": True,
        "summary": summary,
    }


def fetch_transaction_metrics() -> dict:

    metrics = get_transaction_metrics()

    return {
        "found": len(metrics) > 0,
        "metrics": metrics,
    }


def _json_value(value):
    """Keep cold and cached results identical, including Decimal values."""
    return json.loads(json.dumps(value, default=str))


async def fetch_top_products_async(limit: int = 5) -> dict:
    key = f"v1:analytics:top_products:{limit}"
    cached = await cache_get(key)
    if cached is not None:
        return cached
    logger.info("analytics_db_query", extra={
        "event": "analytics_db_query", "cache_type": "top_products",
    })
    result = _json_value(await asyncio.to_thread(fetch_top_products, limit=limit))
    await cache_set(key, result, settings.redis_cache_ttl_seconds)
    return result


async def fetch_revenue_summary_async() -> dict:
    key = "v1:analytics:revenue_summary"
    cached = await cache_get(key)
    if cached is not None:
        return cached
    logger.info("analytics_db_query", extra={
        "event": "analytics_db_query", "cache_type": "revenue_summary",
    })
    result = _json_value(await asyncio.to_thread(fetch_revenue_summary))
    await cache_set(key, result, settings.redis_cache_ttl_seconds)
    return result


async def fetch_transaction_metrics_async() -> dict:
    key = "v1:analytics:transaction_metrics"
    cached = await cache_get(key)
    if cached is not None:
        return cached
    logger.info("analytics_db_query", extra={
        "event": "analytics_db_query", "cache_type": "transaction_metrics",
    })
    result = _json_value(await asyncio.to_thread(fetch_transaction_metrics))
    await cache_set(key, result, settings.redis_cache_ttl_seconds)
    return result
