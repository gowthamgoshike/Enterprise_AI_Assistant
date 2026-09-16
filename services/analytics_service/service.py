from services.analytics_service.repository import (
    get_revenue_summary,
    get_top_products,
    get_transaction_metrics,
)


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