from langchain_core.tools import tool

from services.analytics_service.service import (
    fetch_revenue_summary,
    fetch_top_products,
    fetch_transaction_metrics,
)


@tool
def get_top_products(
    limit: int = 5,
) -> dict:
    """
    Return the highest-revenue products from
    the current enterprise analytics dataset.

    Use this tool when the user asks for top
    products, best-selling products by revenue,
    or highest-revenue products.

    Rank products by total revenue, not units sold.
    The limit defaults to 5 and is clamped to 1 through 20.
    Returns found and a products list ordered by revenue.
    """

    limit = max(
        1,
        min(
            limit,
            20,
        ),
    )

    return fetch_top_products(
        limit=limit
    )


@tool
def get_revenue_summary() -> dict:
    """
    Return the revenue summary from the
    currently loaded enterprise Gold
    analytics dataset.

    Use this tool when the user asks for
    the overall current revenue summary,
    aggregate revenue, transaction count,
    or number of products represented in
    the revenue dataset. For counts grouped by status,
    use get_transaction_metrics instead.
    Returns found and a summary of aggregate totals.

    Do not use this tool to claim revenue
    for a specific date or time range.
    """

    return fetch_revenue_summary()


@tool
def get_transaction_metrics() -> dict:
    """
    Return transaction counts grouped by
    transaction status from the current
    enterprise analytics dataset.

    Use this tool for questions about
    completed, failed, pending, or refunded
    transaction counts. Use this for status breakdowns,
    not revenue totals or a specific transaction lookup.
    Returns found and a metrics list grouped by status.
    """

    return fetch_transaction_metrics()