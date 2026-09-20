from sqlalchemy import text

from data.database.connection import engine
from data.database.reliability import translate_database_errors


TOP_PRODUCTS_QUERY = text("""
    SELECT
        product_id,
        total_revenue,
        transaction_count

    FROM product_revenue

    ORDER BY total_revenue DESC

    LIMIT :limit
""")


REVENUE_SUMMARY_QUERY = text("""
    SELECT
        COALESCE(
            SUM(total_revenue),
            0
        ) AS total_revenue,

        COALESCE(
            SUM(transaction_count),
            0
        ) AS transaction_count,

        COUNT(*) AS product_count

    FROM product_revenue
""")


TRANSACTION_METRICS_QUERY = text("""
    SELECT
        status,
        transaction_count

    FROM transaction_metrics

    ORDER BY status
""")


def get_top_products(
    limit: int = 5,
) -> list[dict]:

    with translate_database_errors(), engine.connect() as connection:

        result = connection.execute(
            TOP_PRODUCTS_QUERY,
            {
                "limit": limit,
            },
        )

        return [
            dict(row)
            for row
            in result.mappings()
        ]


def get_revenue_summary() -> dict:

    with translate_database_errors(), engine.connect() as connection:

        result = connection.execute(
            REVENUE_SUMMARY_QUERY
        )

        row = (
            result
            .mappings()
            .first()
        )

        return dict(row)


def get_transaction_metrics() -> list[dict]:

    with translate_database_errors(), engine.connect() as connection:

        result = connection.execute(
            TRANSACTION_METRICS_QUERY
        )

        return [
            dict(row)
            for row
            in result.mappings()
        ]
