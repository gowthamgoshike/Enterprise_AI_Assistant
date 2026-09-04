from sqlalchemy import text

from data.database.connection import engine


TOP_PRODUCTS = text("""
    SELECT
        product_id,
        total_revenue,
        transaction_count
    FROM product_revenue
    ORDER BY total_revenue DESC
    LIMIT :limit
""")


def get_top_products(limit: int = 5) -> list[dict]:
    with engine.connect() as connection:
        result = connection.execute(
            TOP_PRODUCTS,
            {"limit": limit},
        )
        return [dict(row) for row in result.mappings()]
