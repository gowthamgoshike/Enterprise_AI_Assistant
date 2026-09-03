import os

from dotenv import load_dotenv


load_dotenv()


DATA_ROOT = os.getenv("DATA_ROOT") or "data_lake"

BRONZE_TRANSACTIONS = (
    f"{DATA_ROOT}/bronze/transactions"
)


SILVER_TRANSACTIONS = (
    f"{DATA_ROOT}/silver/transactions"
)


GOLD_PRODUCT_REVENUE = (
    f"{DATA_ROOT}/gold/product_revenue"
)


GOLD_TRANSACTION_METRICS = (
    f"{DATA_ROOT}/gold/transaction_metrics"
)


GOLD_CUSTOMER_DAILY_SPEND = (
    f"{DATA_ROOT}/gold/customer_daily_spend"
)