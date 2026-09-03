from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    sum as spark_sum,
)

from paths import (
    GOLD_CUSTOMER_DAILY_SPEND,
    GOLD_PRODUCT_REVENUE,
    GOLD_TRANSACTION_METRICS,
    SILVER_TRANSACTIONS,
)


spark = (
    SparkSession.builder
    .appName("EnterpriseTransactionGoldTransform")
    .getOrCreate()
)


silver_df = spark.read.parquet(
    SILVER_TRANSACTIONS
)


product_revenue_df = (
    silver_df
    .filter(
        col("status") == "completed"
    )
    .groupBy(
        "product_id"
    )
    .agg(
        spark_sum(
            "amount_usd"
        ).alias(
            "total_revenue"
        ),
        count(
            "*"
        ).alias(
            "transaction_count"
        )
    )
)


(
    product_revenue_df
    .write
    .mode("overwrite")
    .parquet(
        GOLD_PRODUCT_REVENUE
    )
)


transaction_metrics_df = (
    silver_df
    .groupBy(
        "status"
    )
    .agg(
        count("*").alias(
            "transaction_count"
        )
    )
)


(
    transaction_metrics_df
    .write
    .mode("overwrite")
    .parquet(
        GOLD_TRANSACTION_METRICS
    )
)


customer_daily_spend_df = (
    silver_df
    .filter(
        col("status") == "completed"
    )
    .groupBy(
        "customer_id",
        "transaction_date"
    )
    .agg(
        spark_sum(
            "amount_usd"
        ).alias(
            "daily_spend"
        )
    )
)


(
    customer_daily_spend_df
    .write
    .mode("overwrite")
    .partitionBy(
        "transaction_date"
    )
    .parquet(
        GOLD_CUSTOMER_DAILY_SPEND
    )
)


spark.stop()
