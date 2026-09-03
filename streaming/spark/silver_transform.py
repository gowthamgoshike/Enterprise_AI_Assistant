from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

from paths import (
    BRONZE_TRANSACTIONS,
    SILVER_TRANSACTIONS,
)


spark = (
    SparkSession.builder
    .appName("EnterpriseTransactionSilverTransform")
    .getOrCreate()
)


bronze_df = spark.read.parquet(
    BRONZE_TRANSACTIONS
)


silver_df = bronze_df.filter(
    col("event_id").isNotNull()
    & col("transaction_id").isNotNull()
    & col("customer_id").isNotNull()
)


silver_df = silver_df.filter(
    col("amount_cents") > 0
)


silver_df = silver_df.filter(
    col("currency").isin(
        "USD",
        "EUR",
        "GBP"
    )
)


silver_df = silver_df.dropDuplicates(
    ["event_id"]
)


silver_df = silver_df.withColumn(
    "amount_usd",
    col("amount_cents") / 100.0
)


silver_df = silver_df.withColumn(
    "transaction_date",
    to_date("event_time")
)


silver_df = silver_df.select(
    "event_id",
    "transaction_id",
    "customer_id",
    "product_id",
    "amount_cents",
    "amount_usd",
    "currency",
    "status",
    "transaction_type",
    "source",
    "event_time",
    "transaction_date",
)


(
    silver_df
    .write
    .mode("overwrite")
    .partitionBy(
        "transaction_date"
    )
    .parquet(
        SILVER_TRANSACTIONS
    )
)


spark.stop()

