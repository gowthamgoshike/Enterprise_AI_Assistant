from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window

from data.database.connection import engine
from services.transaction_service.repository import upsert_transaction
from streaming.spark.paths import SILVER_TRANSACTIONS


TRANSACTION_COLUMNS = [
    "transaction_id",
    "event_id",
    "customer_id",
    "product_id",
    "amount_cents",
    "currency",
    "status",
    "transaction_type",
    "source",
    "event_time",
]


def load_transactions() -> int:
    spark = (
        SparkSession.builder
        .appName("LoadTransactionsToPostgres")
        .getOrCreate()
    )

    try:
        silver_df = spark.read.parquet(SILVER_TRANSACTIONS)

        window_spec = (
            Window
            .partitionBy("transaction_id")
            .orderBy(col("event_time").desc())
        )

        latest_df = (
            silver_df
            .withColumn("row_number", row_number().over(window_spec))
            .filter(col("row_number") == 1)
            .drop("row_number")
        )

        pandas_df = (
            latest_df
            .select(*TRANSACTION_COLUMNS)
            .withColumnRenamed("event_id", "latest_event_id")
            .toPandas()
        )

        for transaction in pandas_df.to_dict(orient="records"):
            upsert_transaction(transaction)

        return len(pandas_df)
    finally:
        spark.stop()


if __name__ == "__main__":
    loaded_count = load_transactions()
    print(f"Loaded {loaded_count} transactions")
