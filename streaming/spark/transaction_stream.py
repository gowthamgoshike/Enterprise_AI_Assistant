from pyspark.sql import SparkSession

from pyspark.sql.functions import (
    col,
    current_timestamp,
    from_json,
)

from pyspark.sql.types import (
    StructField,
    StructType,
    LongType,
    StringType,
    TimestampType,
)


KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "transactions"


transaction_schema = StructType([
    StructField("event_id", StringType(), False),
    StructField("schema_version", StringType(), False),
    StructField("transaction_id", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("product_id", StringType(), False),
    StructField("amount_cents", LongType(), False),
    StructField("currency", StringType(), False),
    StructField("status", StringType(), False),
    StructField("transaction_type", StringType(), False),
    StructField("source", StringType(), False),
    StructField("event_time", TimestampType(), False),
    StructField("ingestion_time", TimestampType(), False),
])


spark = (
    SparkSession.builder
    .appName(
        "EnterpriseTransactionStream"
    )
    .getOrCreate()
)


spark.sparkContext.setLogLevel(
    "WARN"
)


kafka_df = (
    spark.readStream
    .format("kafka")
    .option(
        "kafka.bootstrap.servers",
        KAFKA_BOOTSTRAP_SERVERS
    )
    .option(
        "subscribe",
        TOPIC
    )
    .option(
        "startingOffsets",
        "earliest"
    )
    .option(
        "failOnDataLoss",
        "false"
    )
    .load()
)


decoded_df = kafka_df.select(
    col("key")
    .cast("string")
    .alias("customer_key"),

    col("value")
    .cast("string")
    .alias("json_value"),

    col("partition"),

    col("offset"),

    col("timestamp")
    .alias("kafka_timestamp"),
)


parsed_df = (
    decoded_df
    .withColumn(
        "data",
        from_json(
            col("json_value"),
            transaction_schema
        )
    )
)


transactions_df = parsed_df.select(
    "data.*",
    "customer_key",
    "partition",
    "offset",
    "kafka_timestamp",
)


valid_df = (
    transactions_df
    .filter(
        col("amount_cents") > 0
    )
    .filter(
        col("currency").isin(
            "USD",
            "EUR",
            "GBP"
        )
    )
    .withColumn(
        "amount_usd",
        col("amount_cents") / 100.0
    )
    .withColumn(
        "processing_time",
        current_timestamp()
    )
)


deduplicated_df = (
    valid_df
    .withWatermark(
        "event_time",
        "10 minutes"
    )
    .dropDuplicates([
        "event_id"
    ])
)


query = (
    deduplicated_df
    .writeStream
    .format("console")
    .outputMode("append")
    .option(
        "truncate",
        False
    )
    .option(
        "checkpointLocation",
        "checkpoints/transactions_console"
    )
    .start()
)


query.awaitTermination()