from pyspark.sql import SparkSession

from paths import  SILVER_TRANSACTIONS


spark = (
    SparkSession.builder
    .appName("ReadSilver")
    .getOrCreate()
)


df = spark.read.parquet(
    SILVER_TRANSACTIONS
)


print(
    "Silver count:",
    df.count()
)


df.show(
    20,
    truncate=False
)