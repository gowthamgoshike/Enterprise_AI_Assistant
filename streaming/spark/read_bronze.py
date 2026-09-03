from pyspark.sql import SparkSession

from paths import BRONZE_TRANSACTIONS


spark = (
    SparkSession.builder
    .appName("ReadBronzeTransactions")
    .getOrCreate()
)


df = spark.read.parquet(
    BRONZE_TRANSACTIONS
)


df.printSchema()

df.show(
    truncate=False
)