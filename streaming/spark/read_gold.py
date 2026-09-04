from pyspark.sql import SparkSession

from paths import GOLD_PRODUCT_REVENUE

spark = (
    SparkSession.builder
    .appName("ReadGold")
    .getOrCreate()
)


df = spark.read.parquet(
    GOLD_PRODUCT_REVENUE
)


df.orderBy(
    "total_revenue",
    ascending=False
).show(
    10,
    truncate=False
)