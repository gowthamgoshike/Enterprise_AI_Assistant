from pyspark.sql import SparkSession

from data.database.connection import engine
from streaming.spark.paths import GOLD_PRODUCT_REVENUE


def load_product_revenue() -> int:
    spark = (
        SparkSession.builder
        .appName("LoadGoldToPostgres")
        .getOrCreate()
    )

    try:
        product_df = spark.read.parquet(GOLD_PRODUCT_REVENUE)
        pandas_df = product_df.toPandas()
        pandas_df.to_sql(
            "product_revenue",
            engine,
            if_exists="replace",
            index=False,
        )
        return len(pandas_df)
    finally:
        spark.stop()


if __name__ == "__main__":
    loaded_count = load_product_revenue()
    print(f"Loaded {loaded_count} product revenue rows")
