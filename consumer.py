# 스파크로 실시간 데이터를 처리하는 파일
import os

os.makedirs(r"C:\spark_tmp", exist_ok=True)
os.makedirs(r"C:\spark_checkpoint", exist_ok=True)

os.environ["JAVA_HOME"] = r"C:\Users\금정산2-PC12\AppData\Local\Programs\Eclipse Adoptium\jdk-17.0.19.10-hotspot"
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["hadoop.home.dir"] = r"C:\hadoop"
os.environ["SPARK_LOCAL_DIRS"] = r"C:\spark_tmp"
os.environ["TEMP"] = r"C:\spark_tmp"
os.environ["TMP"] = r"C:\spark_tmp"
os.environ["PATH"] = (
    r"C:\Users\금정산2-PC12\AppData\Local\Programs\Eclipse Adoptium\jdk-17.0.19.10-hotspot\bin;"
    r"C:\hadoop\bin;"
    + os.environ["PATH"]
)

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, window, count, avg
from pyspark.sql.types import StructType, StructField, StringType, DoubleType


# =========================
# 실행 모드 선택
# =========================
# console: 실시간 데이터 화면 출력
# parquet: 실시간 데이터를 parquet 파일로 저장
# aggregation: 1분 단위 집계 출력
# postgres: PostgreSQL DB에 적재
OUTPUT_MODE = "aggregation"


# =========================
# Spark 세션 생성
# =========================
spark = (
    SparkSession.builder
    .appName("NYCTaxiStreaming")
    .master("local[*]")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.postgresql:postgresql:42.7.3"
    )
    .config("spark.local.dir", "C:/spark_tmp")
    .config(
        "spark.sql.streaming.checkpointFileManagerClass",
        "org.apache.spark.sql.execution.streaming.FileSystemBasedCheckpointFileManager"
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# =========================
# Kafka 메시지 스키마
# =========================
schema = StructType([
    StructField("tpep_pickup_datetime", StringType(), True),
    StructField("tpep_dropoff_datetime", StringType(), True),
    StructField("passenger_count", DoubleType(), True),
    StructField("trip_distance", DoubleType(), True),
    StructField("fare_amount", DoubleType(), True),
])


# =========================
# Kafka 스트림 읽기
# =========================
df = (
    spark
    .readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "nyc-taxi-trips")
    .option("startingOffsets", "latest")
    .load()
)


# =========================
# JSON 파싱
# =========================
parsed_df = (
    df.selectExpr("CAST(value AS STRING) AS value")
    .select(from_json(col("value"), schema).alias("data"))
    .select("data.*")
)


# =========================
# 이상치 필터링
# =========================
clean_df = parsed_df.filter(
    (col("fare_amount").isNotNull()) &
    (col("trip_distance").isNotNull()) &
    (col("passenger_count").isNotNull()) &
    (col("fare_amount") >= 0) &
    (col("fare_amount") <= 300) &
    (col("trip_distance") > 0) &
    (col("passenger_count") > 0)
)


# =========================
# PostgreSQL 저장 함수
# =========================
def write_to_postgres(batch_df, batch_id):
    row_count = batch_df.count()

    print(f"Batch ID: {batch_id}, Row Count: {row_count}")

    if row_count == 0:
        return

    (
        batch_df.write
        .format("jdbc")
        .option("url", "jdbc:postgresql://localhost:5432/nytaxi")
        .option("dbtable", "taxi_trips")
        .option("user", "postgres")
        .option("password", "1234")
        .option("driver", "org.postgresql.Driver")
        .mode("append")
        .save()
    )

    print(f"Batch {batch_id} saved to PostgreSQL")


# =========================
# 출력 모드별 실행
# =========================
if OUTPUT_MODE == "console":
    query = (
        clean_df
        .writeStream
        .outputMode("append")
        .format("console")
        .option("truncate", "false")
        .option("checkpointLocation", "C:/spark_checkpoint/console")
        .start()
    )

elif OUTPUT_MODE == "parquet":
    query = (
        clean_df
        .writeStream
        .outputMode("append")
        .format("parquet")
        .option("path", "output/parquet")
        .option("checkpointLocation", "C:/spark_checkpoint/parquet")
        .start()
    )

elif OUTPUT_MODE == "aggregation":
    stream_df = clean_df.withColumn(
        "pickup_time",
        to_timestamp(col("tpep_pickup_datetime"))
    )

    agg_df = (
        stream_df
        .withWatermark("pickup_time", "10 minutes")
        .groupBy(window(col("pickup_time"), "1 minute"))
        .agg(
            count("*").alias("trip_count"),
            avg("fare_amount").alias("avg_fare"),
            avg("trip_distance").alias("avg_distance")
        )
    )

    query = (
        agg_df
        .writeStream
        .outputMode("update")
        .format("console")
        .option("truncate", "false")
        .option("checkpointLocation", "C:/spark_checkpoint/aggregation")
        .start()
    )

elif OUTPUT_MODE == "postgres":
    query = (
        clean_df
        .writeStream
        .foreachBatch(write_to_postgres)
        .option("checkpointLocation", "C:/spark_checkpoint/postgres")
        .start()
    )

else:
    raise ValueError(f"지원하지 않는 OUTPUT_MODE입니다: {OUTPUT_MODE}")


query.awaitTermination()