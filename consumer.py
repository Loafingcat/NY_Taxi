# Kafka → Spark Structured Streaming → Cleaned Parquet 저장용 운영 consumer

import os

# =========================
# Windows Spark 실행 환경 설정
# =========================
os.makedirs(r"C:\spark_tmp", exist_ok=True)

os.environ["JAVA_HOME"] = (
    r"C:\Users\금정산2-PC12\AppData\Local\Programs\Eclipse Adoptium"
    r"\jdk-17.0.19.10-hotspot"
)
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["hadoop.home.dir"] = r"C:\hadoop"
os.environ["SPARK_LOCAL_DIRS"] = r"C:\spark_tmp"
os.environ["TEMP"] = r"C:\spark_tmp"
os.environ["TMP"] = r"C:\spark_tmp"
os.environ["PATH"] = (
    r"C:\Users\금정산2-PC12\AppData\Local\Programs\Eclipse Adoptium"
    r"\jdk-17.0.19.10-hotspot\bin;"
    r"C:\hadoop\bin;"
    + os.environ["PATH"]
)

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType


# =========================
# 운영 설정
# =========================
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092",
)

KAFKA_TOPIC = os.getenv(
    "KAFKA_TOPIC",
    "nyc-taxi-trips",
)

STARTING_OFFSETS = os.getenv(
    "STARTING_OFFSETS",
    "latest",
)

FAIL_ON_DATA_LOSS = os.getenv(
    "FAIL_ON_DATA_LOSS",
    "false",
)

OUTPUT_PATH = os.getenv(
    "OUTPUT_PATH",
    "output/parquet",
)

CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH",
    "checkpoint/raw_to_parquet",
)



# =========================
# Spark 세션 생성
# =========================
spark = (
    SparkSession.builder
    .appName("NYCTaxiKafkaToParquet")
    .master("local[*]")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1",
    )
    .config("spark.local.dir", "C:/spark_tmp")
    .config("spark.sql.shuffle.partitions", "4")
    .config(
        "spark.sql.streaming.checkpointFileManagerClass",
        "org.apache.spark.sql.execution.streaming.FileSystemBasedCheckpointFileManager",
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# =========================
# Kafka 메시지 스키마
# =========================
schema = StructType(
    [
        StructField("tpep_pickup_datetime", StringType(), True),
        StructField("tpep_dropoff_datetime", StringType(), True),
        StructField("passenger_count", DoubleType(), True),
        StructField("trip_distance", DoubleType(), True),
        StructField("fare_amount", DoubleType(), True),
    ]
)


# =========================
# Kafka 스트림 읽기
# =========================
raw_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", STARTING_OFFSETS)
    .option("failOnDataLoss", FAIL_ON_DATA_LOSS)
    .load()
)


# =========================
# JSON 파싱
# =========================
parsed_df = (
    raw_df
    .selectExpr("CAST(value AS STRING) AS value")
    .select(from_json(col("value"), schema).alias("data"))
    .select("data.*")
)


# =========================
# 모델 학습용 정제 데이터
# =========================
clean_df = parsed_df.filter(
    (col("tpep_pickup_datetime").isNotNull())
    & (col("tpep_dropoff_datetime").isNotNull())
    & (col("fare_amount").isNotNull())
    & (col("trip_distance").isNotNull())
    & (col("passenger_count").isNotNull())
    & (col("fare_amount") >= 0)
    & (col("fare_amount") <= 300)
    & (col("trip_distance") > 0)
    & (col("trip_distance") <= 100)
    & (col("passenger_count") > 0)
    & (col("passenger_count") <= 8)
)


# =========================
# Parquet 누적 저장
# =========================
query = (
    clean_df.writeStream
    .outputMode("append")
    .format("parquet")
    .option("path", OUTPUT_PATH)
    .option("checkpointLocation", CHECKPOINT_PATH)
    .start()
)


print("=" * 80)
print("NY Taxi Spark Consumer Started")
print(f"Kafka Bootstrap Servers : {KAFKA_BOOTSTRAP_SERVERS}")
print(f"Kafka Topic             : {KAFKA_TOPIC}")
print(f"Starting Offsets        : {STARTING_OFFSETS}")
print(f"Output Path             : {OUTPUT_PATH}")
print(f"Checkpoint Path         : {CHECKPOINT_PATH}")
print("Pipeline                : Kafka → Spark → Cleaned Parquet → ML Training")
print("=" * 80)

query.awaitTermination()