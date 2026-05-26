# 스파크로 실시간 데이터를 처리하는 파일
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import from_json, col, to_timestamp, window, count, avg

# 1. Spark 세션 생성
# - local[*] : 별도의 Spark Master 서버 없이 현재 PC에서 로컬 모드로 실행
# - spark-sql-kafka 패키지 : Spark가 Kafka 데이터를 읽기 위해 필요
spark = (
    SparkSession.builder
    .appName("NYCTaxiStreaming")
    .master("local[*]")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1"
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# 2. 들어올 데이터의 스키마 정의
# Kafka에서 들어오는 value는 문자열 JSON이므로,
# from_json()으로 구조화된 컬럼으로 변환하기 위해 스키마를 정의한다.
schema = StructType([
    StructField("tpep_pickup_datetime", StringType(), True),
    StructField("tpep_dropoff_datetime", StringType(), True),
    StructField("passenger_count", DoubleType(), True),
    StructField("trip_distance", DoubleType(), True),
    StructField("fare_amount", DoubleType(), True),
])

# 3. Kafka에서 실시간 스트림 읽기
# Kafka 서버가 localhost:9092에서 실행 중이어야 한다.
# topic 이름은 producer 쪽과 동일해야 한다.
df = (
    spark
    .readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "nyc-taxi-trips")
    .option("startingOffsets", "latest")
    .load()
)

# 4. Kafka의 value 컬럼을 문자열로 변환한 뒤 JSON 파싱
# Kafka value는 binary 타입으로 들어오기 때문에 CAST(value AS STRING)이 필요하다.
parsed_df = (
    df.selectExpr("CAST(value AS STRING) AS value")
    .select(from_json(col("value"), schema).alias("data"))
    .select("data.*")
)
# Spark Structured Streaming의 window 연산과 watermark를 활용해 1분 단위 운행 수, 평균 요금, 평균 이동거리를 실시간으로 집계
stream_df = parsed_df.withColumn(
    "pickup_time",
    to_timestamp(col("tpep_pickup_datetime"))
)

clean_df = parsed_df.filter(
    (col("fare_amount") >= 0) &
    (col("fare_amount") <= 300) &
    (col("trip_distance") > 0) &
    (col("passenger_count") > 0)
)

agg_df = (
    clean_df
    .withWatermark("pickup_time", "10 minutes")
    .groupBy(window(col("pickup_time"), "1 minute"))
    .agg(
        count("*").alias("trip_count"),
        avg("fare_amount").alias("avg_fare"),
        avg("trip_distance").alias("avg_distance")
    )
)

# 5. Spark Structured Streaming의 checkpointLocation을 명시해, 스트리밍 처리 중 장애가 발생해도 Kafka offset 기준으로 이어서 처리할 수 있도록 구성
query = (
    agg_df
    .writeStream
    .outputMode("update")
    .format("console")
    .option("truncate", "false")
    .option("checkpointLocation", "checkpoint/aggregation")
    .start()
)

query.awaitTermination()