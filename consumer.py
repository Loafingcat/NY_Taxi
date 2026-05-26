# 스파크로 실시간 데이터를 처리하는 파일
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

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

# 5. 실시간 들어오는 데이터를 콘솔에 출력
query = (
    parsed_df
    .writeStream
    .outputMode("append")
    .format("console")
    .option("truncate", "false")
    .start()
)

query.awaitTermination()