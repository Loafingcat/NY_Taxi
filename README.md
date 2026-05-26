# NY Taxi Real-Time Streaming Pipeline

## 1. 프로젝트 소개

NY Taxi 운행 데이터를 활용해 **Kafka 기반 실시간 데이터 파이프라인**을 구축한 프로젝트입니다.

원본 데이터는 Parquet 파일 형태로 `data/` 폴더에 저장되어 있으며, `producer.py`가 이 데이터를 한 줄씩 읽어 Kafka Topic으로 전송합니다. 이후 `consumer.py`는 Spark Structured Streaming을 통해 Kafka 데이터를 실시간으로 읽고, JSON 파싱, 스키마 적용, 이상치 필터링, 실시간 집계, Parquet 저장, PostgreSQL 적재를 수행합니다.

이 프로젝트는 단순히 Kafka 메시지를 주고받는 실습이 아니라, 실제 데이터 엔지니어링 환경에서 사용되는 **실시간 수집 → 스트리밍 처리 → 저장 → 집계** 흐름을 직접 구현하고 검증하는 것을 목표로 했습니다.

---

## 2. 주요 기능

- Kafka Producer를 통한 NY Taxi 데이터 실시간 전송
- Spark Structured Streaming 기반 실시간 Consumer 구현
- Kafka 메시지 JSON 파싱 및 명시적 스키마 적용
- 음수 요금, 0 거리, 승객 수 0 등 이상치 필터링
- 1분 단위 운행 수, 평균 요금, 평균 이동거리 집계
- Parquet 파일 저장
- PostgreSQL 적재 구조 구성
- Spark Checkpoint 기반 offset 관리
- Windows 환경에서 PySpark 실행 환경 구성 및 트러블슈팅

---

## 3. 프로젝트 구조

```text
NY_Taxi/
├─ data/
│  └─ yellow_tripdata_2026-01.parquet
├─ output/
│  └─ parquet/
├─ checkpoint/
│  ├─ aggregation/
│  ├─ console/
│  ├─ postgres/
│  └─ raw_to_parquet/
├─ producer.py
├─ consumer.py
├─ docker-compose.yml
├─ requirements.txt
└─ README.md
```

---

## 4. 전체 흐름

```text
NY Taxi Parquet Data
        ↓
producer.py
        ↓
Kafka Topic: nyc-taxi-trips
        ↓
consumer.py / Spark Structured Streaming
        ↓
JSON Parsing
        ↓
Schema Validation
        ↓
Data Quality Filtering
        ↓
Console / Parquet / PostgreSQL / Aggregation
```

---

## 5. 기술 스택

| 구분 | 기술 |
|---|---|
| Language | Python |
| Message Broker | Apache Kafka |
| Stream Processing | PySpark Structured Streaming |
| Database | PostgreSQL |
| Container | Docker Compose |
| Data Format | Parquet, JSON |

---

## 6. Producer 설명

`producer.py`는 Parquet 파일을 읽어 Kafka Topic으로 데이터를 전송합니다.

처리 흐름은 다음과 같습니다.

```text
Parquet 데이터 로딩
→ pickup 시간 기준 정렬
→ datetime 컬럼 문자열 변환
→ row 단위 dict 변환
→ JSON 직렬화
→ Kafka Topic 전송
```

전송 Topic은 다음과 같습니다.

```text
nyc-taxi-trips
```

전송 로그 예시입니다.

```text
Sent: Pickup=2026-01-01 00:07:09, Fare=$7.9
Sent: Pickup=2026-01-01 00:07:10, Fare=$10.7
```

처리량 측정을 위해 일정 건수마다 전송 속도도 확인할 수 있도록 구성했습니다.

```text
Sent 1000 messages
Producer rate: 9.88 msg/sec
```

---

## 7. Consumer 설명

`consumer.py`는 Kafka Topic을 구독하고 Spark Structured Streaming으로 데이터를 처리합니다.

처리 흐름은 다음과 같습니다.

```text
Kafka readStream
→ CAST(value AS STRING)
→ from_json()으로 JSON 파싱
→ 명시적 Schema 적용
→ 이상치 필터링
→ 출력 모드별 처리
```

주요 스키마는 다음과 같습니다.

| 컬럼명 | 설명 |
|---|---|
| `tpep_pickup_datetime` | 택시 승차 시간 |
| `tpep_dropoff_datetime` | 택시 하차 시간 |
| `passenger_count` | 승객 수 |
| `trip_distance` | 운행 거리 |
| `fare_amount` | 운행 요금 |

---

## 8. 데이터 품질 필터링

실시간 데이터에는 음수 요금, 이동거리 0, 승객 수 0 같은 비정상 데이터가 포함될 수 있어 Spark 처리 단계에서 기본적인 필터링을 적용했습니다.

```python
clean_df = parsed_df.filter(
    (col("fare_amount").isNotNull()) &
    (col("trip_distance").isNotNull()) &
    (col("passenger_count").isNotNull()) &
    (col("fare_amount") >= 0) &
    (col("fare_amount") <= 300) &
    (col("trip_distance") > 0) &
    (col("passenger_count") > 0)
)
```

이 필터링을 통해 분석이나 모델 학습에 사용할 수 있는 최소한의 정제 데이터를 만들 수 있도록 했습니다.

---

## 9. 실시간 집계

Spark Structured Streaming의 window 연산을 활용해 1분 단위 집계를 수행했습니다.

집계 항목은 다음과 같습니다.

- 1분 단위 운행 건수
- 1분 단위 평균 요금
- 1분 단위 평균 이동거리

출력 예시는 다음과 같습니다.

```text
+------------------------------------------+----------+------------------+-----------------+
|window                                    |trip_count|avg_fare          |avg_distance     |
+------------------------------------------+----------+------------------+-----------------+
|{2026-01-01 00:01:00, 2026-01-01 00:02:00}|23        |26.05217391304348 |5.355652173913044|
+------------------------------------------+----------+------------------+-----------------+
```

같은 시간 구간에 데이터가 추가되면 `update` 모드로 집계 결과가 계속 갱신됩니다.

---

## 10. Checkpoint

Spark Structured Streaming은 checkpoint를 통해 Kafka offset과 처리 상태를 저장합니다.

이 프로젝트에서는 처리 목적에 따라 checkpoint 경로를 분리했습니다.

```text
checkpoint/
├─ aggregation/
├─ console/
├─ postgres/
└─ raw_to_parquet/
```

checkpoint를 사용하면 스트리밍 작업이 중단되더라도 Kafka offset을 기준으로 이어서 처리할 수 있는 구조를 만들 수 있습니다.

---

## 11. 실행 방법

Docker 컨테이너 실행:

```cmd
docker compose up -d
```

Kafka Topic 생성:

```cmd
docker exec -it ny_taxi-kafka-1 kafka-topics --bootstrap-server localhost:9092 --create --topic nyc-taxi-trips --partitions 1 --replication-factor 1
```

Topic 확인:

```cmd
docker exec -it ny_taxi-kafka-1 kafka-topics --bootstrap-server localhost:9092 --list
```

Producer 실행:

```cmd
python producer.py
```

Consumer 실행:

```cmd
python consumer.py
```

---

## 12. 검증 결과

### Producer → Kafka

Producer가 NY Taxi 데이터를 Kafka Topic으로 정상 전송하는 것을 확인했습니다.

```text
Sent: Pickup=2026-01-01 00:07:09, Fare=$7.9
```

### Kafka → Spark Consumer

Spark Consumer가 Kafka Topic을 구독하고 micro-batch 단위로 데이터를 출력하는 것을 확인했습니다.

```text
Batch: 23
Batch: 24
```

### Window Aggregation

1분 단위 실시간 집계 결과가 정상 출력되는 것을 확인했습니다.

```text
trip_count, avg_fare, avg_distance 값이 batch 단위로 갱신됨
```

---

## 13. 트러블슈팅

## 13.1 Java 미설치 문제

### 문제

```text
'java'은(는) 내부 또는 외부 명령, 실행할 수 있는 프로그램, 또는 배치 파일이 아닙니다.
```

### 원인

PySpark는 내부적으로 JVM 기반 Spark 엔진을 실행하기 때문에 Java JDK가 필요합니다.

### 해결

Eclipse Temurin JDK 17을 설치하고 환경변수를 설정했습니다.

```text
JAVA_HOME=C:\Users\금정산2-PC12\AppData\Local\Programs\Eclipse Adoptium\jdk-17.0.19.10-hotspot
Path=%JAVA_HOME%\bin
```

---

## 13.2 JAVA_HOME 경로 오류

### 문제

```text
지정된 경로를 찾을 수 없습니다.
```

### 원인

`JAVA_HOME`이 실제 JDK 설치 경로가 아닌 `C:\openjdk-24.0.1`을 보고 있었습니다.

### 해결

`JAVA_HOME`을 실제 JDK 17 설치 경로로 수정했습니다.

---

## 13.3 HADOOP_HOME 미설정 문제

### 문제

```text
HADOOP_HOME and hadoop.home.dir are unset
Did not find winutils.exe
```

### 원인

Windows에서 PySpark를 실행할 때 Hadoop 관련 Windows binary 파일이 필요했습니다.

### 해결

`C:\hadoop\bin`에 필요한 파일을 배치하고 환경변수를 설정했습니다.

```text
HADOOP_HOME=C:\hadoop
hadoop.home.dir=C:\hadoop
Path=%HADOOP_HOME%\bin
```

필수 파일은 다음과 같습니다.

```text
winutils.exe
hadoop.dll
hdfs.dll
```

---

## 13.4 Spark Master 연결 실패

### 문제

```text
Failed to connect to master localhost:7077
Connection refused
```

### 원인

코드에서 Spark Master를 `spark://localhost:7077`로 설정했지만, 실제 Spark Standalone Master를 실행하지 않은 상태였습니다.

### 해결

로컬 실습 환경에 맞게 Spark 실행 모드를 변경했습니다.

```python
.master("local[*]")
```

---

## 13.5 Kafka Broker 미실행 문제

### 문제

```text
Connection to node -1 (localhost/127.0.0.1:9092) could not be established.
Broker may not be available.
```

### 원인

Kafka Broker가 실행되지 않은 상태에서 Consumer를 실행했습니다.

### 해결

Docker Compose로 Kafka와 Zookeeper를 실행했습니다.

```cmd
docker compose up -d
```

---

## 13.6 Batch 0이 비어 있는 문제

### 문제

Consumer 실행 시 Batch가 출력되지만 데이터가 비어 있었습니다.

### 원인

`startingOffsets`가 `latest`로 설정되어 있어 Consumer 실행 이후 들어오는 데이터만 읽도록 되어 있었습니다.

### 해결

Consumer를 먼저 실행한 뒤 Producer를 실행했습니다.

테스트 목적에서는 다음 설정도 사용할 수 있습니다.

```python
.option("startingOffsets", "earliest")
```

---

## 13.7 Windows 임시 폴더 문제

### 문제

```text
Java gateway process exited before sending its port number
```

### 원인

Spark가 사용할 임시 폴더가 없거나 Windows 사용자 경로 문제로 임시 파일 생성에 실패했습니다.

### 해결

영문 경로의 Spark 임시 폴더를 만들고 Spark 설정에 반영했습니다.

```cmd
mkdir C:\spark_tmp
mkdir C:\spark_checkpoint
```

```python
.config("spark.local.dir", "C:/spark_tmp")
```

---

## 14. 현재까지의 성과

이번 프로젝트를 통해 다음 내용을 직접 구현하고 검증했습니다.

- Docker Compose 기반 Kafka 실행
- Kafka Topic 생성
- Python Kafka Producer 구현
- PySpark Structured Streaming Consumer 구현
- Kafka 메시지 JSON 파싱
- 명시적 스키마 적용
- 실시간 이상치 필터링
- Micro-batch 처리 확인
- 1분 단위 window aggregation 구현
- Checkpoint 기반 스트리밍 상태 관리
- Windows 환경에서 PySpark 실행 문제 해결

---

## 15. 포트폴리오 설명 문장

NY Taxi 대용량 운행 데이터를 실시간 이벤트처럼 Kafka Topic으로 전송하고, Spark Structured Streaming을 통해 Kafka 데이터를 구독하여 JSON 파싱, 스키마 적용, 이상치 필터링, 1분 단위 window aggregation을 수행하는 실시간 데이터 파이프라인을 구현했습니다.

또한 checkpointLocation을 명시하여 Kafka offset 기반으로 장애 복구가 가능한 구조를 구성했고, Windows 로컬 환경에서 PySpark 실행을 위해 Java, Hadoop, winutils, hadoop.dll 관련 문제를 직접 해결했습니다.

---

## 16. 향후 확장 방향

현재 프로젝트는 실시간 데이터 수집 및 처리 계층입니다. 이후에는 정제된 데이터를 기반으로 머신러닝 모델 학습과 API 서빙 구조로 확장할 수 있습니다.

가장 적합한 예측 문제는 다음과 같습니다.

```text
택시 운행 요금 예측 모델
```

목표 변수는 다음과 같습니다.

```text
fare_amount
```

입력 변수 후보는 다음과 같습니다.

```text
passenger_count
trip_distance
pickup_hour
pickup_dayofweek
pickup_month
duration_minutes
distance_per_minute
```

확장 흐름은 다음과 같습니다.

```text
Kafka 실시간 수집
→ Spark Streaming 이상치 필터링
→ Parquet 또는 PostgreSQL 저장
→ Feature Engineering
→ 모델 학습
→ 모델 평가
→ 모델 저장
→ FastAPI 예측 API
→ Dashboard 연결
```

---

## 17. MLOps / CI-CD 확장 계획

향후에는 다음과 같은 구조로 MLOps 2단계 수준의 자동화를 구성할 수 있습니다.

```text
NY_Taxi/
├─ ml/
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ predict.py
│  ├─ features.py
│  ├─ models/
│  └─ metrics/
│
├─ api/
│  └─ main.py
│
├─ scripts/
│  ├─ init_db.sql
│  ├─ create_topic.bat
│  ├─ run_train.bat
│  └─ run_api.bat
│
├─ .github/
│  └─ workflows/
│     └─ ci.yml
```

자동화 목표는 다음과 같습니다.

- Docker Compose 기반 Kafka/PostgreSQL 실행 자동화
- Kafka Topic 생성 스크립트 작성
- PostgreSQL 테이블 초기화 SQL 작성
- 정제된 Parquet 데이터 기반 모델 학습
- MAE, RMSE, R2 기준 모델 평가
- 학습된 모델 파일 저장
- FastAPI 기반 예측 API 구성
- GitHub Actions 기반 코드 문법 검사 및 학습 smoke test 자동화

최종 확장 구조는 다음과 같습니다.

```text
NY Taxi Data
→ Kafka Producer
→ Kafka Topic
→ Spark Structured Streaming
→ Cleaned Parquet / PostgreSQL
→ Batch Training Pipeline
→ Model File
→ FastAPI Prediction API
→ Streamlit Dashboard
```