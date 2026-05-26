# 카프카로 데이터를 쏘는 시뮬레이터
import pandas as pd
import json
import time
from kafka import KafkaProducer


# =========================
# 1. 기본 설정
# =========================
TOPIC_NAME = "nyc-taxi-trips"
FILE_PATH = "./data/yellow_tripdata_2026-01.parquet"

# 테스트용 전송 개수
LIMIT_ROWS = 10000

# 실시간 느낌을 주기 위한 딜레이
# 0.1초면 초당 약 10건
# 대량 처리량 테스트를 하고 싶으면 0 또는 0.001 정도로 줄이면 됨
SEND_DELAY = 0.1


# =========================
# 2. Kafka Producer 생성
# =========================
producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode("utf-8")
)


# =========================
# 3. 데이터 로딩
# =========================
print("데이터 로딩 중...")

df = pd.read_parquet(FILE_PATH).head(LIMIT_ROWS)
df = df.sort_values("tpep_pickup_datetime")

# JSON 전송을 위해 날짜 컬럼을 문자열로 변환
df["tpep_pickup_datetime"] = df["tpep_pickup_datetime"].astype(str)
df["tpep_dropoff_datetime"] = df["tpep_dropoff_datetime"].astype(str)

print(f"총 {len(df)}건의 데이터를 Kafka로 전송합니다.")
print("실시간 스트리밍 시뮬레이션을 시작합니다.")


# =========================
# 4. Kafka로 한 줄씩 전송
# =========================
# Producer 단계에서 메시지 전송량을 msg/sec 기준으로 측정해, 로컬 환경에서의 처리량을 확인
count = 0
start_time = time.time()

for index, row in df.iterrows():
    message = row.to_dict()

    producer.send(TOPIC_NAME, value=message)

    count += 1

    print(
        f"Sent: Pickup={message['tpep_pickup_datetime']}, "
        f"Fare=${message['fare_amount']}"
    )

    # 1000건마다 처리량 출력
    if count % 1000 == 0:
        elapsed = time.time() - start_time
        rate = count / elapsed

        print("=" * 60)
        print(f"Sent {count} messages")
        print(f"Elapsed time: {elapsed:.2f} sec")
        print(f"Producer rate: {rate:.2f} msg/sec")
        print("=" * 60)

    # 실시간 느낌을 주기 위한 딜레이
    if SEND_DELAY > 0:
        time.sleep(SEND_DELAY)


# =========================
# 5. 남은 메시지 전송 보장
# =========================
producer.flush()
producer.close()

total_elapsed = time.time() - start_time
final_rate = count / total_elapsed if total_elapsed > 0 else 0

print("Kafka 전송 완료")
print(f"Total sent: {count} messages")
print(f"Total elapsed time: {total_elapsed:.2f} sec")
print(f"Average producer rate: {final_rate:.2f} msg/sec")