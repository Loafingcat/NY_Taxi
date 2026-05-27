import json
import os
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer


KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092",
)

TOPIC_NAME = os.getenv(
    "KAFKA_TOPIC",
    "nyc-taxi-trips",
)

FILE_PATH = os.getenv(
    "TAXI_DATA_PATH",
    "./data/yellow_tripdata_2026-01.parquet",
)

STATE_PATH = Path(
    os.getenv("PRODUCER_STATE_PATH", "producer_state.json")
)

SEND_INTERVAL_SECONDS = float(
    os.getenv("PRODUCER_SEND_INTERVAL_SECONDS", "0")
)

PROGRESS_SAVE_INTERVAL = int(
    os.getenv("PRODUCER_PROGRESS_SAVE_INTERVAL", "1000")
)


def load_state() -> int:
    if not STATE_PATH.exists():
        return 0

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)

    return int(state.get("last_sent_index", 0))


def save_state(last_sent_index: int):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "last_sent_index": last_sent_index,
                "data_path": FILE_PATH,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    print("=" * 80)
    print("NY Taxi Kafka Producer Started")
    print(f"Kafka Bootstrap Servers : {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Kafka Topic             : {TOPIC_NAME}")
    print(f"Data Path               : {FILE_PATH}")
    print(f"State Path              : {STATE_PATH}")
    print(f"Send Interval Seconds   : {SEND_INTERVAL_SECONDS}")
    print("=" * 80)

    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
        value_serializer=lambda x: json.dumps(x).encode("utf-8"),
        acks="all",
        retries=5,
        linger_ms=50,
        batch_size=32768,
    )

    print("전체 Parquet 데이터 로딩 중...")
    df = pd.read_parquet(FILE_PATH)
    df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)

    total_rows = len(df)
    start_index = load_state()

    print(f"전체 데이터 수: {total_rows}")
    print(f"이전 마지막 전송 위치: {start_index}")

    if start_index >= total_rows:
        print("이미 전체 데이터를 Kafka로 전송했습니다.")
        producer.close()
        return

    df["tpep_pickup_datetime"] = df["tpep_pickup_datetime"].astype(str)
    df["tpep_dropoff_datetime"] = df["tpep_dropoff_datetime"].astype(str)

    target_df = df.iloc[start_index:]

    print(f"이번 실행 전송 시작 index: {start_index}")
    print(f"이번 실행 전송 대상 건수: {len(target_df)}")

    start_time = time.time()
    sent_count = 0
    last_index = start_index

    try:
        for index, row in target_df.iterrows():
            message = row.to_dict()

            producer.send(TOPIC_NAME, value=message)

            sent_count += 1
            last_index = index + 1

            if sent_count % PROGRESS_SAVE_INTERVAL == 0:
                producer.flush()
                save_state(last_index)

                elapsed = time.time() - start_time
                rate = sent_count / elapsed if elapsed > 0 else 0

                print(
                    f"Sent {sent_count} messages "
                    f"/ current_index={last_index} "
                    f"/ rate={rate:.2f} msg/sec"
                )

            if SEND_INTERVAL_SECONDS > 0:
                time.sleep(SEND_INTERVAL_SECONDS)

        producer.flush()
        save_state(last_index)

        elapsed = time.time() - start_time
        rate = sent_count / elapsed if elapsed > 0 else 0

        print("=" * 80)
        print("전체 데이터 전송 완료")
        print(f"전송 건수: {sent_count}")
        print(f"마지막 저장 index: {last_index}")
        print(f"평균 전송 속도: {rate:.2f} msg/sec")
        print("=" * 80)

    except KeyboardInterrupt:
        print("사용자 중단 감지. 현재 위치를 저장합니다.")
        producer.flush()
        save_state(last_index)
        print(f"저장된 마지막 index: {last_index}")

    finally:
        producer.close()


if __name__ == "__main__":
    main()