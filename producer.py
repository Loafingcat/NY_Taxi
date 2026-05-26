# 카프카로 데이터를 쏘는 시뮬레이터
import pandas as pd
import json
import time
from kafka import KafkaProducer

# 1. 카프카 프로듀서 연결 설정
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

TOPIC_NAME = 'nyc-taxi-trips'
FILE_PATH = './data/yellow_tripdata_2026-01.parquet'

print("데이터 로딩 중...")
# 2. 테스트용으로 만 건만 불러와서 시간순 정렬
df = pd.read_parquet(FILE_PATH).head(10000)
df = df.sort_values('tpep_pickup_datetime')

# JSON 전송을 위해 날짜를 문자열로 형변환
df['tpep_pickup_datetime'] = df['tpep_pickup_datetime'].astype(str)
df['tpep_dropoff_datetime'] = df['tpep_dropoff_datetime'].astype(str)

print("실시간 스트리밍 시뮬레이션을 시작합니다.")

# 3. 한 줄씩 읽어서 카프카로 전송
for index, row in df.iterrows():
    message = row.to_dict()
    producer.send(TOPIC_NAME, value=message)
    
    print(f"Sent: Pickup={message['tpep_pickup_datetime']}, Fare=${message['fare_amount']}")
    
    # 실시간 느낌을 주기 위해 0.1초 딜레이
    time.sleep(0.1) 

producer.flush()