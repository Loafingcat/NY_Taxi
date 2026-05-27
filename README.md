# NY Taxi MLOps Pipeline

NY Taxi 운행 데이터를 기반으로 택시 요금을 예측하는 머신러닝/MLOps 프로젝트입니다.  
Kafka로 데이터를 주입하고, Spark Structured Streaming으로 데이터를 정제해 Parquet으로 저장한 뒤, 실제 누적 데이터를 기반으로 여러 모델을 학습·비교하고, 기준을 통과한 모델만 FastAPI 서비스에 반영하는 구조입니다.

---

## 1. 프로젝트 목표

이 프로젝트의 목표는 단순히 모델을 한 번 학습하는 것이 아니라, 실제 데이터가 계속 들어오는 상황을 가정해 다음 흐름을 구성하는 것입니다.

```text
택시 데이터 주입
→ Kafka 메시지 전송
→ Spark Streaming 정제
→ Parquet 누적 저장
→ 모델 재학습
→ 모델 성능 비교
→ Production 모델 승격
→ FastAPI 예측 서비스
→ Streamlit 대시보드 확인
```

최종적으로는 데이터 수집, 처리, 학습, 검증, 배포, 대시보드 모니터링까지 이어지는 작은 MLOps 파이프라인을 구현하는 것을 목표로 했습니다.

---

## 2. 전체 아키텍처

```text
Producer
  ↓
Kafka + Zookeeper
  ↓
Spark Structured Streaming Consumer
  ↓
output/parquet
  ↓
ML Training Pipeline
  ↓
production_model.pkl
  ↓
FastAPI
  ↓
Streamlit Dashboard
```

---

## 3. 주요 기술 스택

| 구분 | 사용 기술 |
|---|---|
| 데이터 스트리밍 | Kafka, Zookeeper |
| 스트리밍 처리 | Spark Structured Streaming |
| 데이터 저장 | Parquet, PostgreSQL |
| 모델 학습 | scikit-learn |
| 모델 관리 | joblib, metrics.json, model_registry.json |
| API 서버 | FastAPI |
| 대시보드 | Streamlit |
| 컨테이너 | Docker, Docker Compose |
| CI/CD | GitHub Actions, GHCR |
| 배포 검증 | Kubernetes, Rolling Update, Rollback |

---

## 4. 데이터 흐름

### 4.1 Producer

`producer.py`는 원본 Parquet 택시 데이터를 읽어 Kafka topic으로 전송합니다.

운영 흐름에서는 전체 데이터를 한 번에 처음부터 반복 전송하지 않고, `producer_state.json`을 통해 마지막 전송 위치를 기억하도록 구성했습니다.

```text
producer_state.json
→ 마지막으로 Kafka에 보낸 row index 저장
→ 중간에 끊겨도 다음 실행 시 이어서 전송
```

### 4.2 Kafka

Kafka topic은 다음 이름을 사용합니다.

```text
nyc-taxi-trips
```

Producer가 전송한 택시 운행 데이터는 이 topic에 쌓이고, Spark Consumer가 이 topic을 구독합니다.

### 4.3 Spark Consumer

`consumer.py`는 Kafka 데이터를 Spark Structured Streaming으로 읽고, JSON 메시지를 파싱한 뒤 이상치를 제거합니다.

정제 후 데이터는 모델 학습에 사용할 수 있도록 다음 경로에 Parquet 형식으로 저장됩니다.

```text
output/parquet
```

Spark checkpoint는 다음 경로에 저장됩니다.

```text
checkpoint/raw_to_parquet
```

checkpoint는 Spark가 Kafka offset을 기억하기 위한 용도입니다.  
consumer가 중간에 종료되더라도 다시 실행하면 마지막 처리 위치 이후부터 이어서 처리할 수 있습니다.

---

## 5. 모델 학습 방식

모델 학습은 `ml/train.py`에서 수행합니다.

실제 데이터 학습 시에는 Spark가 저장한 다음 경로를 읽습니다.

```text
output/parquet
```

학습 과정은 다음 순서로 진행됩니다.

```text
1. output/parquet 데이터 로딩
2. feature 생성
3. X, y 분리
4. train/test 데이터 분리
5. 여러 모델 학습
6. MAE, RMSE, R2 평가
7. MAE 기준 Best 후보 모델 선택
8. Quality Gate 검사
9. 기존 Production 모델과 비교
10. 기준 통과 시 production_model.pkl로 승격
```

---

## 6. Feature 구성

모델은 택시 요금인 `fare_amount`를 예측합니다.

입력값으로 사용하는 feature는 다음과 같습니다.

| Feature | 의미 |
|---|---|
| `passenger_count` | 승객 수 |
| `trip_distance` | 이동 거리 |
| `pickup_hour` | 탑승 시간대 |
| `pickup_dayofweek` | 탑승 요일 |
| `pickup_month` | 탑승 월 |
| `duration_minutes` | 운행 시간 |

정답값은 다음 컬럼입니다.

```text
fare_amount
```

즉, 모델은 승객 수, 이동 거리, 시간대, 요일, 월, 운행 시간을 보고 예상 택시 요금을 예측합니다.

---

## 7. 학습/테스트 데이터 분리

모델 학습 시 전체 데이터를 학습용과 테스트용으로 나눕니다.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)
```

- `test_size=0.2`: 전체 데이터 중 20%를 테스트 데이터로 사용
- `random_state=42`: 매번 같은 방식으로 데이터를 나누기 위한 고정값

예를 들어 데이터가 10,000건이면 다음과 같이 나뉩니다.

```text
학습 데이터: 8,000건
테스트 데이터: 2,000건
```

이렇게 분리하는 이유는 모델이 학습 데이터를 단순히 외운 것이 아니라, 처음 보는 데이터에도 잘 예측하는지 확인하기 위해서입니다.

---

## 8. 사용한 모델

현재 학습 파이프라인에서는 3개의 회귀 모델을 비교합니다.

| 모델 | 사용 이유 |
|---|---|
| `LinearRegression` | 가장 기본적인 기준 모델 |
| `RandomForestRegressor_default` | 비선형 패턴을 반영하기 위한 기본 RandomForest |
| `RandomForestRegressor_tuned` | 트리 개수, 깊이 등을 조정한 튜닝 모델 |

### LinearRegression

단순한 선형 회귀 모델입니다.  
택시 요금이 거리나 운행 시간에 따라 어느 정도 일정하게 증가한다고 가정하는 기본 모델입니다.  
다른 모델과 비교하기 위한 기준점으로 사용했습니다.

### RandomForestRegressor

여러 개의 결정트리를 만들고 각 트리의 예측 결과를 평균 내는 모델입니다.  
택시 요금은 이동 거리뿐 아니라 운행 시간, 시간대, 요일 등 여러 조건이 함께 영향을 주기 때문에 RandomForest를 사용했습니다.

### RandomForestRegressor_tuned

기본 RandomForest에서 설정값을 조정한 모델입니다.

```text
n_estimators=200
max_depth=12
min_samples_split=4
min_samples_leaf=2
```

기본 모델보다 과적합을 줄이고 안정적인 성능을 내는지 확인하기 위해 추가했습니다.

---

## 9. 평가 지표

모델 성능은 MAE, RMSE, R2로 평가합니다.

### MAE

```text
Mean Absolute Error
```

평균적으로 실제 요금과 예측 요금이 얼마나 차이 나는지 보여주는 지표입니다.

예를 들어 MAE가 2.1이면 모델이 평균적으로 약 2.1달러 정도 틀린다는 의미입니다.

이 프로젝트에서는 MAE를 Best Model 선정 기준으로 사용했습니다.

### RMSE

```text
Root Mean Squared Error
```

큰 오차에 더 민감한 지표입니다.  
일부 장거리 운행이나 이상치에서 크게 틀리면 RMSE가 커집니다.

### R2

R2는 단순히 평균값으로 예측하는 것보다 모델이 얼마나 더 잘 예측하는지를 보여주는 지표입니다.

```text
R2가 1에 가까움 → 실제 요금 변화 패턴을 잘 설명함
R2가 0에 가까움 → 평균값으로 예측하는 것과 큰 차이가 없음
R2가 음수 → 평균 예측보다도 못함
```

---

## 10. 모델 선택 및 Production 승격

학습된 모델 중 MAE가 가장 낮은 모델을 Best 후보 모델로 선택합니다.

하지만 Best 후보 모델이라고 바로 서비스에 반영하지 않습니다.

Production 모델로 승격되기 위해서는 다음 조건을 통과해야 합니다.

```text
1. Quality Gate 통과
2. 기존 Production 모델보다 성능 개선
```

기준을 통과하면 다음 파일로 저장됩니다.

```text
ml/models/production_model.pkl
```

FastAPI는 이 Production 모델을 사용해 예측을 수행합니다.

---

## 11. 모델 산출물

학습 후 생성되는 주요 파일은 다음과 같습니다.

| 파일/폴더 | 설명 |
|---|---|
| `ml/models/production_model.pkl` | 실제 서비스에서 사용하는 모델 |
| `ml/models/fare_model.pkl` | 기존 호환용 최신 후보 모델 |
| `ml/models/candidates/` | 후보 모델 저장 위치 |
| `ml/models/history/` | 모델 이력 저장 위치 |
| `ml/metrics/metrics.json` | 최근 학습 결과 |
| `ml/metrics/history/` | 학습 결과 이력 |
| `ml/registry/model_registry.json` | Production 모델 및 승격 이력 |
| `ml/registry/train_trigger_state.json` | 자동 재학습 기준 상태 |

---

## 12. FastAPI

FastAPI는 학습된 Production 모델을 사용해 택시 요금을 예측합니다.

주요 API는 다음과 같습니다.

| Endpoint | 설명 |
|---|---|
| `GET /health` | API 상태 확인 |
| `POST /predict` | 택시 요금 예측 |
| `GET /model-info` | 현재 모델 정보 확인 |
| `POST /train` | 모델 재학습 요청 |

예측 요청 예시는 다음과 같습니다.

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"passenger_count\":1,\"trip_distance\":3.2,\"pickup_hour\":14,\"pickup_dayofweek\":2,\"pickup_month\":1,\"duration_minutes\":18}"
```

---

## 13. Streamlit Dashboard

Streamlit 대시보드에서는 다음 내용을 확인할 수 있습니다.

```text
모델 예측
모델 성능 비교
모델 개선 이력
Production 모델 정보
이전 모델 대비 개선율
역대 모델 성능 기록
```

대시보드 실행 주소는 다음과 같습니다.

```text
http://localhost:8501
```

---

## 14. Docker Compose 실행

전체 인프라는 Docker Compose로 실행합니다.

```bash
docker compose up -d
```

실행 후 컨테이너 확인:

```bash
docker ps
```

주요 컨테이너는 다음과 같습니다.

```text
ny_taxi_zookeeper
ny_taxi_kafka
ny_taxi_spark_master
ny_taxi_spark_worker
ny_taxi_postgres
ny_taxi_api
ny_taxi_dashboard
```

---

## 15. Spark UI

Spark Master UI는 다음 주소에서 확인할 수 있습니다.

```text
http://localhost:8080
```

Spark Worker가 정상적으로 연결되면 Worker 상태가 `ALIVE`로 표시됩니다.

---

## 16. 운영 실행 순서

### 1. 인프라 실행

```bash
docker compose up -d
```

### 2. Kafka topic 확인

```bash
docker exec -it ny_taxi_kafka kafka-topics --bootstrap-server localhost:9092 --list
```

topic이 없다면 생성합니다.

```bash
docker exec -it ny_taxi_kafka kafka-topics --bootstrap-server localhost:9092 --create --topic nyc-taxi-trips --partitions 1 --replication-factor 1
```

### 3. Spark Consumer 실행

```bash
python consumer.py
```

### 4. Producer 실행

```bash
python producer.py
```

### 5. Parquet 데이터 확인

```bash
python -c "import pandas as pd; df=pd.read_parquet('output/parquet'); print(len(df)); print(df.head())"
```

### 6. 모델 학습

```bash
python -m ml.train
```

### 7. API/Dashboard 반영

```bash
docker compose restart api dashboard
```

### 8. 대시보드 확인

```text
http://localhost:8501
```

---

## 17. 자동 재학습 흐름

데이터가 계속 누적되면 일정량 이상 쌓였을 때 자동 재학습을 수행할 수 있도록 구성합니다.

```text
output/parquet 데이터 수 확인
→ 마지막 학습 시점 이후 신규 데이터 수 확인
→ 기준 이상이면 python -m ml.train 실행
→ Production 승격 여부 판단
```

예시 실행:

```bash
python scripts/auto_train_if_ready.py
```

초기 기준은 다음과 같이 설정할 수 있습니다.

```text
최소 학습 데이터 수: 5,000건
재학습 기준 신규 데이터 수: 5,000건
```

---

## 18. CI/CD

GitHub Actions를 통해 다음 작업을 자동화했습니다.

```text
Python 의존성 설치
문법 검사
데이터 검증
샘플 학습 smoke test
모델 품질 기준 검사
pytest 실행
모델 산출물 확인
Docker Compose 설정 검증
Docker 이미지 빌드
GHCR 이미지 Push
```

현재 CI는 모델 학습 파이프라인과 API, Docker 이미지 빌드까지 검증합니다.  
GHCR에 이미지를 push한 뒤 Kubernetes 배포로 확장할 수 있도록 구성했습니다.

---

## 19. Kubernetes 배포 검증

FastAPI와 Streamlit은 Kubernetes에 배포해 Rolling Update와 Rollback을 검증했습니다.

검증한 내용은 다음과 같습니다.

```text
FastAPI Deployment 배포
Streamlit Deployment 배포
Service 연결
port-forward 접속 확인
새 이미지로 Rolling Update
이전 버전으로 Rollback
```

주요 명령어는 다음과 같습니다.

```bash
kubectl apply -f k8s/
kubectl rollout status deployment/ny-taxi-api
kubectl rollout undo deployment/ny-taxi-api
```

로컬 확인은 port-forward를 사용했습니다.

```bash
kubectl port-forward svc/ny-taxi-api-service 8000:8000
kubectl port-forward svc/ny-taxi-dashboard-service 8501:8501
```

---

## 20. 주의할 운영 파일

아래 파일과 폴더는 운영 상태를 유지하기 위한 파일입니다.

| 경로 | 설명 |
|---|---|
| `producer_state.json` | Producer가 어디까지 Kafka에 보냈는지 저장 |
| `checkpoint/raw_to_parquet` | Spark가 Kafka offset을 어디까지 처리했는지 저장 |
| `output/parquet` | 모델 학습용 누적 데이터 |
| `ml/models/production_model.pkl` | 실제 서비스 모델 |
| `ml/metrics/metrics.json` | 최근 모델 성능 |
| `ml/registry/model_registry.json` | 모델 승격 및 이력 정보 |

운영 중에는 아래 명령을 주의해야 합니다.

```bash
docker compose down -v
```

`-v` 옵션은 Docker volume까지 삭제할 수 있으므로 Kafka, PostgreSQL 데이터가 사라질 수 있습니다.

---

## 21. 현재까지 구현한 내용

```text
Kafka/Zookeeper 기반 데이터 주입 구조
Spark Structured Streaming 기반 정제 파이프라인
Parquet 누적 저장
실제 데이터 기반 모델 학습
LinearRegression / RandomForest 모델 비교
MAE 기준 Best 후보 모델 선정
Quality Gate 기반 Production 승격
metrics.json / model_registry.json 기반 모델 이력 관리
FastAPI 예측 API
Streamlit 모델 대시보드
GitHub Actions CI
GHCR 이미지 Push
Kubernetes Rolling Update / Rollback 검증
```

---

## 22. 향후 개선 방향

```text
Feature 추가
- 주말 여부
- 야간 여부
- 출퇴근 시간 여부
- 거리 대비 운행 시간

모델 후보 추가
- GradientBoostingRegressor
- HistGradientBoostingRegressor
- XGBoost / LightGBM

자동화 고도화
- 일정량 데이터 누적 시 자동 재학습
- 모델 드리프트 감지
- Kubernetes CronJob 기반 재학습
- Kafka/Spark/PostgreSQL까지 Kubernetes 이전
- CD 자동 배포
```

---

## 23. 프로젝트 요약

이 프로젝트는 NY Taxi 데이터를 활용해 택시 요금을 예측하는 모델을 만들고, 실제 데이터가 계속 들어오는 상황을 가정해 데이터 처리부터 모델 학습, 검증, 서비스 반영까지 연결한 MLOps 파이프라인입니다.

단순히 모델 하나를 학습하는 데서 끝내지 않고, Kafka와 Spark로 데이터를 누적하고, 여러 모델을 비교한 뒤, 기준을 통과한 모델만 Production 모델로 승격시키는 구조를 구현했습니다.

FastAPI와 Streamlit을 통해 예측 서비스와 모델 성능 확인이 가능하며, GitHub Actions와 Kubernetes를 통해 CI/CD와 배포 검증까지 확장했습니다.
