import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from ml.predict import predict_fare
from ml.train import train


REGISTRY_PATH = Path("ml/registry/model_registry.json")
METRICS_PATH = Path("ml/metrics/metrics.json")
PRODUCTION_MODEL_PATH = Path("ml/models/production_model.pkl")


app = FastAPI(
    title="NY Taxi Fare Prediction API",
    description="NY Taxi 운행 정보를 기반으로 예상 요금을 예측하고 모델 학습을 관리하는 API",
    version="1.0.0",
)


class FarePredictRequest(BaseModel):
    passenger_count: int = Field(
        ...,
        description="승객 수. 정수만 허용됩니다.",
    )
    trip_distance: float = Field(
        ...,
        description="이동 거리. 모델 입력 기준은 mile입니다.",
    )
    pickup_hour: int = Field(
        ...,
        description="탑승 시간대. 0~23",
    )
    pickup_dayofweek: int = Field(
        ...,
        description="요일. 월요일=0, 일요일=6",
    )
    pickup_month: int = Field(
        ...,
        description="월. 1~12",
    )
    duration_minutes: float = Field(
        ...,
        description="운행 시간. 분 단위",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "passenger_count": 1,
                "trip_distance": 3.2,
                "pickup_hour": 14,
                "pickup_dayofweek": 2,
                "pickup_month": 1,
                "duration_minutes": 18,
            }
        }
    )


class TrainRequest(BaseModel):
    sample: bool = Field(
        default=False,
        description="True이면 CI/CD 또는 테스트용 내장 샘플 데이터로 학습합니다.",
    )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "ny-taxi-fare-api",
        "version": "1.0.1",
        "production_model_exists": PRODUCTION_MODEL_PATH.exists(),
    }


@app.post("/predict")
def predict(request: FarePredictRequest):
    try:
        input_data = request.model_dump()
        predicted_fare = predict_fare(input_data)

        return {
            "predicted_fare": predicted_fare,
            "unit": "USD",
            "model_source": "production_model.pkl",
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"예측 처리 중 오류가 발생했습니다: {e}",
        )


@app.get("/model-info")
def get_model_info():
    """
    현재 Production 모델과 최신 학습 결과 정보를 반환한다.
    """
    registry = None
    metrics = None

    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return {
        "production_model_exists": PRODUCTION_MODEL_PATH.exists(),
        "production_model_path": str(PRODUCTION_MODEL_PATH),
        "registry": registry,
        "latest_metrics": metrics,
    }


@app.post("/train")
def train_model(request: TrainRequest):
    """
    모델 재학습을 실행한다.

    sample=True:
    - 내장 샘플 데이터로 빠르게 학습한다.

    sample=False:
    - output/parquet 데이터를 기반으로 실제 학습한다.
    """
    try:
        train(sample=request.sample)

        registry = None
        metrics = None

        if REGISTRY_PATH.exists():
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                registry = json.load(f)

        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        return {
            "status": "success",
            "message": "모델 학습이 완료되었습니다.",
            "sample": request.sample,
            "production_model_exists": PRODUCTION_MODEL_PATH.exists(),
            "registry": registry,
            "latest_metrics": metrics,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"모델 학습 중 오류가 발생했습니다: {e}",
        )