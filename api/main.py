from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

from ml.predict import predict_fare


app = FastAPI(
    title="NY Taxi Fare Prediction API",
    description="NY Taxi 운행 정보를 기반으로 예상 요금을 예측하는 API",
    version="1.0.0",
)


class FarePredictRequest(BaseModel):
    passenger_count: float = Field(...)
    trip_distance: float = Field(...)
    pickup_hour: int = Field(...)
    pickup_dayofweek: int = Field(...)
    pickup_month: int = Field(...)
    duration_minutes: float = Field(...)

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


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "ny-taxi-fare-api",
    }


@app.post("/predict")
def predict(request: FarePredictRequest):
    input_data = request.model_dump()

    predicted_fare = predict_fare(input_data)

    return {
        "predicted_fare": predicted_fare,
        "unit": "USD",
    }