from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from ml.train import train


client = TestClient(app)

PRODUCTION_MODEL_PATH = Path("ml/models/production_model.pkl")


@pytest.fixture(scope="session", autouse=True)
def ensure_production_model():
    """
    GitHub Actions 같은 깨끗한 CI 환경에서는 production_model.pkl이 없을 수 있다.
    /predict API는 production_model.pkl을 필요로 하므로,
    테스트 시작 전에 샘플 데이터로 모델을 한 번 학습해 테스트용 production 모델을 생성한다.
    """
    if not PRODUCTION_MODEL_PATH.exists():
        train(sample=True)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert "production_model_exists" in data
    assert "production_model_path" in data
    assert "registry" in data
    assert "latest_metrics" in data


def test_predict_api():
    payload = {
        "passenger_count": 1,
        "trip_distance": 3.2,
        "pickup_hour": 14,
        "pickup_dayofweek": 2,
        "pickup_month": 1,
        "duration_minutes": 18,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "predicted_fare" in data
    assert "unit" in data
    assert data["unit"] == "USD"
    assert data["model_source"] == "production_model.pkl"


def test_train_api_sample():
    payload = {
        "sample": True,
    }

    response = client.post("/train", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert data["sample"] is True
    assert data["production_model_exists"] is True