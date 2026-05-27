from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


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