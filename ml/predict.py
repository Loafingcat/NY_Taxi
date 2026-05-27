from pathlib import Path

import joblib
import pandas as pd

from ml.features import FEATURE_COLUMNS


MODEL_PATH = Path("ml/models/fare_model.pkl")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델 파일이 없습니다. 먼저 학습을 실행하세요: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_fare(input_data: dict) -> float:
    model = load_model()

    df = pd.DataFrame([input_data])

    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"예측 입력에 필요한 컬럼이 없습니다: {missing_columns}")

    prediction = model.predict(df[FEATURE_COLUMNS])[0]

    return round(float(prediction), 2)


if __name__ == "__main__":
    sample_input = {
        "passenger_count": 1,
        "trip_distance": 3.2,
        "pickup_hour": 14,
        "pickup_dayofweek": 2,
        "pickup_month": 1,
        "duration_minutes": 18,
    }

    result = predict_fare(sample_input)
    print("예상 요금:", result)