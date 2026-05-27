import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ml.features import (
    load_parquet_data,
    create_features,
    split_features_target,
)


MODEL_DIR = Path("ml/models")
METRICS_DIR = Path("ml/metrics")

MODEL_PATH = MODEL_DIR / "fare_model.pkl"
METRICS_PATH = METRICS_DIR / "metrics.json"


def evaluate_model(model, X_test, y_test):
    """
    학습된 모델을 평가하고 MAE, RMSE, R2를 반환한다.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
    }


def train(sample: bool = False):
    """
    정제된 Parquet 데이터를 기반으로 여러 회귀 모델을 학습하고,
    MAE 기준으로 가장 좋은 모델을 저장한다.

    sample=True인 경우:
    - CI/CD 또는 빠른 테스트용 샘플 데이터를 사용한다.
    - GitHub Actions 환경에는 output/parquet 데이터가 없을 수 있으므로,
      내장 샘플 데이터를 사용하도록 구성한다.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if sample:
        from ml.features import create_sample_training_data

        print("CI/CD 샘플 학습 모드: 내장 샘플 데이터를 사용합니다.")
        df = create_sample_training_data()
    else:
        print("정제된 Parquet 데이터를 로딩합니다.")
        df = load_parquet_data("output/parquet")

    print(f"로드된 데이터 수: {len(df)}")

    df = create_features(df)
    print(f"Feature 생성 후 데이터 수: {len(df)}")

    X, y = split_features_target(df)

    if len(X) < 10:
        raise ValueError("학습에 사용할 데이터가 너무 적습니다.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    candidate_models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor_default": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "RandomForestRegressor_tuned": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    best_model_name = None
    best_model = None
    best_mae = float("inf")

    for model_name, model in candidate_models.items():
        print("=" * 60)
        print(f"모델 학습 시작: {model_name}")

        model.fit(X_train, y_train)

        score = evaluate_model(model, X_test, y_test)

        result = {
            "model": model_name,
            **score,
        }

        results.append(result)

        print("모델 평가 결과:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        if score["mae"] < best_mae:
            best_mae = score["mae"]
            best_model_name = model_name
            best_model = model

    if best_model is None:
        raise RuntimeError("최적 모델을 선택하지 못했습니다.")

    joblib.dump(best_model, MODEL_PATH)

    metrics = {
        "best_model": best_model_name,
        "best_metric": "mae",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "total_rows": int(len(X)),
        "results": results,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("최적 모델 저장 완료:", MODEL_PATH)
    print("성능 지표 저장 완료:", METRICS_PATH)
    print("최종 선택 모델:", best_model_name)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        action="store_true",
        help="CI/CD 또는 빠른 테스트용 샘플 학습 모드",
    )
    args = parser.parse_args()

    train(sample=args.sample)