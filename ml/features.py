import glob
from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "duration_minutes",
]

# 모델이 맞춰야 하는 값
TARGET_COLUMN = "fare_amount"


def load_parquet_data(data_dir: str = "output/parquet") -> pd.DataFrame:
    """
    Spark Streaming이 저장한 Parquet 파일을 읽는다.
    output/parquet 아래의 part-*.parquet 파일을 대상으로 한다.
    """
    parquet_files = glob.glob(str(Path(data_dir) / "*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"Parquet 파일을 찾을 수 없습니다. 경로를 확인하세요: {data_dir}"
        )

    df = pd.concat(
        [pd.read_parquet(file) for file in parquet_files],
        ignore_index=True,
    )

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    모델 학습에 사용할 파생 변수를 생성한다.
    """
    required_columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "fare_amount",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")

    df = df.copy()

    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"],
        errors="coerce",
    )
    df["tpep_dropoff_datetime"] = pd.to_datetime(
        df["tpep_dropoff_datetime"],
        errors="coerce",
    )

    df["duration_minutes"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["tpep_pickup_datetime"].dt.month

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    df = df[
        (df["fare_amount"] >= 0)
        & (df["fare_amount"] <= 300)
        & (df["trip_distance"] > 0)
        & (df["passenger_count"] > 0)
        & (df["duration_minutes"] > 0)
        & (df["duration_minutes"] <= 240)
    ]

    return df


def split_features_target(df: pd.DataFrame):
    """
    모델 입력 X와 정답 y를 분리한다.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y


def create_sample_training_data() -> pd.DataFrame:
    data = {
        "tpep_pickup_datetime": [
            "2026-01-01 10:00:00",
            "2026-01-01 11:00:00",
            "2026-01-01 12:00:00",
            "2026-01-01 13:00:00",
            "2026-01-01 14:00:00",
            "2026-01-01 15:00:00",
            "2026-01-01 16:00:00",
            "2026-01-01 17:00:00",
            "2026-01-01 18:00:00",
            "2026-01-01 19:00:00",
            "2026-01-01 20:00:00",
            "2026-01-01 21:00:00",
        ],
        "tpep_dropoff_datetime": [
            "2026-01-01 10:15:00",
            "2026-01-01 11:20:00",
            "2026-01-01 12:08:00",
            "2026-01-01 13:30:00",
            "2026-01-01 14:10:00",
            "2026-01-01 15:25:00",
            "2026-01-01 16:18:00",
            "2026-01-01 17:12:00",
            "2026-01-01 18:40:00",
            "2026-01-01 19:05:00",
            "2026-01-01 20:22:00",
            "2026-01-01 21:16:00",
        ],
        "passenger_count": [1, 2, 1, 3, 1, 2, 1, 1, 4, 1, 2, 1],
        "trip_distance": [2.1, 4.2, 1.0, 6.5, 1.8, 5.1, 3.3, 2.7, 8.4, 1.2, 4.7, 3.9],
        "fare_amount": [12.5, 21.0, 8.0, 33.5, 10.5, 26.0, 18.5, 15.0, 42.0, 7.5, 24.0, 19.5],
    }

    return pd.DataFrame(data)