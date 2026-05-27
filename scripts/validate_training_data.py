import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features import create_sample_training_data, create_features


REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "fare_amount",
]


def main():
    df = create_sample_training_data()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        print(f"필수 컬럼 누락: {missing_columns}")
        sys.exit(1)

    if df.empty:
        print("데이터가 비어 있습니다.")
        sys.exit(1)

    try:
        feature_df = create_features(df)
    except Exception as e:
        print(f"Feature 생성 실패: {e}")
        sys.exit(1)

    if feature_df.empty:
        print("Feature 생성 후 데이터가 비어 있습니다.")
        sys.exit(1)

    if (feature_df["fare_amount"] < 0).any():
        print("fare_amount에 음수 값이 있습니다.")
        sys.exit(1)

    if (feature_df["trip_distance"] <= 0).any():
        print("trip_distance에 0 이하 값이 있습니다.")
        sys.exit(1)

    if (feature_df["passenger_count"] <= 0).any():
        print("passenger_count에 0 이하 값이 있습니다.")
        sys.exit(1)

    print("데이터 검증을 통과했습니다.")
    print(f"검증 데이터 수: {len(feature_df)}")


if __name__ == "__main__":
    main()