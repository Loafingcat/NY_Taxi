import pandas as pd

from ml.features import create_features, split_features_target


def test_create_features():
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": ["2026-01-01 10:00:00"],
            "tpep_dropoff_datetime": ["2026-01-01 10:20:00"],
            "passenger_count": [1.0],
            "trip_distance": [3.5],
            "fare_amount": [18.0],
        }
    )

    result = create_features(df)

    assert "pickup_hour" in result.columns
    assert "pickup_dayofweek" in result.columns
    assert "pickup_month" in result.columns
    assert "duration_minutes" in result.columns
    assert result.iloc[0]["duration_minutes"] == 20


def test_split_features_target():
    df = pd.DataFrame(
        {
            "passenger_count": [1.0],
            "trip_distance": [3.5],
            "pickup_hour": [10],
            "pickup_dayofweek": [3],
            "pickup_month": [1],
            "duration_minutes": [20],
            "fare_amount": [18.0],
        }
    )

    X, y = split_features_target(df)

    assert len(X) == 1
    assert len(y) == 1
    assert "fare_amount" not in X.columns