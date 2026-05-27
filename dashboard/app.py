import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


API_URL = "http://localhost:8000"
METRICS_PATH = Path("ml/metrics/metrics.json")


st.set_page_config(
    page_title="NY Taxi MLOps Dashboard",
    layout="wide",
)

st.title("NY Taxi Real-Time Streaming & MLOps Dashboard")

tab1, tab2, tab3 = st.tabs(["모델 예측", "모델 성능", "프로젝트 구조"])


with tab1:
    st.header("택시 요금 예측")

    col1, col2 = st.columns(2)

    with col1:
        passenger_count = st.number_input("승객 수", min_value=1.0, value=1.0)
        trip_distance = st.number_input("이동 거리", min_value=0.1, value=3.2)
        duration_minutes = st.number_input("운행 시간(분)", min_value=1.0, value=18.0)

    with col2:
        pickup_hour = st.number_input("탑승 시간대", min_value=0, max_value=23, value=14)
        pickup_dayofweek = st.number_input("요일(월=0, 일=6)", min_value=0, max_value=6, value=2)
        pickup_month = st.number_input("월", min_value=1, max_value=12, value=1)

    if st.button("예상 요금 예측"):
        payload = {
            "passenger_count": passenger_count,
            "trip_distance": trip_distance,
            "pickup_hour": pickup_hour,
            "pickup_dayofweek": pickup_dayofweek,
            "pickup_month": pickup_month,
            "duration_minutes": duration_minutes,
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            st.success(f"예상 요금: ${result['predicted_fare']}")
        except Exception as e:
            st.error(f"API 호출 실패: {e}")


with tab2:
    st.header("모델 성능 지표")

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        col1, col2, col3 = st.columns(3)

        col1.metric("MAE", metrics.get("mae"))
        col2.metric("RMSE", metrics.get("rmse"))
        col3.metric("R2", metrics.get("r2"))

        st.json(metrics)
    else:
        st.warning("metrics.json 파일이 없습니다. 먼저 모델 학습을 실행하세요.")


with tab3:
    st.header("프로젝트 파이프라인")

    st.code(
        """
NY Taxi Parquet Data
→ Kafka Producer
→ Kafka Topic
→ Spark Structured Streaming
→ Cleaned Parquet / PostgreSQL
→ ML Training Pipeline
→ FastAPI Prediction API
→ Streamlit Dashboard
        """,
        language="text",
    )