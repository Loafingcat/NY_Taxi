import os
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


API_URL = os.getenv("API_URL", "http://localhost:8000")
METRICS_PATH = Path("ml/metrics/metrics.json")
REGISTRY_PATH = Path("ml/registry/model_registry.json")


st.set_page_config(
    page_title="NY Taxi MLOps Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --card-border: rgba(128, 128, 128, 0.25);
        --soft-bg: rgba(128, 128, 128, 0.08);
        --soft-bg-strong: rgba(128, 128, 128, 0.14);
        --success-bg: rgba(46, 160, 67, 0.12);
        --success-border: rgba(46, 160, 67, 0.55);
        --warning-bg: rgba(245, 158, 11, 0.12);
        --warning-border: rgba(245, 158, 11, 0.55);
    }

    .metric-card {
        padding: 1.1rem 1.2rem;
        border-radius: 18px;
        border: 1px solid var(--card-border);
        background: var(--soft-bg);
        margin-bottom: 1rem;
    }

    .best-model-card {
        padding: 1.25rem 1.35rem;
        border-radius: 20px;
        border: 1.5px solid var(--success-border);
        background: linear-gradient(
            135deg,
            rgba(46, 160, 67, 0.18),
            rgba(128, 128, 128, 0.06)
        );
        margin: 1rem 0 1.25rem 0;
    }

    .best-model-title {
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .best-model-subtitle {
        font-size: 0.95rem;
        opacity: 0.82;
        line-height: 1.5;
    }

    .section-caption {
        opacity: 0.75;
        font-size: 0.92rem;
        margin-bottom: 0.75rem;
    }

    .small-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        border: 1px solid var(--card-border);
        background: var(--soft-bg-strong);
        font-size: 0.8rem;
        margin-right: 0.35rem;
    }

    .best-badge {
        border: 1px solid var(--success-border);
        background: var(--success-bg);
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        border: 1px solid var(--card-border);
        background: var(--soft-bg);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NY Taxi Real-Time Streaming & MLOps Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(
    ["모델 예측", "모델 성능 비교", "모델 개선 이력", "프로젝트 구조"]
)

template_name = (
    "plotly_dark"
    if st.get_option("theme.base") == "dark"
    else "plotly_white"
)


with tab1:
    st.header("택시 요금 예측")

    st.caption(
        "이동 거리는 사용자가 보기 편하도록 km로 입력받고, "
        "모델에는 NY Taxi 데이터 기준에 맞춰 mile로 변환하여 전달합니다."
    )

    col1, col2 = st.columns(2)

    with col1:
        passenger_count = st.number_input(
            "승객 수",
            min_value=1,
            max_value=8,
            value=1,
            step=1,
            help="승객 수는 정수만 입력할 수 있습니다.",
        )

        trip_distance_km = st.number_input(
            "이동 거리(km)",
            min_value=0.1,
            value=5.0,
            step=0.1,
            help="화면에서는 km로 입력받고, 모델 입력 시 mile로 변환합니다.",
        )

        duration_minutes = st.number_input(
            "운행 시간(분)",
            min_value=1.0,
            value=18.0,
            step=1.0,
        )

    with col2:
        pickup_hour = st.number_input(
            "탑승 시간대",
            min_value=0,
            max_value=23,
            value=14,
            step=1,
        )

        pickup_dayofweek = st.number_input(
            "요일(월=0, 일=6)",
            min_value=0,
            max_value=6,
            value=2,
            step=1,
        )

        pickup_month = st.number_input(
            "월",
            min_value=1,
            max_value=12,
            value=1,
            step=1,
        )

    trip_distance_mile = trip_distance_km / 1.609344

    st.info(
        f"입력한 이동 거리: {trip_distance_km:.2f} km "
        f"→ 모델 입력값: {trip_distance_mile:.2f} mile"
    )

    if st.button("예상 요금 예측"):
        payload = {
            "passenger_count": int(passenger_count),
            "trip_distance": float(trip_distance_mile),
            "pickup_hour": int(pickup_hour),
            "pickup_dayofweek": int(pickup_dayofweek),
            "pickup_month": int(pickup_month),
            "duration_minutes": float(duration_minutes),
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            st.success(f"예상 요금: ${result['predicted_fare']}")
        except Exception as e:
            st.error(f"API 호출 실패: {e}")


with tab2:
    st.header("모델 성능 비교")

    st.markdown(
        """
        <div class="section-caption">
            학습된 모델들의 MAE, RMSE, R2를 비교하고, 현재 서비스에 사용하기 가장 적합한 모델을 강조 표시합니다.
            MAE와 RMSE는 낮을수록 좋고, R2는 높을수록 좋습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not METRICS_PATH.exists():
        st.warning("metrics.json 파일이 없습니다. 먼저 모델 학습을 실행하세요.")
    else:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        best_model = metrics.get("best_model")
        best_metric = metrics.get("best_metric", "mae")
        results = metrics.get("results", [])

        if not results:
            st.warning("모델 성능 결과가 비어 있습니다.")
        else:
            result_df = pd.DataFrame(results)

            for col in ["mae", "rmse", "r2"]:
                if col in result_df.columns:
                    result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

            result_df["status"] = result_df["model"].apply(
                lambda x: "Best" if x == best_model else "Candidate"
            )

            sorted_df = result_df.sort_values("mae", ascending=True).reset_index(drop=True)
            best_row_df = sorted_df[sorted_df["model"] == best_model]

            if best_row_df.empty:
                st.warning("best_model 값과 results의 model명이 일치하지 않습니다.")
            else:
                best_row = best_row_df.iloc[0]

                st.markdown(
                    f"""
                    <div class="best-model-card">
                        <div class="best-model-title">🏆 Best Model: {best_model}</div>
                        <div class="best-model-subtitle">
                            선택 기준: <b>{best_metric.upper()}</b><br>
                            현재 학습 결과에서 가장 좋은 성능을 보여 서비스 후보 모델로 선택되었습니다.
                        </div>
                        <div style="margin-top: 0.85rem;">
                            <span class="small-badge best-badge">Production Candidate</span>
                            <span class="small-badge">MAE {best_row["mae"]:.4f}</span>
                            <span class="small-badge">RMSE {best_row["rmse"]:.4f}</span>
                            <span class="small-badge">R2 {best_row["r2"]:.4f}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    label="Best MAE",
                    value=f"{best_row['mae']:.4f}",
                    help="평균 절대 오차입니다. 낮을수록 좋습니다.",
                )
                col2.metric(
                    label="Best RMSE",
                    value=f"{best_row['rmse']:.4f}",
                    help="큰 오차에 더 민감한 지표입니다. 낮을수록 좋습니다.",
                )
                col3.metric(
                    label="Best R2",
                    value=f"{best_row['r2']:.4f}",
                    help="모델 설명력입니다. 1에 가까울수록 좋습니다.",
                )

            st.divider()

            st.subheader("모델별 성능 비교표")

            display_df = sorted_df[["status", "model", "mae", "rmse", "r2"]].copy()

            def highlight_best(row):
                if row["model"] == best_model:
                    return [
                        "background-color: rgba(46, 160, 67, 0.18); font-weight: 700;"
                        for _ in row
                    ]
                return ["" for _ in row]

            st.dataframe(
                display_df.style.apply(highlight_best, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            st.subheader("성능 지표 시각화")

            chart_df = sorted_df.copy()
            chart_df["is_best"] = chart_df["model"].apply(
                lambda x: "Best Model" if x == best_model else "Candidate"
            )

            col1, col2 = st.columns(2)

            with col1:
                fig_mae = px.bar(
                    chart_df,
                    x="model",
                    y="mae",
                    color="is_best",
                    title="MAE 비교 낮을수록 좋음",
                    text="mae",
                    template=template_name,
                )
                fig_mae.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig_mae.update_layout(
                    xaxis_title="Model",
                    yaxis_title="MAE",
                    legend_title="Status",
                    margin=dict(l=20, r=20, t=55, b=20),
                )
                st.plotly_chart(fig_mae, use_container_width=True)

            with col2:
                fig_rmse = px.bar(
                    chart_df,
                    x="model",
                    y="rmse",
                    color="is_best",
                    title="RMSE 비교 낮을수록 좋음",
                    text="rmse",
                    template=template_name,
                )
                fig_rmse.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig_rmse.update_layout(
                    xaxis_title="Model",
                    yaxis_title="RMSE",
                    legend_title="Status",
                    margin=dict(l=20, r=20, t=55, b=20),
                )
                st.plotly_chart(fig_rmse, use_container_width=True)

            fig_r2 = px.bar(
                chart_df,
                x="model",
                y="r2",
                color="is_best",
                title="R2 비교 높을수록 좋음",
                text="r2",
                template=template_name,
            )
            fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_r2.update_layout(
                xaxis_title="Model",
                yaxis_title="R2",
                legend_title="Status",
                margin=dict(l=20, r=20, t=55, b=20),
            )
            st.plotly_chart(fig_r2, use_container_width=True)

            st.divider()

            st.subheader("종합 성능 레이더 차트")

            radar_df = sorted_df.copy()
            mae_max = radar_df["mae"].max()
            rmse_max = radar_df["rmse"].max()

            radar_df["mae_score"] = (
                1 - (radar_df["mae"] / mae_max) if mae_max else 0
            )
            radar_df["rmse_score"] = (
                1 - (radar_df["rmse"] / rmse_max) if rmse_max else 0
            )
            radar_df["r2_score"] = radar_df["r2"]

            fig_radar = go.Figure()

            for _, row in radar_df.iterrows():
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=[
                            row["mae_score"],
                            row["rmse_score"],
                            row["r2_score"],
                        ],
                        theta=[
                            "MAE Score",
                            "RMSE Score",
                            "R2 Score",
                        ],
                        fill="toself",
                        name=row["model"],
                    )
                )

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                    )
                ),
                showlegend=True,
                title="모델 종합 성능 비교 1에 가까울수록 좋음",
                template=template_name,
                margin=dict(l=20, r=20, t=60, b=20),
            )

            st.plotly_chart(fig_radar, use_container_width=True)

            st.divider()

            st.subheader("학습 데이터 정보")

            col1, col2, col3 = st.columns(3)
            col1.metric("전체 데이터 수", metrics.get("total_rows", "-"))
            col2.metric("학습 데이터 수", metrics.get("train_rows", "-"))
            col3.metric("테스트 데이터 수", metrics.get("test_rows", "-"))

            with st.expander("metrics.json 원본 보기"):
                st.json(metrics)


with tab3:
    st.header("모델 개선 이력")

    st.markdown(
        """
        <div class="section-caption">
            모델 재학습, Production 모델 상태, 이전 모델 대비 개선율, 역대 성능 상위 모델을 확인합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================
    # 1. Action Panel
    # =========================
    st.subheader("모델 재학습")

    action_col1, action_col2, action_col3 = st.columns([1, 1, 1])

    with action_col1:
        run_sample_train = st.button(
            "샘플 데이터로 재학습",
            key="tab3_sample_train_button",
            use_container_width=True,
            help="CI/CD 또는 빠른 테스트용 내장 샘플 데이터로 모델을 학습합니다.",
        )

    with action_col2:
        run_real_train = st.button(
            "정제 데이터로 재학습",
            key="tab3_real_train_button",
            use_container_width=True,
            help="output/parquet에 저장된 실제 정제 데이터를 사용해 모델을 학습합니다.",
        )

    with action_col3:
        refresh_model_info = st.button(
            "모델 정보 조회",
            key="tab3_refresh_model_info_button",
            use_container_width=True,
            help="FastAPI의 /model-info 엔드포인트를 호출합니다.",
        )

    if run_sample_train or run_real_train:
        sample_mode = run_sample_train

        with st.spinner("모델 재학습을 실행 중입니다."):
            try:
                response = requests.post(
                    f"{API_URL}/train",
                    json={"sample": sample_mode},
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()

                st.success("모델 재학습이 완료되었습니다.")

                promotion = result.get("latest_metrics", {}).get("promotion", {})

                if promotion:
                    promoted = promotion.get("promoted")
                    reason = promotion.get("promotion_reason")

                    if promoted:
                        st.info(f"Production 모델 승격 완료: {reason}")
                    else:
                        st.warning(f"Production 모델 승격 안 됨: {reason}")

                st.caption("최신 결과를 보려면 화면을 새로고침하거나 모델 정보 조회를 눌러 확인하세요.")

            except Exception as e:
                st.error(f"모델 재학습 요청 실패: {e}")

    if refresh_model_info:
        try:
            response = requests.get(f"{API_URL}/model-info", timeout=10)
            response.raise_for_status()
            model_info = response.json()

            st.success("모델 정보를 성공적으로 조회했습니다.")

            with st.expander("조회된 모델 정보 보기", expanded=False):
                st.json(model_info)

        except Exception as e:
            st.error(f"모델 정보 조회 실패: {e}")

    st.divider()

    # =========================
    # 2. Registry Load
    # =========================
    if not REGISTRY_PATH.exists():
        st.info("아직 model_registry.json 파일이 없습니다. 모델 학습을 먼저 실행하세요.")
    else:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)

        production = registry.get("production")
        previous = registry.get("previous_production")
        history = registry.get("history", [])

        # =========================
        # 3. Production Model Summary
        # =========================
        st.subheader("현재 Production 모델")

        if production:
            st.markdown(
                f"""
                <div class="best-model-card">
                    <div class="best-model-title">🚀 {production.get("model", "-")}</div>
                    <div class="best-model-subtitle">
                        현재 FastAPI 예측에 사용되는 Production 모델입니다.<br>
                        Run ID: <b>{production.get("run_id", "-")}</b><br>
                        Promoted At: <b>{production.get("promoted_at", "-")}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            prod_col1, prod_col2, prod_col3 = st.columns(3)

            prod_col1.metric(
                "MAE",
                production.get("mae", "-"),
                help="평균 절대 오차입니다. 낮을수록 좋습니다.",
            )
            prod_col2.metric(
                "RMSE",
                production.get("rmse", "-"),
                help="큰 오차에 더 민감한 지표입니다. 낮을수록 좋습니다.",
            )
            prod_col3.metric(
                "R2",
                production.get("r2", "-"),
                help="모델 설명력입니다. 높을수록 좋습니다.",
            )

        else:
            st.warning("현재 Production 모델 정보가 없습니다.")

        st.divider()

        # =========================
        # 4. Improvement Summary
        # =========================
        st.subheader("이전 Production 모델 대비 개선율")

        if production and previous:
            prev_mae = previous.get("mae")
            curr_mae = production.get("mae")

            prev_rmse = previous.get("rmse")
            curr_rmse = production.get("rmse")

            prev_r2 = previous.get("r2")
            curr_r2 = production.get("r2")

            mae_improvement = None
            rmse_improvement = None
            r2_improvement = None

            if prev_mae not in (None, 0) and curr_mae is not None:
                mae_improvement = ((prev_mae - curr_mae) / prev_mae) * 100

            if prev_rmse not in (None, 0) and curr_rmse is not None:
                rmse_improvement = ((prev_rmse - curr_rmse) / prev_rmse) * 100

            if prev_r2 not in (None, 0) and curr_r2 is not None:
                r2_improvement = ((curr_r2 - prev_r2) / abs(prev_r2)) * 100

            imp_col1, imp_col2, imp_col3 = st.columns(3)

            imp_col1.metric(
                "MAE 개선율",
                f"{mae_improvement:.2f}%" if mae_improvement is not None else "-",
                delta=f"{mae_improvement:.2f}%" if mae_improvement is not None else None,
                help="MAE는 낮을수록 좋습니다. 양수면 평균 오차가 줄었다는 의미입니다.",
            )

            imp_col2.metric(
                "RMSE 개선율",
                f"{rmse_improvement:.2f}%" if rmse_improvement is not None else "-",
                delta=f"{rmse_improvement:.2f}%" if rmse_improvement is not None else None,
                help="RMSE는 낮을수록 좋습니다. 양수면 큰 오차가 줄었다는 의미입니다.",
            )

            imp_col3.metric(
                "R2 변화율",
                f"{r2_improvement:.2f}%" if r2_improvement is not None else "-",
                delta=f"{r2_improvement:.2f}%" if r2_improvement is not None else None,
                help="R2는 높을수록 좋습니다. 양수면 설명력이 좋아졌다는 의미입니다.",
            )

            improvement_df = pd.DataFrame(
                [
                    {
                        "metric": "MAE",
                        "improvement_percent": mae_improvement,
                        "direction": "낮을수록 좋음",
                    },
                    {
                        "metric": "RMSE",
                        "improvement_percent": rmse_improvement,
                        "direction": "낮을수록 좋음",
                    },
                    {
                        "metric": "R2",
                        "improvement_percent": r2_improvement,
                        "direction": "높을수록 좋음",
                    },
                ]
            ).dropna(subset=["improvement_percent"])

            if not improvement_df.empty:
                fig_improvement = px.bar(
                    improvement_df,
                    x="metric",
                    y="improvement_percent",
                    color="metric",
                    text="improvement_percent",
                    title="이전 Production 모델 대비 개선율",
                    template=template_name,
                )

                fig_improvement.update_traces(
                    texttemplate="%{text:.2f}%",
                    textposition="outside",
                )

                fig_improvement.update_layout(
                    xaxis_title="Metric",
                    yaxis_title="Improvement (%)",
                    showlegend=False,
                    margin=dict(l=20, r=20, t=55, b=20),
                )

                st.plotly_chart(fig_improvement, use_container_width=True)

            st.caption(
                "MAE와 RMSE는 낮아질수록 좋기 때문에 이전 값에서 현재 값을 뺀 비율로 계산합니다. "
                "R2는 높아질수록 좋기 때문에 현재 값이 이전 값보다 얼마나 증가했는지 계산합니다."
            )

        elif production and not previous:
            st.info("이전 Production 모델이 아직 없습니다. 모델을 한 번 더 학습하면 개선율을 계산할 수 있습니다.")
        else:
            st.warning("개선율을 계산할 Production 정보가 부족합니다.")

        st.divider()

        # =========================
        # 5. History Charts
        # =========================
        if history:
            history_df = pd.DataFrame(history)

            for col in ["mae", "rmse", "r2"]:
                if col in history_df.columns:
                    history_df[col] = pd.to_numeric(history_df[col], errors="coerce")

            if "created_at" in history_df.columns:
                history_df["created_at"] = pd.to_datetime(
                    history_df["created_at"],
                    errors="coerce",
                )

            st.subheader("역대 모델 TOP 5")

            top_df = (
                history_df.sort_values("mae", ascending=True)
                .head(5)
                .copy()
            )

            top_df["label"] = (
                top_df["model"].astype(str)
                + " / "
                + top_df["run_id"].astype(str)
            )

            fig_top = px.bar(
                top_df.sort_values("mae", ascending=True),
                x="mae",
                y="label",
                color="status" if "status" in top_df.columns else None,
                orientation="h",
                text="mae",
                title="역대 TOP 5 모델 MAE 낮을수록 좋음",
                template=template_name,
            )

            fig_top.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside",
            )

            fig_top.update_layout(
                xaxis_title="MAE",
                yaxis_title="Model / Run ID",
                margin=dict(l=20, r=20, t=55, b=20),
            )

            st.plotly_chart(fig_top, use_container_width=True)

            show_columns = [
                col for col in [
                    "run_id",
                    "model",
                    "mae",
                    "rmse",
                    "r2",
                    "status",
                    "created_at",
                ]
                if col in top_df.columns
            ]

            st.dataframe(
                top_df[show_columns],
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            st.subheader("학습 이력 추세")

            if "created_at" in history_df.columns and history_df["created_at"].notna().any():
                trend_df = history_df.sort_values("created_at").copy()

                trend_col1, trend_col2 = st.columns(2)

                with trend_col1:
                    fig_mae_trend = px.line(
                        trend_df,
                        x="created_at",
                        y="mae",
                        color="model",
                        markers=True,
                        title="MAE 변화 추세 낮을수록 좋음",
                        template=template_name,
                    )
                    fig_mae_trend.update_layout(
                        xaxis_title="Created At",
                        yaxis_title="MAE",
                        margin=dict(l=20, r=20, t=55, b=20),
                    )
                    st.plotly_chart(fig_mae_trend, use_container_width=True)

                with trend_col2:
                    fig_r2_trend = px.line(
                        trend_df,
                        x="created_at",
                        y="r2",
                        color="model",
                        markers=True,
                        title="R2 변화 추세 높을수록 좋음",
                        template=template_name,
                    )
                    fig_r2_trend.update_layout(
                        xaxis_title="Created At",
                        yaxis_title="R2",
                        margin=dict(l=20, r=20, t=55, b=20),
                    )
                    st.plotly_chart(fig_r2_trend, use_container_width=True)

                fig_rmse_trend = px.line(
                    trend_df,
                    x="created_at",
                    y="rmse",
                    color="model",
                    markers=True,
                    title="RMSE 변화 추세 낮을수록 좋음",
                    template=template_name,
                )
                fig_rmse_trend.update_layout(
                    xaxis_title="Created At",
                    yaxis_title="RMSE",
                    margin=dict(l=20, r=20, t=55, b=20),
                )
                st.plotly_chart(fig_rmse_trend, use_container_width=True)
            else:
                st.info("created_at 정보가 없어 학습 이력 추세 그래프를 표시할 수 없습니다.")

            with st.expander("전체 모델 학습 이력 보기", expanded=False):
                if "created_at" in history_df.columns:
                    history_df = history_df.sort_values("created_at", ascending=False)

                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander("model_registry.json 원본 보기", expanded=False):
                st.json(registry)

        else:
            st.info("아직 모델 학습 이력이 없습니다.")


with tab4:
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