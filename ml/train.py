import argparse
import json
from datetime import datetime
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
from ml.model_gate import (
    passes_quality_gate,
    is_better_than_production,
    calculate_improvement,
)


MODEL_DIR = Path("ml/models")
METRICS_DIR = Path("ml/metrics")

# 기존 호환용 모델 파일
MODEL_PATH = MODEL_DIR / "fare_model.pkl"

# 실제 서비스에서 사용할 production 모델
PRODUCTION_MODEL_PATH = MODEL_DIR / "production_model.pkl"

METRICS_PATH = METRICS_DIR / "metrics.json"

MODEL_HISTORY_DIR = MODEL_DIR / "history"
CANDIDATE_MODEL_DIR = MODEL_DIR / "candidates"
METRICS_HISTORY_DIR = METRICS_DIR / "history"

REGISTRY_DIR = Path("ml/registry")
REGISTRY_PATH = REGISTRY_DIR / "model_registry.json"


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
    }


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {
            "production": None,
            "previous_production": None,
            "history": [],
            "promotion_events": [],
        }

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: dict):
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def find_best_result(results: list[dict], best_model_name: str) -> dict:
    for result in results:
        if result["model"] == best_model_name:
            return result

    raise ValueError("best_model에 해당하는 평가 결과를 찾을 수 없습니다.")


def update_registry(
    metrics: dict,
    best_result: dict,
    promoted: bool,
    promotion_reason: str,
    quality_gate_passed: bool,
    quality_gate_reasons: list[str],
):
    registry = load_registry()

    previous_production = registry.get("production")
    now = datetime.now().isoformat(timespec="seconds")

    candidate_record = {
        "run_id": metrics["run_id"],
        "model": metrics["best_model"],
        "mae": best_result["mae"],
        "rmse": best_result["rmse"],
        "r2": best_result["r2"],
        "candidate_model_path": metrics["candidate_model_path"],
        "quality_gate_passed": quality_gate_passed,
        "quality_gate_reasons": quality_gate_reasons,
        "promotion_reason": promotion_reason,
        "created_at": now,
    }

    # 이번 학습 회차의 모든 모델 이력 저장
    for result in metrics["results"]:
        if result["model"] == metrics["best_model"]:
            status = "promoted" if promoted else "rejected"
        else:
            status = "candidate"

        history_item = {
            "run_id": metrics["run_id"],
            "model": result["model"],
            "mae": result["mae"],
            "rmse": result["rmse"],
            "r2": result["r2"],
            "status": status,
            "created_at": now,
        }

        registry["history"].append(history_item)

    promotion_event = {
        **candidate_record,
        "promoted": promoted,
        "event_at": now,
    }

    registry["promotion_events"].append(promotion_event)

    if promoted:
        current_production = {
            "run_id": metrics["run_id"],
            "model": metrics["best_model"],
            "mae": best_result["mae"],
            "rmse": best_result["rmse"],
            "r2": best_result["r2"],
            "model_path": str(PRODUCTION_MODEL_PATH),
            "candidate_model_path": metrics["candidate_model_path"],
            "promoted_at": now,
        }

        improvement = calculate_improvement(previous_production, current_production)
        current_production.update(improvement)

        registry["previous_production"] = previous_production
        registry["production"] = current_production
    else:
        # 승격 실패 시 production은 유지
        registry["previous_production"] = registry.get("previous_production")
        registry["production"] = previous_production

    save_registry(registry)


def train(sample: bool = False):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    if best_model is None or best_model_name is None:
        raise RuntimeError("최적 후보 모델을 선택하지 못했습니다.")

    best_result = find_best_result(results, best_model_name)

    candidate_model_path = CANDIDATE_MODEL_DIR / f"{run_id}_{best_model_name}.pkl"
    history_model_path = MODEL_HISTORY_DIR / f"{run_id}_{best_model_name}.pkl"

    # 후보 모델 저장
    joblib.dump(best_model, candidate_model_path)

    # history에도 동일 모델 보관
    joblib.dump(best_model, history_model_path)

    # 기존 호환용 fare_model.pkl에도 일단 최신 후보 저장
    # 단, 실제 API는 production_model.pkl을 사용하도록 바꿀 예정
    joblib.dump(best_model, MODEL_PATH)

    registry = load_registry()
    current_production = registry.get("production")

    quality_gate_passed, quality_gate_reasons = passes_quality_gate(best_result)

    better_than_production, promotion_reason = is_better_than_production(
        best_result,
        current_production,
    )

    promoted = quality_gate_passed and better_than_production

    if promoted:
        joblib.dump(best_model, PRODUCTION_MODEL_PATH)
        print("신규 모델이 Production 모델로 승격되었습니다.")
    else:
        print("신규 모델이 Production 모델로 승격되지 않았습니다.")
        print("사유:", promotion_reason)
        if quality_gate_reasons:
            print("Quality Gate 실패 사유:", quality_gate_reasons)

        if not PRODUCTION_MODEL_PATH.exists():
            print("기존 Production 모델이 없어, 초기 모델 보호를 위해 현재 후보를 Production으로 저장합니다.")
            joblib.dump(best_model, PRODUCTION_MODEL_PATH)
            promoted = True
            promotion_reason = "기존 Production 모델이 없어 초기 Production 모델로 등록했습니다."

    metrics = {
        "run_id": run_id,
        "best_model": best_model_name,
        "best_metric": "mae",
        "promotion": {
            "promoted": promoted,
            "quality_gate_passed": quality_gate_passed,
            "quality_gate_reasons": quality_gate_reasons,
            "better_than_production": better_than_production,
            "promotion_reason": promotion_reason,
        },
        "production_model_path": str(PRODUCTION_MODEL_PATH),
        "candidate_model_path": str(candidate_model_path),
        "history_model_path": str(history_model_path),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "total_rows": int(len(X)),
        "results": results,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    history_metrics_path = METRICS_HISTORY_DIR / f"metrics_{run_id}.json"

    with open(history_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    update_registry(
        metrics=metrics,
        best_result=best_result,
        promoted=promoted,
        promotion_reason=promotion_reason,
        quality_gate_passed=quality_gate_passed,
        quality_gate_reasons=quality_gate_reasons,
    )

    print("=" * 60)
    print("후보 모델 저장 완료:", candidate_model_path)
    print("Production 모델 경로:", PRODUCTION_MODEL_PATH)
    print("성능 지표 저장 완료:", METRICS_PATH)
    print("성능 지표 이력 저장 완료:", history_metrics_path)
    print("모델 레지스트리 저장 완료:", REGISTRY_PATH)
    print("최종 후보 모델:", best_model_name)
    print("Production 승격 여부:", promoted)
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