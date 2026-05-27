import json
import sys
from pathlib import Path


METRICS_PATH = Path("ml/metrics/metrics.json")

MAX_MAE = 5.0
MAX_RMSE = 8.0
MIN_R2 = 0.70


def main():
    if not METRICS_PATH.exists():
        print(f"metrics 파일이 없습니다: {METRICS_PATH}")
        sys.exit(1)

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    best_model = metrics.get("best_model")
    results = metrics.get("results", [])

    if not best_model or not results:
        print("best_model 또는 results 정보가 없습니다.")
        sys.exit(1)

    best_result = None

    for result in results:
        if result.get("model") == best_model:
            best_result = result
            break

    if best_result is None:
        print(f"best_model 결과를 찾을 수 없습니다: {best_model}")
        sys.exit(1)

    mae = best_result.get("mae")
    rmse = best_result.get("rmse")
    r2 = best_result.get("r2")

    failed_reasons = []

    if mae is None or mae > MAX_MAE:
        failed_reasons.append(f"MAE 기준 실패: {mae} > {MAX_MAE}")

    if rmse is None or rmse > MAX_RMSE:
        failed_reasons.append(f"RMSE 기준 실패: {rmse} > {MAX_RMSE}")

    if r2 is None or r2 < MIN_R2:
        failed_reasons.append(f"R2 기준 실패: {r2} < {MIN_R2}")

    print("Best Model:", best_model)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    if failed_reasons:
        print("모델 성능 기준을 통과하지 못했습니다.")
        for reason in failed_reasons:
            print("-", reason)
        sys.exit(1)

    print("모델 성능 기준을 통과했습니다.")


if __name__ == "__main__":
    main()