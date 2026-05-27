MAX_MAE = 5.0
MAX_RMSE = 8.0
MIN_R2 = 0.70

IMPROVEMENT_THRESHOLD = 0.03


def passes_quality_gate(metrics: dict) -> tuple[bool, list[str]]:
    """
    모델이 최소 품질 기준을 통과하는지 검사한다.

    기준:
    - MAE는 낮을수록 좋음
    - RMSE는 낮을수록 좋음
    - R2는 높을수록 좋음

    반환:
    - 통과 여부
    - 실패 사유 목록
    """
    reasons = []

    mae = metrics.get("mae")
    rmse = metrics.get("rmse")
    r2 = metrics.get("r2")

    if mae is None:
        reasons.append("MAE 값이 없습니다.")
    elif mae > MAX_MAE:
        reasons.append(f"MAE 기준 초과: {mae} > {MAX_MAE}")

    if rmse is None:
        reasons.append("RMSE 값이 없습니다.")
    elif rmse > MAX_RMSE:
        reasons.append(f"RMSE 기준 초과: {rmse} > {MAX_RMSE}")

    if r2 is None:
        reasons.append("R2 값이 없습니다.")
    elif r2 < MIN_R2:
        reasons.append(f"R2 기준 미달: {r2} < {MIN_R2}")

    return len(reasons) == 0, reasons


def is_better_than_production(
    candidate_metrics: dict,
    production_metrics: dict | None,
) -> tuple[bool, str]:
    """
    신규 후보 모델이 기존 Production 모델보다 충분히 개선되었는지 검사한다.

    Production 모델이 없는 경우:
    - 품질 기준만 통과하면 승격 가능

    Production 모델이 있는 경우:
    - MAE 기준으로 일정 비율 이상 개선되어야 승격
    """
    if production_metrics is None:
        return True, "기존 Production 모델이 없어 신규 모델을 승격할 수 있습니다."

    candidate_mae = candidate_metrics.get("mae")
    production_mae = production_metrics.get("mae")

    if candidate_mae is None:
        return False, "후보 모델의 MAE 값이 없습니다."

    if production_mae is None:
        return True, "기존 Production 모델의 MAE 값이 없어 신규 모델을 승격합니다."

    required_mae = production_mae * (1 - IMPROVEMENT_THRESHOLD)

    if candidate_mae <= required_mae:
        improvement = ((production_mae - candidate_mae) / production_mae) * 100
        return True, f"MAE가 기존 대비 {improvement:.2f}% 개선되었습니다."

    improvement = ((production_mae - candidate_mae) / production_mae) * 100

    return (
        False,
        f"MAE 개선율이 부족합니다. 현재 개선율: {improvement:.2f}%, "
        f"필요 개선율: {IMPROVEMENT_THRESHOLD * 100:.2f}%",
    )


def calculate_improvement(previous: dict | None, current: dict) -> dict:
    """
    이전 Production 모델 대비 현재 모델의 개선율을 계산한다.
    """
    if not previous:
        return {
            "mae_improvement_percent": None,
            "rmse_improvement_percent": None,
            "r2_improvement_percent": None,
        }

    prev_mae = previous.get("mae")
    curr_mae = current.get("mae")

    prev_rmse = previous.get("rmse")
    curr_rmse = current.get("rmse")

    prev_r2 = previous.get("r2")
    curr_r2 = current.get("r2")

    mae_improvement = None
    rmse_improvement = None
    r2_improvement = None

    if prev_mae not in (None, 0) and curr_mae is not None:
        mae_improvement = ((prev_mae - curr_mae) / prev_mae) * 100

    if prev_rmse not in (None, 0) and curr_rmse is not None:
        rmse_improvement = ((prev_rmse - curr_rmse) / prev_rmse) * 100

    if prev_r2 not in (None, 0) and curr_r2 is not None:
        r2_improvement = ((curr_r2 - prev_r2) / abs(prev_r2)) * 100

    return {
        "mae_improvement_percent": round(mae_improvement, 4)
        if mae_improvement is not None
        else None,
        "rmse_improvement_percent": round(rmse_improvement, 4)
        if rmse_improvement is not None
        else None,
        "r2_improvement_percent": round(r2_improvement, 4)
        if r2_improvement is not None
        else None,
    }