import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


DATA_PATH = Path("output/parquet")
STATE_PATH = Path("ml/registry/train_trigger_state.json")

MIN_ROWS_FOR_TRAINING = 50000
MIN_NEW_ROWS_FOR_RETRAINING = 10000


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {
            "last_trained_rows": 0,
        }

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def main():
    if not DATA_PATH.exists():
        print(f"학습 데이터 경로가 없습니다: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_parquet(DATA_PATH)
    current_rows = len(df)

    print(f"현재 누적 학습 데이터 수: {current_rows}")

    state = load_state()
    last_trained_rows = int(state.get("last_trained_rows", 0))
    new_rows = current_rows - last_trained_rows

    print(f"마지막 학습 시점 데이터 수: {last_trained_rows}")
    print(f"새로 누적된 데이터 수: {new_rows}")

    if current_rows < MIN_ROWS_FOR_TRAINING:
        print(
            f"아직 최소 학습 기준에 도달하지 못했습니다. "
            f"현재={current_rows}, 필요={MIN_ROWS_FOR_TRAINING}"
        )
        return

    if new_rows < MIN_NEW_ROWS_FOR_RETRAINING:
        print(
            f"재학습 기준에 도달하지 못했습니다. "
            f"신규={new_rows}, 필요={MIN_NEW_ROWS_FOR_RETRAINING}"
        )
        return

    print("재학습 기준을 만족했습니다. 실제 데이터로 모델 학습을 시작합니다.")

    result = subprocess.run(
        [sys.executable, "-m", "ml.train"],
        check=False,
    )

    if result.returncode != 0:
        print("모델 학습 실패")
        sys.exit(result.returncode)

    save_state(
        {
            "last_trained_rows": current_rows,
        }
    )

    print("모델 학습 완료 및 train trigger state 갱신 완료")


if __name__ == "__main__":
    main()