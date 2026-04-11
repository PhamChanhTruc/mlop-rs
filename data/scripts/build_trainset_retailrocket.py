from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_repo.feature_definitions import (
    LABEL_COLUMN,
    MODEL_FEATURES,
    build_labeled_candidate_frame,
    downsample_train_negatives,
    split_candidate_frame,
)

IN_PATH = Path("data/processed/events_retailrocket.parquet")
OUT_DIR = Path("data/processed/dataset_retailrocket")
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON_HOURS = 24
NEG_RATIO_TRAIN = 10


def main() -> None:
    events = pd.read_parquet(IN_PATH)
    candidates = build_labeled_candidate_frame(events, horizon_hours=HORIZON_HOURS)
    train, val, test = split_candidate_frame(candidates)
    train = downsample_train_negatives(train, neg_ratio=NEG_RATIO_TRAIN, random_state=42)

    cols_out = ["user_id", "item_id", "event_ts", "event_type", LABEL_COLUMN] + MODEL_FEATURES
    train[cols_out].to_parquet(OUT_DIR / "train.parquet", index=False)
    val[cols_out].to_parquet(OUT_DIR / "val.parquet", index=False)
    test[cols_out].to_parquet(OUT_DIR / "test.parquet", index=False)

    def stats(name: str, frame: pd.DataFrame) -> str:
        return f"{name}: rows={len(frame):,}, pos_rate={frame[LABEL_COLUMN].mean():.4f}"

    print(stats("train", train))
    print(stats("val", val))
    print(stats("test", test))
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
