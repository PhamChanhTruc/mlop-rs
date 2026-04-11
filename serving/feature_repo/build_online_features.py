from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data"


def resolve_input_path() -> Path:
    candidates = [
        REPO_DIR.parent.parent / "data/processed/events_retailrocket.parquet",
        REPO_DIR.parent / "data/processed/events_retailrocket.parquet",
        Path("/app/data/processed/events_retailrocket.parquet"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_user_stats(events: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        events.groupby("user_id", as_index=False)
        .agg(
            user_event_count_prev=("event_ts", "size"),
            user_last_event_ts=("event_ts", "max"),
        )
        .sort_values("user_id")
        .reset_index(drop=True)
    )
    grouped["event_timestamp"] = grouped["user_last_event_ts"]
    return grouped


def build_item_stats(events: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        events.groupby("item_id", as_index=False)
        .agg(
            item_event_count_prev=("event_ts", "size"),
            item_last_event_ts=("event_ts", "max"),
        )
        .sort_values("item_id")
        .reset_index(drop=True)
    )
    grouped["event_timestamp"] = grouped["item_last_event_ts"]
    return grouped[["item_id", "item_event_count_prev", "event_timestamp"]]


def build_user_item_stats(events: pd.DataFrame) -> pd.DataFrame:
    enriched = pd.DataFrame(
        {
            "user_item_key": events["user_id"].astype("int64").astype(str)
            + ":"
            + events["item_id"].astype("int64").astype(str),
            "event_ts": events["event_ts"],
        }
    )
    grouped = (
        enriched.groupby("user_item_key", as_index=False)
        .agg(
            user_item_event_count_prev=("event_ts", "size"),
            user_item_last_event_ts=("event_ts", "max"),
        )
        .sort_values("user_item_key")
        .reset_index(drop=True)
    )
    grouped["event_timestamp"] = grouped["user_item_last_event_ts"]
    return grouped


def main() -> None:
    input_path = resolve_input_path()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run data/scripts/preprocess_retailrocket.py first."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(input_path, columns=["user_id", "item_id", "event_ts"])
    events["event_ts"] = pd.to_datetime(events["event_ts"], utc=True, errors="coerce")
    events = events.dropna(subset=["user_id", "item_id", "event_ts"]).copy()

    user_stats = build_user_stats(events)
    item_stats = build_item_stats(events)
    user_item_stats = build_user_item_stats(events)

    user_stats.to_parquet(DATA_DIR / "user_stats.parquet", index=False)
    item_stats.to_parquet(DATA_DIR / "item_stats.parquet", index=False)
    user_item_stats.to_parquet(DATA_DIR / "user_item_stats.parquet", index=False)

    print(f"Saved {DATA_DIR / 'user_stats.parquet'} rows={len(user_stats):,}")
    print(f"Saved {DATA_DIR / 'item_stats.parquet'} rows={len(item_stats):,}")
    print(f"Saved {DATA_DIR / 'user_item_stats.parquet'} rows={len(user_item_stats):,}")


if __name__ == "__main__":
    main()
