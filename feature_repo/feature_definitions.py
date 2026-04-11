from __future__ import annotations

import pandas as pd

MODEL_FEATURES = [
    "is_addtocart",
    "user_event_count_prev",
    "item_event_count_prev",
    "user_item_event_count_prev",
    "user_time_since_prev_event_sec",
    "item_time_since_prev_event_sec",
    "user_item_time_since_prev_event_sec",
]
LABEL_COLUMN = "label"
DEFAULT_HORIZON_HOURS = 24


def build_labeled_candidate_frame(
    events: pd.DataFrame,
    *,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> pd.DataFrame:
    df = events.copy()
    df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "event_ts"]).sort_values("event_ts").reset_index(drop=True)

    df["is_purchase"] = df["event_type"] == "transaction"
    df["purchase_ts"] = df["event_ts"].where(df["is_purchase"])
    df["next_purchase_ts"] = df.groupby(["user_id", "item_id"])["purchase_ts"].bfill()

    df["user_event_count_prev"] = df.groupby("user_id").cumcount()
    df["item_event_count_prev"] = df.groupby("item_id").cumcount()
    df["user_item_event_count_prev"] = df.groupby(["user_id", "item_id"]).cumcount()

    df["user_time_since_prev_event_sec"] = (
        df.groupby("user_id")["event_ts"].diff().dt.total_seconds().fillna(-1.0)
    )
    df["item_time_since_prev_event_sec"] = (
        df.groupby("item_id")["event_ts"].diff().dt.total_seconds().fillna(-1.0)
    )
    df["user_item_time_since_prev_event_sec"] = (
        df.groupby(["user_id", "item_id"])["event_ts"].diff().dt.total_seconds().fillna(-1.0)
    )

    candidates = df[df["event_type"].isin(["view", "addtocart"])].copy()
    horizon = pd.Timedelta(hours=horizon_hours)
    candidates[LABEL_COLUMN] = (
        candidates["next_purchase_ts"].notna()
        & (candidates["next_purchase_ts"] > candidates["event_ts"])
        & (candidates["next_purchase_ts"] <= candidates["event_ts"] + horizon)
    ).astype(int)
    candidates["is_addtocart"] = (candidates["event_type"] == "addtocart").astype(int)

    feature_columns = [feature for feature in MODEL_FEATURES if feature != "is_addtocart"]
    candidates = candidates.join(df[feature_columns], how="left")
    return candidates


def split_candidate_frame(
    candidates: pd.DataFrame,
    *,
    val_days: int = 7,
    test_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_ts = candidates["event_ts"].max()
    test_start = max_ts - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)

    train = candidates[candidates["event_ts"] < val_start].copy()
    val = candidates[
        (candidates["event_ts"] >= val_start) & (candidates["event_ts"] < test_start)
    ].copy()
    test = candidates[candidates["event_ts"] >= test_start].copy()
    return train, val, test


def downsample_train_negatives(
    train: pd.DataFrame,
    *,
    neg_ratio: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    pos = train[train[LABEL_COLUMN] == 1]
    neg = train[train[LABEL_COLUMN] == 0]
    if len(pos) == 0:
        return train
    neg_sample = neg.sample(n=min(len(neg), len(pos) * neg_ratio), random_state=random_state)
    return pd.concat([pos, neg_sample], axis=0).sample(frac=1.0, random_state=random_state)
