from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from feature_repo.feature_definitions import MODEL_FEATURES

POPULAR_EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}
GENERATED_REQUEST_MODE = "generated_recent_and_popular"


def resolve_recommendation_events_path(configured_path: str) -> Path:
    candidates = [
        Path(configured_path),
        Path("data/processed/events_retailrocket.parquet"),
    ]
    for path in candidates:
        if path.exists():
            return path
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Candidate generation requires a processed events parquet. Checked: {checked}"
    )


def load_events_frame(events_path: Path) -> pd.DataFrame:
    events = pd.read_parquet(events_path, columns=["user_id", "item_id", "event_type", "event_ts"])
    events["event_ts"] = pd.to_datetime(events["event_ts"], utc=True, errors="coerce")
    events = events.dropna(subset=["user_id", "item_id", "event_ts"]).copy()
    if events.empty:
        raise ValueError(f"No usable events found in {events_path}")
    events["user_id"] = events["user_id"].astype("int64")
    events["item_id"] = events["item_id"].astype("int64")
    events["event_weight"] = events["event_type"].map(POPULAR_EVENT_WEIGHTS).fillna(1.0)
    return events.sort_values("event_ts", ascending=False).reset_index(drop=True)


def build_point_in_time_feature_rows(
    history: pd.DataFrame,
    *,
    user_id: int,
    item_ids: Sequence[int],
    prediction_time: pd.Timestamp,
    is_addtocart: int = 0,
) -> pd.DataFrame:
    user_history = history[history["user_id"] == int(user_id)]
    item_history = history[history["item_id"].isin(item_ids)]
    user_item_history = history[
        (history["user_id"] == int(user_id)) & (history["item_id"].isin(item_ids))
    ].copy()

    user_event_count_prev = float(len(user_history))
    user_last_event_ts = user_history["event_ts"].max() if not user_history.empty else pd.NaT

    item_stats = (
        item_history.groupby("item_id", as_index=False)
        .agg(
            item_event_count_prev=("event_ts", "size"),
            item_last_event_ts=("event_ts", "max"),
        )
        .set_index("item_id")
    )
    user_item_stats = (
        user_item_history.groupby("item_id", as_index=False)
        .agg(
            user_item_event_count_prev=("event_ts", "size"),
            user_item_last_event_ts=("event_ts", "max"),
        )
        .set_index("item_id")
    )

    def _seconds_since(ts: Optional[pd.Timestamp]) -> float:
        if ts is None or pd.isna(ts):
            return -1.0
        return float(max((prediction_time - ts).total_seconds(), 0.0))

    rows = []
    for item_id in item_ids:
        item_stat = item_stats.loc[item_id] if item_id in item_stats.index else None
        user_item_stat = user_item_stats.loc[item_id] if item_id in user_item_stats.index else None
        rows.append(
            {
                "is_addtocart": float(is_addtocart),
                "user_event_count_prev": user_event_count_prev,
                "item_event_count_prev": float(
                    item_stat["item_event_count_prev"] if item_stat is not None else 0.0
                ),
                "user_item_event_count_prev": float(
                    user_item_stat["user_item_event_count_prev"] if user_item_stat is not None else 0.0
                ),
                "user_time_since_prev_event_sec": _seconds_since(user_last_event_ts),
                "item_time_since_prev_event_sec": _seconds_since(
                    item_stat["item_last_event_ts"] if item_stat is not None else pd.NaT
                ),
                "user_item_time_since_prev_event_sec": _seconds_since(
                    user_item_stat["user_item_last_event_ts"] if user_item_stat is not None else pd.NaT
                ),
            }
        )

    return pd.DataFrame(rows, columns=MODEL_FEATURES)


@dataclass(slots=True)
class RecommendationCandidate:
    item_id: int
    is_addtocart: int = 0
    user_event_count_prev: Optional[float] = None
    item_event_count_prev: Optional[float] = None
    user_item_event_count_prev: Optional[float] = None
    user_time_since_prev_event_sec: Optional[float] = None
    item_time_since_prev_event_sec: Optional[float] = None
    user_item_time_since_prev_event_sec: Optional[float] = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "RecommendationCandidate":
        return cls(
            item_id=int(payload["item_id"]),
            is_addtocart=int(payload.get("is_addtocart", 0)),
            user_event_count_prev=_optional_float(payload.get("user_event_count_prev")),
            item_event_count_prev=_optional_float(payload.get("item_event_count_prev")),
            user_item_event_count_prev=_optional_float(payload.get("user_item_event_count_prev")),
            user_time_since_prev_event_sec=_optional_float(
                payload.get("user_time_since_prev_event_sec")
            ),
            item_time_since_prev_event_sec=_optional_float(payload.get("item_time_since_prev_event_sec")),
            user_item_time_since_prev_event_sec=_optional_float(
                payload.get("user_item_time_since_prev_event_sec")
            ),
        )


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    return float(value)


@dataclass(slots=True)
class CandidateGenerator:
    events_path: Path
    user_recent_items: Dict[int, List[int]]
    popular_items: List[int]
    candidate_pool_multiplier: int = 10
    min_candidate_pool_size: int = 50

    @classmethod
    def from_events(
        cls,
        events: pd.DataFrame,
        *,
        events_path: Optional[Path] = None,
        recent_days: int = 90,
        max_user_recent_items: int = 50,
        max_popular_items: int = 500,
        candidate_pool_multiplier: int = 10,
        min_candidate_pool_size: int = 50,
    ) -> "CandidateGenerator":
        if events.empty:
            raise ValueError("No usable events were provided for candidate generation")
        events = events.sort_values("event_ts", ascending=False).reset_index(drop=True)

        cutoff = events["event_ts"].max() - pd.Timedelta(days=recent_days)
        recent_events = events[events["event_ts"] >= cutoff].copy()
        if recent_events.empty:
            recent_events = events

        user_recent_items: Dict[int, List[int]] = {}
        for user_id, frame in events.groupby("user_id", sort=False):
            recent_unique = frame["item_id"].drop_duplicates().head(max_user_recent_items)
            user_recent_items[int(user_id)] = [int(item_id) for item_id in recent_unique.tolist()]

        popular_items = (
            recent_events.groupby("item_id", as_index=False)["event_weight"]
            .sum()
            .sort_values(["event_weight", "item_id"], ascending=[False, True])
            .head(max_popular_items)["item_id"]
            .astype("int64")
            .tolist()
        )

        return cls(
            events_path=events_path or Path("<in_memory_events>"),
            user_recent_items=user_recent_items,
            popular_items=[int(item_id) for item_id in popular_items],
            candidate_pool_multiplier=int(candidate_pool_multiplier),
            min_candidate_pool_size=int(min_candidate_pool_size),
        )

    @classmethod
    def from_parquet(
        cls,
        events_path: Path,
        *,
        recent_days: int = 90,
        max_user_recent_items: int = 50,
        max_popular_items: int = 500,
        candidate_pool_multiplier: int = 10,
        min_candidate_pool_size: int = 50,
    ) -> "CandidateGenerator":
        events = load_events_frame(events_path)
        return cls.from_events(
            events,
            events_path=events_path,
            recent_days=recent_days,
            max_user_recent_items=max_user_recent_items,
            max_popular_items=max_popular_items,
            candidate_pool_multiplier=candidate_pool_multiplier,
            min_candidate_pool_size=min_candidate_pool_size,
        )

    def candidate_pool_limit(self, top_k: int) -> int:
        return max(int(top_k) * self.candidate_pool_multiplier, self.min_candidate_pool_size)

    def _dedupe_extend(self, ordered: List[int], seen: set[int], values: Iterable[int], *, pool_limit: int) -> None:
        for value in values:
            item_id = int(value)
            if item_id in seen:
                continue
            ordered.append(item_id)
            seen.add(item_id)
            if len(ordered) >= pool_limit:
                return

    def generate_for_user(self, user_id: int, top_k: int) -> List[int]:
        pool_limit = self.candidate_pool_limit(top_k)
        ordered: List[int] = []
        seen: set[int] = set()

        recent_items = self.user_recent_items.get(int(user_id), [])
        self._dedupe_extend(ordered, seen, recent_items, pool_limit=pool_limit)
        if len(ordered) >= pool_limit:
            return ordered

        self._dedupe_extend(ordered, seen, self.popular_items, pool_limit=pool_limit)
        return ordered


def resolve_recommendation_candidates(
    *,
    user_id: Optional[int],
    top_k: int,
    is_addtocart: int,
    candidate_item_ids: Optional[Sequence[int]],
    manual_candidates: Optional[Sequence[Mapping[str, object]]],
    candidate_generator: Optional[CandidateGenerator],
    candidate_generator_error: Optional[str],
) -> tuple[List[RecommendationCandidate], str]:
    if candidate_item_ids is not None and manual_candidates is not None:
        raise ValueError("Provide either candidate_item_ids or candidates, not both")

    if candidate_item_ids is not None:
        candidates = [
            RecommendationCandidate(item_id=int(item_id), is_addtocart=int(is_addtocart))
            for item_id in candidate_item_ids
        ]
        return candidates, "candidate_item_ids"

    if manual_candidates is not None:
        candidates = [
            RecommendationCandidate.from_mapping(candidate_payload)
            for candidate_payload in manual_candidates
        ]
        return candidates, "candidates"

    if user_id is None:
        raise ValueError(
            "Provide user_id for generated candidates, or candidate_item_ids/candidates for manual candidate scoring"
        )

    if candidate_generator is None:
        raise RuntimeError(
            candidate_generator_error
            or "Candidate generator is not available; provide candidate_item_ids or candidates explicitly"
        )

    generated_item_ids = candidate_generator.generate_for_user(int(user_id), int(top_k))
    if not generated_item_ids:
        raise ValueError(
            "Candidate generation returned no items; provide candidate_item_ids or candidates explicitly"
        )

    candidates = [
        RecommendationCandidate(item_id=item_id, is_addtocart=int(is_addtocart))
        for item_id in generated_item_ids
    ]
    return candidates, GENERATED_REQUEST_MODE
