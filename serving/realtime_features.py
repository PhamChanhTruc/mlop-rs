from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from redis import Redis


def _coerce_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None or value == "":
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _seconds_since(ts: Optional[pd.Timestamp]) -> float:
    if ts is None:
        return -1.0
    delta = (pd.Timestamp.now(tz="UTC") - ts).total_seconds()
    return float(max(delta, 0.0))


@dataclass(slots=True)
class RealtimeFeatureReader:
    host: str
    port: int
    db: int = 0

    @classmethod
    def from_env(cls) -> "RealtimeFeatureReader":
        return cls(
            host=os.getenv("REALTIME_REDIS_HOST", os.getenv("REDIS_HOST", "redis")),
            port=int(os.getenv("REALTIME_REDIS_PORT", os.getenv("REDIS_PORT", "6379"))),
            db=int(os.getenv("REALTIME_REDIS_DB", "0")),
        )

    def _client(self) -> Redis:
        return Redis(host=self.host, port=self.port, db=self.db, decode_responses=True)

    def ping(self) -> None:
        self._client().ping()

    def build_feature_row(self, user_id: int, item_id: int, is_addtocart: int) -> Dict[str, float]:
        client = self._client()
        user_key = f"user:{int(user_id)}"
        item_key = f"item:{int(item_id)}"
        user_item_key = f"user_item:{int(user_id)}:{int(item_id)}"

        user_values = client.hgetall(user_key)
        item_values = client.hgetall(item_key)
        user_item_values = client.hgetall(user_item_key)

        if not any([user_values, item_values, user_item_values]):
            raise ValueError(
                f"No realtime Redis features found for user_id={user_id}, item_id={item_id}"
            )

        return {
            "is_addtocart": float(is_addtocart),
            "user_event_count_prev": float(user_values.get("user_event_count_prev", 0.0)),
            "item_event_count_prev": float(item_values.get("item_event_count_prev", 0.0)),
            "user_item_event_count_prev": float(
                user_item_values.get("user_item_event_count_prev", 0.0)
            ),
            "user_time_since_prev_event_sec": _seconds_since(
                _coerce_timestamp(user_values.get("user_last_event_ts"))
            ),
            "item_time_since_prev_event_sec": _seconds_since(
                _coerce_timestamp(item_values.get("item_last_event_ts"))
            ),
            "user_item_time_since_prev_event_sec": _seconds_since(
                _coerce_timestamp(user_item_values.get("user_item_last_event_ts"))
            ),
        }
