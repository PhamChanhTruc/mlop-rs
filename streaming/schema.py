from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

ALLOWED_EVENT_TYPES = {"view", "addtocart", "favorite", "transaction"}


@dataclass(slots=True)
class UserEvent:
    user_id: int
    item_id: int
    event_type: str
    event_ts: str
    session_id: Optional[str] = None

    def validate(self) -> None:
        if self.user_id < 0:
            raise ValueError("user_id must be non-negative")
        if self.item_id < 0:
            raise ValueError("item_id must be non-negative")
        if self.event_type not in ALLOWED_EVENT_TYPES:
            raise ValueError(
                f"event_type must be one of {sorted(ALLOWED_EVENT_TYPES)}, got '{self.event_type}'"
            )
        datetime.fromisoformat(self.event_ts.replace("Z", "+00:00"))

    def to_json_bytes(self) -> bytes:
        self.validate()
        return json.dumps(asdict(self), separators=(",", ":")).encode("utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_user_event(
    user_id: int,
    item_id: int,
    event_type: str,
    *,
    event_ts: Optional[str] = None,
    session_id: Optional[str] = None,
) -> UserEvent:
    event = UserEvent(
        user_id=int(user_id),
        item_id=int(item_id),
        event_type=event_type,
        event_ts=event_ts or utc_now_iso(),
        session_id=session_id,
    )
    event.validate()
    return event
