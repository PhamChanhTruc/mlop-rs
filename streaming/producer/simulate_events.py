from __future__ import annotations

import argparse
import random
import time
import uuid
from pathlib import Path
import sys

from kafka import KafkaProducer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streaming.schema import ALLOWED_EVENT_TYPES, make_user_event

DEFAULT_TOPIC = "user_events"
DEFAULT_BOOTSTRAP_SERVERS = "localhost:29092"
DEFAULT_EVENT_TYPES = ["view", "addtocart", "favorite", "transaction"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish simulated user events to Kafka.")
    parser.add_argument("--bootstrap-servers", default=DEFAULT_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--num-events", type=int, default=10)
    parser.add_argument("--sleep-sec", type=float, default=0.25)
    parser.add_argument("--user-id-start", type=int, default=1)
    parser.add_argument("--user-id-end", type=int, default=20)
    parser.add_argument("--item-id-start", type=int, default=100)
    parser.add_argument("--item-id-end", type=int, default=140)
    parser.add_argument(
        "--event-types",
        default=",".join(DEFAULT_EVENT_TYPES),
        help=f"Comma-separated event types. Allowed: {sorted(ALLOWED_EVENT_TYPES)}",
    )
    parser.add_argument("--session-prefix", default="demo-session")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_types = [value.strip() for value in args.event_types.split(",") if value.strip()]
    invalid = sorted(set(event_types) - ALLOWED_EVENT_TYPES)
    if invalid:
        raise ValueError(f"Unsupported event types requested: {invalid}")

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        key_serializer=lambda value: str(value).encode("utf-8"),
        value_serializer=lambda value: value,
    )

    print(
        f"Publishing {args.num_events} events to topic '{args.topic}' via {args.bootstrap_servers}"
    )
    try:
        for index in range(args.num_events):
            event = make_user_event(
                user_id=random.randint(args.user_id_start, args.user_id_end),
                item_id=random.randint(args.item_id_start, args.item_id_end),
                event_type=random.choice(event_types),
                session_id=f"{args.session_prefix}-{uuid.uuid4().hex[:8]}",
            )
            producer.send(
                args.topic,
                key=event.user_id,
                value=event.to_json_bytes(),
            ).get(timeout=30)
            print(event.to_json_bytes().decode("utf-8"))
            if index + 1 < args.num_events:
                time.sleep(args.sleep_sec)
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
