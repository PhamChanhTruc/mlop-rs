from __future__ import annotations

import argparse
import json
from redis import Redis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect realtime Redis feature hashes.")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--item-id", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        decode_responses=True,
    )

    keys = {
        f"user:{args.user_id}": client.hgetall(f"user:{args.user_id}"),
        f"item:{args.item_id}": client.hgetall(f"item:{args.item_id}"),
        f"user_item:{args.user_id}:{args.item_id}": client.hgetall(
            f"user_item:{args.user_id}:{args.item_id}"
        ),
    }
    print(json.dumps(keys, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
