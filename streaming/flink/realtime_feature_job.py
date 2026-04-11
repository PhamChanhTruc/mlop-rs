from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from kafka import KafkaConsumer, TopicPartition
from kafka.errors import CommitFailedError
from redis import Redis

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streaming.schema import UserEvent

LOGGER = logging.getLogger("realtime_feature_job")
DEFAULT_BOOTSTRAP_SERVERS = "kafka:9092"
DEFAULT_TOPIC = "user_events"
DEFAULT_GROUP_ID = "realtime-feature-job"
DEFAULT_REDIS_HOST = "redis"
DEFAULT_REDIS_PORT = 6379
DEFAULT_POLL_TIMEOUT_MS = 250
DEFAULT_MAX_POLL_RECORDS = 25
DEFAULT_MAX_POLL_INTERVAL_MS = 300000
DEFAULT_SESSION_TIMEOUT_MS = 10000
DEFAULT_HEARTBEAT_INTERVAL_MS = 3000
DEFAULT_METADATA_MAX_AGE_MS = 600000
DEFAULT_TOPIC_METADATA_WAIT_SEC = 30
DEFAULT_MODE = "standalone"
DEFAULT_PARTITION = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consume Kafka user events and update simple online feature hashes in Redis."
    )
    parser.add_argument("--bootstrap-servers", default=DEFAULT_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=["standalone", "group"])
    parser.add_argument("--partition", type=int, default=DEFAULT_PARTITION)
    parser.add_argument("--group-id", default=DEFAULT_GROUP_ID)
    parser.add_argument("--auto-offset-reset", default="earliest", choices=["earliest", "latest"])
    parser.add_argument("--redis-host", default=DEFAULT_REDIS_HOST)
    parser.add_argument("--redis-port", type=int, default=DEFAULT_REDIS_PORT)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--poll-timeout-ms", type=int, default=DEFAULT_POLL_TIMEOUT_MS)
    parser.add_argument("--max-poll-records", type=int, default=DEFAULT_MAX_POLL_RECORDS)
    parser.add_argument("--max-poll-interval-ms", type=int, default=DEFAULT_MAX_POLL_INTERVAL_MS)
    parser.add_argument("--session-timeout-ms", type=int, default=DEFAULT_SESSION_TIMEOUT_MS)
    parser.add_argument("--heartbeat-interval-ms", type=int, default=DEFAULT_HEARTBEAT_INTERVAL_MS)
    parser.add_argument("--metadata-max-age-ms", type=int, default=DEFAULT_METADATA_MAX_AGE_MS)
    parser.add_argument("--topic-metadata-wait-sec", type=int, default=DEFAULT_TOPIC_METADATA_WAIT_SEC)
    return parser.parse_args()


def _parse_event(raw_value: bytes) -> UserEvent:
    payload = json.loads(raw_value.decode("utf-8"))
    event = UserEvent(**payload)
    event.validate()
    return event


def _update_redis(redis_client: Redis, event: UserEvent) -> None:
    user_key = f"user:{event.user_id}"
    item_key = f"item:{event.item_id}"
    user_item_key = f"user_item:{event.user_id}:{event.item_id}"

    pipe = redis_client.pipeline(transaction=True)
    pipe.hincrby(user_key, "user_event_count_prev", 1)
    pipe.hset(user_key, "user_last_event_ts", event.event_ts)

    pipe.hincrby(item_key, "item_event_count_prev", 1)
    pipe.hset(item_key, "item_last_event_ts", event.event_ts)

    pipe.hincrby(user_item_key, "user_item_event_count_prev", 1)
    pipe.hset(user_item_key, "user_item_last_event_ts", event.event_ts)
    pipe.execute()


def _wait_for_topic_partition(consumer: KafkaConsumer, topic: str, partition: int, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        partitions = consumer.partitions_for_topic(topic)
        if partitions and partition in partitions:
            LOGGER.info("Kafka metadata ready for topic=%s partitions=%s", topic, sorted(partitions))
            return
        LOGGER.info("Waiting for Kafka metadata for topic=%s partition=%s", topic, partition)
        time.sleep(1)
    raise RuntimeError(
        f"Timed out waiting for Kafka metadata for topic={topic} partition={partition}"
    )


def _build_consumer(args: argparse.Namespace) -> KafkaConsumer:
    common_kwargs = dict(
        bootstrap_servers=args.bootstrap_servers,
        auto_offset_reset=args.auto_offset_reset,
        enable_auto_commit=False,
        consumer_timeout_ms=0,
        max_poll_records=args.max_poll_records,
        metadata_max_age_ms=args.metadata_max_age_ms,
        value_deserializer=lambda value: value,
        key_deserializer=lambda value: value.decode("utf-8") if value is not None else None,
    )

    if args.mode == "group":
        consumer = KafkaConsumer(
            args.topic,
            group_id=args.group_id,
            max_poll_interval_ms=args.max_poll_interval_ms,
            session_timeout_ms=args.session_timeout_ms,
            heartbeat_interval_ms=args.heartbeat_interval_ms,
            **common_kwargs,
        )
        LOGGER.info(
            "Consuming in group mode topic=%s via %s with group_id=%s max_poll_records=%s max_poll_interval_ms=%s session_timeout_ms=%s heartbeat_interval_ms=%s metadata_max_age_ms=%s",
            args.topic,
            args.bootstrap_servers,
            args.group_id,
            args.max_poll_records,
            args.max_poll_interval_ms,
            args.session_timeout_ms,
            args.heartbeat_interval_ms,
            args.metadata_max_age_ms,
        )
        return consumer

    consumer = KafkaConsumer(group_id=None, **common_kwargs)
    _wait_for_topic_partition(
        consumer=consumer,
        topic=args.topic,
        partition=args.partition,
        timeout_sec=args.topic_metadata_wait_sec,
    )
    topic_partition = TopicPartition(args.topic, args.partition)
    consumer.assign([topic_partition])
    if args.auto_offset_reset == "earliest":
        consumer.seek_to_beginning(topic_partition)
    else:
        consumer.seek_to_end(topic_partition)
    LOGGER.info(
        "Consuming in standalone mode topic=%s partition=%s via %s auto_offset_reset=%s max_poll_records=%s metadata_max_age_ms=%s",
        args.topic,
        args.partition,
        args.bootstrap_servers,
        args.auto_offset_reset,
        args.max_poll_records,
        args.metadata_max_age_ms,
    )
    return consumer


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    redis_client = Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        decode_responses=True,
    )
    redis_client.ping()
    LOGGER.info(
        "Connected to Redis at %s:%s db=%s",
        args.redis_host,
        args.redis_port,
        args.redis_db,
    )

    consumer = _build_consumer(args)

    while True:
        records = consumer.poll(timeout_ms=args.poll_timeout_ms, max_records=args.max_poll_records)
        if not records:
            continue

        processed_count = 0
        try:
            for _partition, messages in records.items():
                for message in messages:
                    event = _parse_event(message.value)
                    _update_redis(redis_client, event)
                    processed_count += 1
                    LOGGER.info(
                        "Processed event user_id=%s item_id=%s event_type=%s event_ts=%s offset=%s",
                        event.user_id,
                        event.item_id,
                        event.event_type,
                        event.event_ts,
                        message.offset,
                    )
            if args.mode == "group":
                consumer.commit()
                LOGGER.info("Committed offsets after %s successful Redis updates", processed_count)
        except CommitFailedError as exc:
            LOGGER.warning(
                "Offset commit failed after %s processed messages; the consumer likely rebalanced: %s",
                processed_count,
                exc,
            )
        except Exception:
            LOGGER.exception(
                "Processing failed before offset commit; Kafka may redeliver messages depending on mode"
            )


if __name__ == "__main__":
    main()