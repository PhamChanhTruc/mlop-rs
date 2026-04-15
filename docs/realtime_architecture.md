# Realtime Architecture

## Purpose

The current realtime path exists to demonstrate that new user events can update online features quickly enough for live inference in the local thesis stack.

## Implemented Path

The implemented local path is:

`User/Event -> Kafka topic (user_events) -> realtime processor -> Redis -> FastAPI inference`

Files involved:
- `streaming/producer/simulate_events.py`: publishes demo events to Kafka
- `streaming/schema.py`: shared event schema and validation
- `streaming/flink/realtime_feature_job.py`: consumes Kafka events and updates Redis hashes
- `serving/realtime_features.py`: reads Redis hashes into model feature rows
- `serving/realtime_app.py`: prefers realtime Redis features during inference

## What The Processor Actually Does

For each incoming event, the processor updates three Redis hash namespaces:
- `user:<user_id>`
- `item:<item_id>`
- `user_item:<user_id>:<item_id>`

It stores:
- event counts
- last event timestamps

These values are converted by the API into the same model inputs used by the offline-trained XGBoost model:
- previous event counts
- time since previous event
- `is_addtocart`

## Relationship To Feast

The repository uses Feast + Redis for the batch-built online feature store.

The realtime path is related but not identical:
- Feast materialization populates Redis from offline-generated parquet sources
- the realtime processor writes Redis hashes directly for low-latency demo updates
- `serving/realtime_app.py` tries realtime Redis first, then Feast online lookup, then manual payload fallback

So the realtime demo complements Feast-based online serving, but it is not a full Feast streaming ingestion pipeline.

## What Is Done

- local Kafka topic for incoming demo events
- shared streaming schema
- stable realtime processor process
- Redis updates for user/item/user-item state
- API inference that can use those realtime updates immediately
- verification helpers in `scripts/verify_realtime_redis.py` and `scripts/verify_realtime_inference.sh`

## What Is Partial

- Flink infrastructure exists in Docker Compose, but the implemented processor is a Python Kafka consumer running inside the Flink-stage image
- the pipeline demonstrates realtime feature updates, but not advanced stream processing semantics
- online feature management is split between Feast materialization and direct Redis realtime updates

## What Should Not Be Claimed

- do not describe this as a submitted PyFlink/Flink job
- do not claim checkpointing, exactly-once guarantees, or complex windowed aggregation
- do not claim that Feast alone manages the realtime update path

## Defense-Friendly Wording

Recommended phrasing:

"The project includes a lightweight realtime feature update path for local demonstration. User events are published to Kafka, consumed by a Python processor running in the Flink-stage container, written into Redis, and then used by the FastAPI service at inference time. This demonstrates realtime feature refresh, but it is not yet a full production Flink streaming pipeline."
