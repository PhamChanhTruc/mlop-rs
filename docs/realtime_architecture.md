# Realtime Architecture Scaffolding

Current local realtime path scaffold:

User -> Event -> Kafka topic (`user_events`) -> Flink -> Redis

Implemented in this phase:
- single-node local Kafka in Docker Compose
- local Flink JobManager + TaskManager in Docker Compose
- shared streaming event schema in `streaming/schema.py`
- demo event producer in `streaming/producer/simulate_events.py`

Deferred to later phases:
- actual Flink job that consumes from Kafka
- writing streaming aggregates into Redis
- integrating streaming updates into Feast online features
- changing `serving/app.py` to consume realtime-updated online features differently
