# MLOps-RS Runbook

## Project overview
This repository implements a thesis-oriented MLOps system for RetailRocket-style
purchase probability prediction and top-K recommendation. The current
implementation focuses on a local, containerized end-to-end workflow with a
minimal Cloud Run deployment path for the FastAPI inference service.

## Architecture overview
The implemented architecture is:
1. raw event preprocessing and leakage-aware train/validation/test split generation
2. canonical feature contract in `feature_repo/`
3. XGBoost baseline training and Optuna tuning with MLflow tracking and model registry
4. FastAPI serving for `/predict_proba` and retrieval-plus-ranking `/recommend`
5. Feast + Redis online feature lookup for serving
6. Airflow orchestration for local preprocessing, feature materialization, training, validation, and promotion
7. Prometheus + Grafana monitoring for the local API stack
8. Docker Compose for local multi-container development
9. local realtime demo path: User -> Event -> Kafka -> processor/Flink stage -> Redis -> Serving API
10. minimal Artifact Registry + Cloud Run deployment path for the inference service

## Repository map
- `data/`: preprocessing scripts and train/validation/test dataset construction
- `feature_repo/`: canonical feature definitions and Feast repository used by serving
- `training/`: baseline training, Optuna tuning, and MLflow promotion utilities
- `serving/`: FastAPI inference service and serving container image
- `infra/`: local Docker Compose stack, Airflow, MLflow, monitoring, Kafka, and Flink config
- `deploy/gcp/`: minimal Cloud Build, Artifact Registry, and Cloud Run deployment path
- `.github/workflows/`: lightweight GitHub Actions CI
- `streaming/`: shared event schema, demo Kafka producer, and local realtime processor
- `scripts/`: lightweight verification helpers for Redis and end-to-end realtime inference
- `smoke_mlflow.py`: small local MLflow connectivity helper

## Current end-to-end flow
1. preprocess raw RetailRocket events into processed parquet files
2. build a future-purchase prediction dataset with time-based splits
3. build online Feast source files and materialize features to Redis
4. optionally send simulated realtime events through Kafka so the processor updates Redis online state
5. train a baseline XGBoost model or optionally run Optuna tuning
6. register and promote a selected model version in MLflow
7. load the promoted model in FastAPI and serve `/predict_proba` and `/recommend`
8. expose inference metrics to Prometheus and visualize them in Grafana
9. optionally build and deploy the inference container to Cloud Run

## Fully implemented
- local data preprocessing and train/validation/test dataset generation
- purchase probability prediction through FastAPI
- top-K recommendation through FastAPI with simple candidate generation plus ranking
- canonical feature definitions and offline feature engineering in `feature_repo/`
- Feast + Redis online feature lookup for serving
- XGBoost baseline training and Optuna tuning with MLflow tracking
- MLflow model registration and promotion with PostgreSQL backend
- Airflow local orchestration with PostgreSQL metadata backend
- Prometheus + Grafana local monitoring stack
- Docker and Docker Compose local development stack
- lightweight GitHub Actions CI for syntax, imports, and CLI smoke checks
- minimal Cloud Run deployment files for the inference service
- local realtime event path from Kafka through a processor/Flink stage into Redis, with realtime-first inference in the API

## Partially implemented
- candidate generation is intentionally lightweight: recent user interactions plus recent popular items, not a learned retriever
- monitoring is focused on API-level health and inference metrics, not full model/data quality monitoring
- the Cloud Run path covers the FastAPI inference service only, not the full MLOps stack
- the realtime path is intentionally lightweight for thesis demo use: a stable Kafka processor in the Flink stage container, not a submitted Flink job
- the cloud story still depends on external MLflow and Redis/Feast connectivity

## Cloud Run path: external/manual pieces
The repository includes the deployment files, but the following still require
manual cloud setup:
- GCP project, billing, IAM, and `gcloud` authentication
- Artifact Registry and Cloud Run permissions
- a reachable `MLFLOW_TRACKING_URI` if the service loads `MODEL_URI=models:/...`
- a reachable Redis-backed Feast online store for Feast-backed online inference
- a cloud-appropriate Feast configuration instead of the current local `redis:6379` setting
- any external storage such as GCS for artifacts, exported datasets, or feature files

## Limitations and future work
- add true Feast historical retrieval if the project later needs a fully managed offline feature retrieval workflow instead of the current canonical offline feature builder
- replace the current lightweight processor-in-Flink-stage approach with a submitted Flink job only if the project needs stronger streaming semantics
- expand monitoring to model-quality, data-drift, and infrastructure-level signals
- separate cloud runtime dependencies cleanly for MLflow, Redis, and Feast configuration
- add broader CI/CD and cloud automation once project secrets and deployment environments are available
## 1) Start infra
```bash
cd infra
docker compose up -d --build
docker compose ps
```

Optional: start Airflow for orchestration
```bash
cd infra
AIRFLOW_UID=$(id -u) docker compose up -d --build airflow
docker compose logs -f airflow
```

Airflow is already configured to use PostgreSQL metadata in the local stack via
`AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` in `infra/docker-compose.yml`, not SQLite.

Airflow now uses a dedicated PostgreSQL metadata database:
`postgresql+psycopg2://airflow:airflow@postgres:5432/airflow`

If you previously started Airflow against the shared `mlflow` database and hit
an Alembic revision error such as `Can't locate revision identified by
'1b5f0d9ad7c1'`, reset only the Airflow metadata database before restarting
Airflow:
```bash
cd infra
docker compose up -d postgres
docker compose exec postgres psql -U mlflow -d postgres -c "DO \$\$ BEGIN IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'airflow') THEN CREATE ROLE airflow LOGIN PASSWORD 'airflow'; ELSE ALTER ROLE airflow WITH LOGIN PASSWORD 'airflow'; END IF; END \$\$;"
docker compose exec postgres psql -U mlflow -d postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'airflow' AND pid <> pg_backend_pid();"
docker compose exec postgres psql -U mlflow -d postgres -c "DROP DATABASE IF EXISTS airflow;"
docker compose exec postgres psql -U mlflow -d postgres -c "CREATE DATABASE airflow OWNER airflow;"
docker compose stop airflow
docker compose rm -sf airflow
docker volume rm infra_airflow_home
```

Root cause:
- `1b5f0d9ad7c1` is an MLflow Alembic revision, not an Airflow one
- MLflow and Airflow both use Alembic metadata tables, so reusing the same PostgreSQL database can leave Airflow reading MLflow's revision chain
- once that happens, Airflow startup fails during metadata migration because it cannot resolve MLflow's revision history
- the clean fix is to keep Airflow on `postgresql+psycopg2://airflow:airflow@postgres:5432/airflow` and recreate only the dedicated `airflow` database if it was previously contaminated

Verify the Airflow metadata backend and DAG registration:
```bash
cd infra
docker compose exec airflow airflow config get-value database sql_alchemy_conn
docker compose exec airflow airflow db check
docker compose exec airflow airflow dags list
docker compose exec airflow airflow tasks list retailrocket_training_feature_pipeline --tree
```

Trigger the thesis pipeline manually:
```bash
cd infra
docker compose exec airflow airflow dags unpause retailrocket_training_feature_pipeline
docker compose exec airflow airflow dags trigger retailrocket_training_feature_pipeline
docker compose logs -f airflow
```

## 2) Build dataset (only when needed)
```bash
cd ..
python3 data/scripts/preprocess_retailrocket.py
python3 data/scripts/build_trainset_retailrocket.py
```

`build_trainset_retailrocket.py` now bootstraps the repo root into `sys.path`, so
running it directly from the repository root remains reliable even though it
imports the canonical top-level `feature_repo` package.

## 3) Train and log model to MLflow
```bash
cd ..
python3 -m training.train_xgb_baseline
```

Optional environment overrides:
```bash
MLFLOW_TRACKING_URI=http://localhost:5000 \
MLFLOW_EXPERIMENT=xgb-baseline-retailrocket \
python3 -m training.train_xgb_baseline
```

## 4) Reload model in API
```bash
curl -s -X POST http://localhost:8000/reload_model
curl -s http://localhost:8000/readyz
```

## 4.1) Build Feast online features and materialize to Redis
```bash
cd infra
docker compose exec api bash -lc "cd /app && python -m feature_repo.build_online_features"
docker compose exec api bash -lc "cd /app/feature_repo && feast apply"
docker compose exec api bash -lc "cd /app/feature_repo && feast materialize-incremental \$(date -u +%Y-%m-%dT%H:%M:%S)"
```

The canonical Feast repo lives at the project root in `feature_repo/` and is
mounted into the API container at `/app/feature_repo`.

Why these commands are robust now:
- `python -m feature_repo.build_online_features` runs from the repo root package context
- `feast apply` and `feast materialize-incremental` still run from the Feast repo directory
- `feature_repo/build_online_features.py` and `feature_repo/repo.py` now bootstrap the parent repo path into `sys.path`, so absolute imports from `feature_repo.*` still resolve even when the current working directory is `/app/feature_repo`

## 4.1.1) Start the realtime demo path: Kafka -> processor/Flink stage -> Redis
The implemented local thesis-demo path is:
`User -> Event -> Kafka -> processor/Flink stage -> Redis -> Serving API`

What is implemented now:
- local Kafka broker with one topic path: `user_events`
- local Flink JobManager + TaskManager infrastructure for the declared thesis stack
- a lightweight, stable realtime processor in `streaming/flink/realtime_feature_job.py`
  running inside the Flink-stage container image
- shared event schema in `streaming/schema.py`
- demo event producer in `streaming/producer/simulate_events.py`
- Redis verification helper in `scripts/verify_realtime_redis.py`
- end-to-end inference verification helper in `scripts/verify_realtime_inference.sh`

Why this is considered done for the local thesis demo:
- incoming simulated user events update Redis online features for `user`, `item`, and `user_item`
- the FastAPI service prefers those realtime Redis features first at inference time
- the path is stable and easy to demonstrate locally without adding unnecessary streaming complexity

What remains intentionally simplified:
- the processor runs as a lightweight Kafka consumer in the Flink stage container rather than a submitted PyFlink job
- no advanced windows, checkpointing, or exactly-once guarantees are claimed

Start realtime infrastructure and the processor:
```bash
cd /home/truc/mlops-rs/infra
docker compose up -d --build kafka flink-jobmanager flink-taskmanager realtime-processor
```

If Kafka previously failed with KRaft metadata errors, restart it cleanly:
```bash
cd /home/truc/mlops-rs/infra
docker compose rm -sf kafka realtime-processor
# optional: remove old local Kafka state if you want a fresh single-node KRaft log
docker volume rm infra_kafka_data

docker compose up -d --build kafka flink-jobmanager flink-taskmanager realtime-processor
```

Check services:
```bash
cd /home/truc/mlops-rs/infra
docker compose ps kafka flink-jobmanager flink-taskmanager realtime-processor
```

Open Flink locally:
- Flink UI: `http://localhost:8081`

Watch the realtime processor logs:
```bash
cd /home/truc/mlops-rs/infra
docker compose logs -f realtime-processor
```

Stability note for the current processor:
- the local stack runs the realtime processor in standalone Kafka mode on topic `user_events` partition `0`
- the processor waits for Kafka topic metadata before assigning the partition, which avoids repeated metadata-refresh warnings in local startup
- this removes Kafka group heartbeats and avoids `member_id` / coordinator churn for the single intended processor instance
- the compose service pins one named container: `mlops-rs-realtime-processor`
- if you still see stale behavior, recreate only the processor container first

Install the demo event producer dependency:
```bash
cd /home/truc/mlops-rs
source .venv/bin/activate
pip install -r streaming/requirements.txt
```

Publish sample events to Kafka:
```bash
cd /home/truc/mlops-rs
source .venv/bin/activate
python3 streaming/producer/simulate_events.py --bootstrap-servers localhost:29092 --topic user_events --num-events 5 --sleep-sec 0.2 --user-id-start 1 --user-id-end 1 --item-id-start 101 --item-id-end 101
```

Inspect the Kafka topic from the Kafka container:
```bash
cd /home/truc/mlops-rs/infra
docker compose exec kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server kafka:9092 \
  --topic user_events \
  --from-beginning \
  --max-messages 5
```

Inspect Redis feature hashes from WSL:
```bash
cd /home/truc/mlops-rs
source .venv/bin/activate
python3 scripts/verify_realtime_redis.py --user-id 1 --item-id 101
```

Or inspect Redis directly from the container:
```bash
cd /home/truc/mlops-rs/infra
docker compose exec redis redis-cli HGETALL user:1
docker compose exec redis redis-cli HGETALL item:101
docker compose exec redis redis-cli HGETALL user_item:1:101
```

Expected Redis behavior after the sample events above:
- `user:1.user_event_count_prev` increases with each event for user `1`
- `item:101.item_event_count_prev` increases with each event for item `101`
- `user_item:1:101.user_item_event_count_prev` increases with each event for that exact pair
- the corresponding `*_last_event_ts` fields update to the most recent processed `event_ts`

For example, after 5 events for the same user-item pair, Redis should show counts of `5`
on all three hashes and the latest ISO timestamp from the last processed event.

## 4.2) Canonical feature contract
`feature_repo/` is now the source of truth for the current model feature
definitions in both training and serving.

- Offline dataset construction imports the leakage-safe feature engineering from
  `feature_repo.feature_definitions`.
- Training imports the same canonical ordered model inputs from
  `feature_repo.feature_definitions` before reading parquet splits.
- `feature_repo.build_online_features` builds the corresponding online
  aggregates that Feast materializes to Redis.
- Serving reconstructs the same model inputs in the FastAPI layer, with
  `is_addtocart` coming from the request and historical aggregates coming from
  realtime Redis first, then Feast + Redis if needed.

This keeps offline and online features aligned through one canonical feature
module:
- `is_addtocart` stays request-time input
- offline counts/recency are built from processed events using the same named
  features training consumes
- online counts come from Redis-backed sources and online recency is derived
  from the stored last-event timestamps

Online inference path:
1. client sends `user_id`, `item_id`, and request-time context such as `is_addtocart`
2. FastAPI first tries realtime Redis hashes, then falls back to Feast + Redis if needed
3. FastAPI reconstructs the model row using the same feature names used in training
4. the model predicts purchase probability

Why Feast + Redis is considered done for the current thesis scope:
- the canonical feature definitions live in `feature_repo/`
- the offline training dataset is built from `feature_repo` feature logic rather
  than a separate ad hoc script
- Feast materializes online aggregate features to Redis for serving
- the API uses Redis-backed online state with the same feature names as training

## 5) Predict with realtime Redis first, then Feast + manual fallback
```bash
curl -s -X POST http://localhost:8000/predict_proba \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": 1,
    "item_id": 101,
    "is_addtocart": 1
  }'
```

The response now includes `feature_source`, which will typically be one of:
- `redis_realtime`
- `feast_online`
- `manual_fallback` or `manual_payload`

Training/serving feature parity for the current model is preserved as follows:
- `is_addtocart` is request-time context in both training and serving
- count features come from the same processed event history
- user/item/user-item recency features are reconstructed online from stored last-event timestamps

## 5.1) Predict with the old fully manual payload fallback
```bash
curl -s -X POST http://localhost:8000/predict_proba \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": 1,
    "item_id": 101,
    "is_addtocart": 1,
    "user_event_count_prev": 12,
    "item_event_count_prev": 34,
    "user_item_event_count_prev": 2,
    "user_time_since_prev_event_sec": 1800,
    "item_time_since_prev_event_sec": 3600,
    "user_item_time_since_prev_event_sec": 7200
  }'
```

Use this fallback only for local demo/testing or when Feast is unavailable.

## 5.2) Recommend Top-K with generated candidates plus ranking
```bash
curl -s -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": 1,
    "top_k": 5
  }'
```

This is now the preferred recommendation request shape:
- the caller provides `user_id` and `top_k`
- FastAPI generates a candidate pool from the user's recent interacted items, then fills with recent popular items if needed
- the existing scorer applies the realtime Redis first, Feast second, manual fallback feature path to each candidate
- the response ranks candidates by predicted purchase probability

If you want to override the generated pool, you can still supply explicit
candidate item IDs:
```bash
curl -s -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": 1,
    "top_k": 3,
    "candidate_item_ids": [101, 102, 103]
  }'
```

If you need shared request-time context for all candidates, you can also set
`is_addtocart` at the request level:
```bash
curl -s -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": 1,
    "top_k": 3,
    "is_addtocart": 1,
    "candidate_item_ids": [101, 102, 103]
  }'
```

## 5.2.1) Recommend Top-K with the old manual candidate payload fallback
```bash
curl -s -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": 1,
    "top_k": 3,
    "candidates": [
      {
        "item_id": 101,
        "is_addtocart": 1,
        "user_event_count_prev": 12,
        "item_event_count_prev": 34,
        "user_item_event_count_prev": 2,
        "user_time_since_prev_event_sec": 1800,
        "item_time_since_prev_event_sec": 3600,
        "user_item_time_since_prev_event_sec": 7200
      },
      {
        "item_id": 102,
        "is_addtocart": 0,
        "user_event_count_prev": 12,
        "item_event_count_prev": 20,
        "user_item_event_count_prev": 1,
        "user_time_since_prev_event_sec": 2400,
        "item_time_since_prev_event_sec": 5400,
        "user_item_time_since_prev_event_sec": 9600
      },
      {
        "item_id": 103,
        "is_addtocart": 1,
        "user_event_count_prev": 12,
        "item_event_count_prev": 8,
        "user_item_event_count_prev": 1,
        "user_time_since_prev_event_sec": 300,
        "item_time_since_prev_event_sec": 600,
        "user_item_time_since_prev_event_sec": 300
      }
    ]
  }'
```

## 5.2.2) End-to-end realtime inference demo
This is the clean end-to-end demo flow for the implemented realtime path:
1. send one or more events into Kafka
2. let the realtime processor update Redis
3. call `/predict_proba` or `/recommend`
4. confirm that `feature_source` is `redis_realtime`

Start or rebuild the local stack:
```bash
cd /home/truc/mlops-rs/infra
docker compose up -d --build redis kafka flink-jobmanager flink-taskmanager realtime-processor api
```

Watch the realtime processor and API logs:
```bash
cd /home/truc/mlops-rs/infra
docker compose logs -f realtime-processor api
```

Send demo events for a known user-item pair:
```bash
cd /home/truc/mlops-rs
source .venv/bin/activate
python3 streaming/producer/simulate_events.py \
  --bootstrap-servers localhost:29092 \
  --topic user_events \
  --num-events 5 \
  --sleep-sec 0.2 \
  --user-id-start 1 \
  --user-id-end 1 \
  --item-id-start 101 \
  --item-id-end 101
```

Wait briefly, then inspect the realtime Redis hashes:
```bash
cd /home/truc/mlops-rs
source .venv/bin/activate
python3 scripts/verify_realtime_redis.py --user-id 1 --item-id 101
```

Or run the whole demo verification in one command:
```bash
cd /home/truc/mlops-rs
source .venv/bin/activate
bash scripts/verify_realtime_inference.sh
```

A stable processor run should show:
- normal `Processed event ...` log lines with increasing offsets
- no repeated `Heartbeat session expired`, `member_id was not recognized`, or `NotCoordinatorForGroupError` loops

Call `/predict_proba` and confirm the response uses `redis_realtime`:
```bash
curl -s -X POST http://localhost:8000/predict_proba \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": 1,
    "item_id": 101,
    "is_addtocart": 1
  }'
```

Call `/recommend` and confirm the returned items use `redis_realtime` when the
candidate hashes exist in Redis:
```bash
curl -s -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": 1,
    "top_k": 3,
    "candidate_item_ids": [101, 102, 103]
  }'
```

What should change after the demo events:
- `user_event_count_prev` should increase for user `1`
- `item_event_count_prev` should increase for item `101`
- `user_item_event_count_prev` should increase for pair `1:101`
- the derived recency features should become small positive values rather than `-1`
- the API response should show `feature_source: "redis_realtime"` for the user-item pair that was updated

This is enough to treat the realtime path as done for the thesis demo because:
- the event source is simulated but real
- Kafka ingestion is real
- the processor updates real Redis online state
- the serving API consumes that updated state at inference time
- the entire path is containerized and demoable locally end to end
## 5.3) Tune XGBoost with Optuna + MLflow
```bash
cd ..
python3 -m training.tune_xgb_optuna --n-trials 30
```

## 5.4) Promote baseline model to Production used by the API
```bash
cd ..
python3 -m training.promote_model \
  --run-id <BASELINE_RUN_ID> \
  --model-name xgb-baseline-retailrocket \
  --artifact-path model
```

Then restart API so it reloads `models:/xgb-baseline-retailrocket/Production`:
```bash
cd infra
docker compose up -d --build api
curl -s http://localhost:8000/
```

## 6) Health and logs
```bash
curl -s http://localhost:8000/
curl -s http://localhost:8000/metrics | head
cd infra && docker compose logs -f api mlflow
```

## 6.1) Minimal demo UI
The repository now includes a lightweight thesis demo UI in `ui/app.py`.
It calls the existing FastAPI service and does not replace any API endpoint.

What it supports:
- purchase probability prediction with `user_id`, `item_id`, and `is_addtocart`
- top-K recommendation with `user_id`, `top_k`, and a comma-separated candidate item list

How it connects:
- the UI sends HTTP requests directly to the existing FastAPI endpoints:
  - `POST /predict_proba`
  - `POST /recommend`
- by default it uses `http://localhost:8000`
- you can override that with `API_BASE_URL`

Run locally in WSL:
```bash
cd /home/truc/mlops-rs
source .venv/bin/activate

pip install -r ui/requirements.txt
streamlit run ui/app.py
```

Optional API base URL override:
```bash
cd /home/truc/mlops-rs
source .venv/bin/activate

API_BASE_URL=http://localhost:8000 streamlit run ui/app.py
```

Open:
- UI: `http://localhost:8501`
- API: `http://localhost:8000`

Recommendation note:
- the current API still expects candidate item IDs for recommendation scoring
- the demo UI pre-fills this field with popular item IDs from `data/processed/events_retailrocket.parquet` when available
- you can also replace them manually in the UI for a custom demo

## 7) Airflow DAG for the local training + Feast pipeline
The repository now includes a more thesis-aligned local Airflow DAG:
- DAG id: `retailrocket_training_feature_pipeline`
- tasks: `preprocess_events` -> `build_trainset` -> both the Feast branch and the model branch

Feast branch:
1. `build_online_feature_sources`
2. `feast_apply`
3. `feast_materialize_online_features`

Model branch:
1. `train_baseline_model`
2. `tune_optuna_model` (optional, controlled by `AIRFLOW_ENABLE_OPTUNA_TUNING`)
3. `select_model_for_promotion`
4. `validate_model_candidate`
5. `promote_model`

What it does:
1. preprocesses RetailRocket events
2. rebuilds the train/validation/test parquet splits
3. rebuilds Feast parquet sources, applies the Feast repo, and materializes online features to Redis
4. trains the baseline XGBoost model
5. optionally runs Optuna tuning and captures the latest tuning parent run
6. validates the selected promotion candidate by checking required metrics and downloading the registered artifact path
7. promotes the selected run into MLflow Production

Airflow metadata backend:
- Airflow uses the existing local PostgreSQL service, but now keeps its metadata in a dedicated `airflow` database
- the connection string is `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres:5432/airflow`
- on a fresh PostgreSQL volume, `infra/postgres/init/01-create-airflow-db.sql` creates the Airflow role and database automatically
- if you previously ran Airflow against the shared `mlflow` database, reset the Airflow database before starting Airflow again so no corrupted Alembic state is reused
- `1b5f0d9ad7c1` specifically belongs to MLflow, so that error means Airflow is reading a shared or previously contaminated metadata database instead of a clean Airflow-only one

Open Airflow:
- UI: `http://localhost:8080`
- username: `admin`
- password: `admin`

Create or reset the dedicated Airflow metadata database:
```bash
cd infra
docker compose up -d postgres
docker compose exec postgres psql -U mlflow -d postgres -c "DO \$\$ BEGIN IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'airflow') THEN CREATE ROLE airflow LOGIN PASSWORD 'airflow'; ELSE ALTER ROLE airflow WITH LOGIN PASSWORD 'airflow'; END IF; END \$\$;"
docker compose exec postgres psql -U mlflow -d postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'airflow' AND pid <> pg_backend_pid();"
docker compose exec postgres psql -U mlflow -d postgres -c "DROP DATABASE IF EXISTS airflow;"
docker compose exec postgres psql -U mlflow -d postgres -c "CREATE DATABASE airflow OWNER airflow;"
docker compose stop airflow
docker compose rm -sf airflow
docker volume rm infra_airflow_home
```

Start or rebuild Airflow cleanly:
```bash
cd infra
AIRFLOW_UID=$(id -u) docker compose up -d --build airflow
docker compose logs -f airflow
```

The Airflow container now runs `airflow db migrate && airflow standalone`, so a clean dedicated metadata database is migrated before the local standalone webserver and scheduler start.

Verify Airflow is using PostgreSQL metadata:
```bash
cd infra
docker compose exec airflow airflow config get-value database sql_alchemy_conn
docker compose exec airflow airflow db check
```

Inspect the DAG:
```bash
cd infra
docker compose exec airflow airflow dags list
docker compose exec airflow airflow tasks list retailrocket_training_feature_pipeline --tree
```

Verify the DAG is visible in the UI:
- open `http://localhost:8080`
- log in with `admin` / `admin`
- confirm `retailrocket_training_feature_pipeline` appears in the DAG list

Trigger the default baseline-first pipeline:
```bash
cd infra
docker compose exec airflow airflow dags unpause retailrocket_training_feature_pipeline
docker compose exec airflow airflow dags trigger retailrocket_training_feature_pipeline
docker compose logs -f airflow
```

Trigger a tuning-enabled run by recreating the Airflow service with tuning enabled:
```bash
cd infra
AIRFLOW_UID=$(id -u) \
AIRFLOW_ENABLE_OPTUNA_TUNING=true \
AIRFLOW_PROMOTE_TUNED_IF_AVAILABLE=true \
docker compose up -d --build airflow

docker compose exec airflow airflow dags trigger retailrocket_training_feature_pipeline
```

Notes:
- the DAG keeps local development simple by running repository modules directly from `/opt/mlops-rs`
- validation is intentionally lightweight: it checks the candidate metric threshold and verifies the model artifact path can actually be downloaded from MLflow before promotion
- the default path still promotes the baseline registry model used by the API; tuned promotion is opt-in

This Airflow setup is still simplified:
- it uses `airflow standalone` in one container instead of split scheduler/webserver/triggerer services
- it shares the same PostgreSQL service with MLflow, but not the same database
- tuning selection is controlled by environment flags rather than DAG run parameters
- validation gates are basic metric + artifact checks, not full evaluation, drift, or approval workflows
- it remains a local orchestration layer and does not cover Google Cloud deployment or GitHub Actions automation

## 8) Monitoring (Prometheus + Grafana)
```bash
cd infra
docker compose up -d --build prometheus grafana
docker compose ps
```

Access:
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (`admin` / `admin`)

Grafana dashboard is auto-provisioned:
- Folder: `MLOps`
- Dashboard: `MLOps API Overview`

## 9) GitHub Actions
The repository now includes three thesis-aligned GitHub Actions workflows:
- `.github/workflows/ci.yml`
- `.github/workflows/manual-train.yml`
- `.github/workflows/deploy-cloud-run.yml`

### 9.1) `ci.yml`
This remains the lightweight automatic validation workflow for pushes and pull requests.

What it validates:
- Python 3.11 setup
- dependency installation for the current serving/training/feature-management stack
- syntax checks for the main data, feature, training, serving, and Airflow DAG entrypoints
- import smoke checks for the canonical feature module, the base FastAPI app, and the actual runtime entrypoint `serving.realtime_app`
- basic CLI smoke checks for the training promotion and tuning entrypoints
- shell syntax for the Cloud Run deploy script

What it still does not validate:
- live Docker Compose services
- live MLflow, Redis, Feast, Airflow, PostgreSQL, or Kafka connectivity
- Cloud Run deployment itself

Equivalent local WSL commands:
```bash
cd /home/truc/mlops-rs
source .venv/bin/activate

python3 -m py_compile \
  data/scripts/preprocess_retailrocket.py \
  data/scripts/build_trainset_retailrocket.py \
  feature_repo/__init__.py \
  feature_repo/feature_definitions.py \
  feature_repo/build_online_features.py \
  feature_repo/repo.py \
  training/__init__.py \
  training/train_xgb_baseline.py \
  training/tune_xgb_optuna.py \
  training/promote_model.py \
  serving/__init__.py \
  serving/app.py \
  serving/realtime_features.py \
  serving/realtime_app.py \
  infra/airflow/dags/retailrocket_baseline_pipeline.py \
  smoke_mlflow.py

python3 -c "import feature_repo.feature_definitions, feature_repo.build_online_features, training.train_xgb_baseline, training.tune_xgb_optuna, training.promote_model, serving.app, serving.realtime_app; print('core imports ok')"
python3 -m training.promote_model --help
python3 -m training.tune_xgb_optuna --help
bash -n deploy/gcp/deploy_cloud_run.sh
```

### 9.2) `manual-train.yml`
This is a manual `workflow_dispatch` workflow for lightweight repository-native training orchestration.

What it does:
- checks out the repository
- installs Python dependencies
- rebuilds the processed parquet files and train/validation/test dataset
- trains the baseline XGBoost model
- optionally runs Optuna tuning
- stores the resulting local MLflow file-tracking directory and rebuilt processed data as GitHub Actions artifacts

Why it is useful:
- it provides a real GitHub-hosted training/integration path
- it does not require secrets
- it stays aligned with the local training modules instead of inventing a separate CI-only training script

Important note:
- this workflow uses `MLFLOW_TRACKING_URI=file:${GITHUB_WORKSPACE}/mlruns`, so it is a GitHub-hosted training run, not a remote MLflow registry promotion flow

### 9.3) `deploy-cloud-run.yml`
This is a manual `workflow_dispatch` workflow for the existing Cloud Run deploy path.

What it does:
- authenticates to Google Cloud from GitHub Actions
- calls the existing `deploy/gcp/deploy_cloud_run.sh` script
- builds the inference image with Cloud Build
- pushes it to Artifact Registry
- deploys it to Cloud Run with the selected runtime environment variables

What still requires secrets or manual cloud setup:
- GitHub secret `GCP_SA_KEY_JSON` containing a service-account JSON key with permission for Cloud Build, Artifact Registry, and Cloud Run
- a configured GCP project with billing and required APIs enabled
- a reachable `MLFLOW_TRACKING_URI` if the deployed service should load `models:/...`
- a reachable Redis-backed Feast online store and cloud-appropriate Feast repo config if Cloud Run inference should use Feast-backed online features

Why GitHub Actions is considered done for the thesis scope:
- automatic CI exists for repository validation
- a real manual GitHub-hosted training workflow now exists without needing secrets
- a real manual deployment workflow now exists for the Cloud Run path
- the workflows align with the current runtime structure, including the actual serving entrypoint `serving.realtime_app`
## 10) Google Cloud Run deployment
The repository now includes a minimal Google Cloud deployment path for the
FastAPI inference service under `deploy/gcp/`.

Files:
- `deploy/gcp/deploy_cloud_run.sh`: builds the image with Cloud Build, pushes it to Artifact Registry, and deploys it to Cloud Run
- `deploy/gcp/cloudrun.env.example`: example environment variables for the deployment script
- `deploy/gcp/cloudrun.service.yaml`: optional Cloud Run service manifest with placeholders for a more explicit `gcloud run services replace` flow

Required local tools:
- `gcloud` CLI
- Docker is not required for the scripted path because it uses `gcloud builds submit`

Required manual cloud setup before deploy:
- authenticate with `gcloud auth login`
- select or create a GCP project
- enable billing on the project
- ensure you have permission to use Cloud Build, Artifact Registry, and Cloud Run
- if you want MLflow-backed model loading in Cloud Run, provide a reachable `MLFLOW_TRACKING_URI`

Minimal deploy flow:
```bash
cd /home/truc/mlops-rs
cp deploy/gcp/cloudrun.env.example deploy/gcp/cloudrun.env
# edit deploy/gcp/cloudrun.env with your project values
set -a
source deploy/gcp/cloudrun.env
set +a

bash deploy/gcp/deploy_cloud_run.sh
```

What the script does:
1. sets the active GCP project
2. enables Cloud Run, Artifact Registry, and Cloud Build APIs
3. creates the Artifact Registry repository if it does not exist yet
4. builds the current repository into the existing `serving/Dockerfile` image with Cloud Build
5. pushes that image to Artifact Registry
6. deploys the image to Cloud Run with the configured runtime environment variables

Alternative manifest-based flow:
```bash
cd /home/truc/mlops-rs
# edit deploy/gcp/cloudrun.service.yaml placeholders first

gcloud run services replace deploy/gcp/cloudrun.service.yaml --region "$GCP_REGION"
```

Important notes for the current thesis scope:
- the local Docker Compose workflow in `infra/docker-compose.yml` is unchanged
- the serving image now includes the top-level `feature_repo/` package so Cloud Run can import the same canonical feature contract used locally
- the serving container now honors Cloud Run's `PORT` environment variable while still defaulting to port `8000` locally
- this deploy path is intentionally minimal and only covers the FastAPI inference service

Service runtime dependencies to understand before deploy:
- `MODEL_URI` in `serving/app.py` controls which MLflow-registered or run-based model the API loads
- `MLFLOW_TRACKING_URI` in `serving/app.py` must point to a reachable MLflow server from Cloud Run if you want model loading to work
- `FEAST_REPO_PATH` in `serving/app.py` should remain `/app/feature_repo` for this container image
- Feast online inference also depends on Redis connectivity, but that is not currently configured through `serving/app.py` environment variables
- instead, the current top-level `feature_repo/feature_store.yaml` hardcodes `online_store.connection_string: redis:6379`, which works locally in Docker Compose but must be changed to a cloud-reachable Redis address before Feast-backed Cloud Run inference will work

What is implemented in-repo:
- a reusable Cloud Build + Artifact Registry + Cloud Run deployment script
- a placeholder Cloud Run service manifest
- example deployment environment variables
- README deployment steps

What is still manual or external:
- GCP project creation, billing, IAM, and authentication
- a reachable MLflow tracking server for `MODEL_URI=models:/...` loading
- an external Redis-backed Feast online store if you want Feast-backed online inference in Cloud Run
- a cloud-friendly Feast repo/store configuration if you move Feast online serving off the local Docker network

Where GCS and Compute Engine fit:
- GCS is a natural next step for exported datasets, parquet feature sources, or model artifacts if you move beyond the current local `/mlruns` setup
- Compute Engine is not required for this minimal inference deployment path, but it is a reasonable place to host longer-running components such as a self-managed MLflow server, Airflow stack, or Redis instance for a thesis demo















for a thesis demo




