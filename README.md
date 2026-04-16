# MLOps-RS

Thesis-oriented MLOps system for purchase probability prediction and top-N product recommendation on RetailRocket-style interaction data.

The repository is strongest as a local, containerized end-to-end demo. It also includes a minimal Google Cloud deployment path for the inference service, but it does not yet automate the full cloud-native stack.

## Thesis Story

Business goal:
- predict the probability that a user will purchase a given item within a future horizon
- recommend the top-N items with the highest predicted purchase probability

Implemented pipeline:
1. preprocess raw interaction events into cleaned parquet data
2. build a leakage-aware future-purchase dataset with time-based train/validation/test splits
3. define the canonical model feature contract in `feature_repo/`
4. build offline feature files and materialize online features to Redis through Feast
5. train an XGBoost baseline and optionally tune with Optuna
6. track experiments and register/promote models with MLflow + PostgreSQL
7. orchestrate the local pipeline with Airflow
8. serve `/predict_proba` and `/recommend` through FastAPI
9. expose API metrics to Prometheus and Grafana
10. demonstrate a local realtime update path from Kafka to Redis
11. optionally deploy the FastAPI service container to Cloud Run

## What Is Actually Done

Done:
- data preprocessing and dataset construction in `data/scripts/`
- canonical feature definitions in `feature_repo/feature_definitions.py`
- Feast repo for online feature materialization in `feature_repo/`
- XGBoost baseline training in `training/train_xgb_baseline.py`
- Optuna tuning in `training/tune_xgb_optuna.py`
- MLflow registration and promotion in `training/promote_model.py`
- Airflow DAG for local orchestration in `infra/airflow/dags/retailrocket_baseline_pipeline.py`
- FastAPI prediction and recommendation endpoints in `serving/app.py`
- top-N recommendation via lightweight retrieval + purchase-probability ranking in `serving/recommendation.py`
- unified serving entrypoint `serving.app:app` in `serving/Dockerfile`
- Prometheus/Grafana local monitoring in `infra/monitoring/`
- Docker Compose local stack in `infra/docker-compose.yml`
- GitHub Actions CI plus manual train/deploy workflows in `.github/workflows/`
- minimal Cloud Run deployment files in `deploy/gcp/`

Partial:
- realtime updates are implemented as a Kafka consumer process in the Flink-stage container, not as a submitted PyFlink/Flink job
- monitoring covers API/inference metrics, not full drift, data quality, or feedback-loop monitoring
- Cloud Run deployment covers only the inference service, not MLflow, Airflow, Feast, Redis, Kafka, or Grafana
- CI/CD is thesis-appropriate but still light: validation, manual training, and manual Cloud Run deployment rather than full environment promotion automation

Future work:
- stronger online feature parity between Feast-managed online state and realtime Redis updates
- fuller streaming semantics such as checkpointing, windowed aggregation, and production-grade fault tolerance
- richer retrieval before ranking, such as co-occurrence models, ANN search, or learned retrieval
- cloud-managed backing services and secrets management
- model/data quality monitoring beyond request-level metrics

## Repository Map

- `data/`: preprocessing and dataset generation
- `feature_repo/`: canonical feature definitions and Feast repo
- `training/`: baseline training, Optuna tuning, model promotion
- `serving/`: FastAPI inference service, recommendation logic, realtime feature reader
- `streaming/`: event schema, Kafka demo producer, local realtime processor
- `infra/`: Docker Compose stack, Airflow, MLflow, monitoring, Kafka, Flink-stage images
- `deploy/gcp/`: Cloud Run deployment script and example config
- `ui/`: Streamlit demo UI for the FastAPI service
- `docs/`: focused architecture and thesis-scope notes

## Dependency Files

- `serving/requirements.txt`: serving/runtime dependencies for FastAPI inference, online feature lookup, and monitoring
- `requirements-training.txt`: shared training/tuning dependencies for `training/train_xgb_baseline.py`, `training/tune_xgb_optuna.py`, and training-focused local or CI checks
- `infra/airflow/requirements.txt`: Airflow image dependencies; includes the training/tuning stack plus Airflow-specific extras such as Feast CLI support and PostgreSQL client packages

## Current End-to-End Flow

Offline and batch path:
1. `data/scripts/preprocess_retailrocket.py` cleans raw events into `data/processed/events_retailrocket.parquet`
2. `data/scripts/build_trainset_retailrocket.py` builds candidate rows, labels, and time-based splits
3. `feature_repo/build_online_features.py` derives online feature source files from processed events
4. `feast apply` and `feast materialize-incremental` load those features into Redis
5. `training/train_xgb_baseline.py` trains the baseline model and logs to MLflow
6. `training/tune_xgb_optuna.py` optionally tunes and logs a stronger model
7. `training/promote_model.py` registers and promotes a selected model in MLflow
8. `serving.app:app` loads the promoted model for inference

Realtime demo path:
1. `streaming/producer/simulate_events.py` publishes demo user events to Kafka
2. `streaming/flink/realtime_feature_job.py` consumes those events and updates Redis hashes
3. `serving.app` reads Redis realtime features first during inference
4. if realtime Redis data is unavailable, the API falls back to Feast online features
5. if neither online source is available, the API can still score fully specified manual feature payloads

## Serving Architecture

Runtime entrypoint:
- the deployed/local container starts `uvicorn serving.app:app`
- `serving.app` owns the full serving flow: model loading, feature resolution, recommendation, and metrics
- the canonical Feast repository for serving and training is the top-level `feature_repo/`
- `serving/realtime_features.py` is a helper for reading realtime Redis hashes
- `serving/realtime_app.py` is kept only as a thin compatibility shim for older imports and is not the primary runtime module

Prediction path:
- `/predict_proba` accepts either `user_id + item_id` or a full manual feature row
- when `user_id + item_id` are provided, runtime resolution is:
  1. Redis realtime hashes from `serving/realtime_features.py`
  2. Feast online lookup from `feature_repo/`
  3. manual payload fallback if all model features are present
- returned `feature_source` is one of:
  - `redis_realtime`
  - `feast_online`
  - `manual_fallback`

Recommendation path:
- `/recommend` is a lightweight retrieval + ranking endpoint
- it can score:
  - explicit `candidate_item_ids`
  - explicit per-candidate feature payloads
  - generated candidates for a known `user_id`
- generated candidates come from `serving/recommendation.py`
- retrieval uses two simple sources that are easy to explain in a thesis defense:
  - the user's most recent unique interacted items
  - globally popular items as fallback or cold-start padding
- each candidate is then ranked by the same purchase-probability model used by `/predict_proba`
- manual candidate modes remain available as fallback when the caller wants to control the candidate set directly

## Realtime Path

What the code really does:
- Kafka is used for local event ingestion
- the process in `streaming/flink/realtime_feature_job.py` is a Python Kafka consumer that runs inside the Flink-stage container image
- that process writes simple counters and latest timestamps into Redis hashes for `user`, `item`, and `user_item`
- `serving.app` reads those Redis hashes directly for inference

What it does not currently do:
- it does not submit a PyFlink/Flink streaming job to the Flink cluster
- it does not claim advanced streaming guarantees such as checkpointing or exactly-once semantics
- it does not update Feast online features through a dedicated streaming materialization flow

This means the realtime story is valid for a local thesis demo, but it should be described as a lightweight realtime feature update path, not as a full production Flink streaming system.

## Local Containerized Stack

`infra/docker-compose.yml` provides:
- PostgreSQL for MLflow and Airflow metadata
- Redis for online features and realtime hashes
- Kafka for demo event ingestion
- Flink JobManager and TaskManager containers for stack completeness
- `realtime-processor` for the implemented realtime Redis updates
- MLflow tracking server
- Airflow standalone instance with the thesis DAG
- FastAPI inference service
- Prometheus and Grafana

Typical local demo sequence:
```bash
cd infra
docker compose up -d --build

cd ..
python3 data/scripts/preprocess_retailrocket.py
python3 data/scripts/build_trainset_retailrocket.py

cd infra
docker compose exec api bash -lc "cd /app && python -m feature_repo.build_online_features"
docker compose exec api bash -lc "cd /app/feature_repo && feast apply"
docker compose exec api bash -lc "cd /app/feature_repo && feast materialize-incremental \$(date -u +%Y-%m-%dT%H:%M:%S)"

cd ..
python3 -m training.train_xgb_baseline
python3 -m training.promote_model --run-id <RUN_ID_FROM_TRAINING_OUTPUT> --model-name xgb-baseline-retailrocket --artifact-path model --tracking-uri http://localhost:5000

curl -X POST http://localhost:8000/reload_model
curl -i http://localhost:8000/readyz
```

Smallest post-promotion reload helper:
```bash
cd ..
python3 scripts/promote_and_reload_model.py \
  --run-id <RUN_ID_FROM_TRAINING_OUTPUT> \
  --model-name xgb-baseline-retailrocket \
  --artifact-path model \
  --tracking-uri http://localhost:5000 \
  --reload-url http://localhost:8000/reload_model
```

This helper keeps promotion and serving loosely coupled:
- it runs `python3 -m training.promote_model`
- then it calls `POST /reload_model`
- it exits nonzero if promotion fails or if the API reload call fails

Readiness notes:
- `GET /healthz` is liveness only and returns `200` when the FastAPI process is up
- Docker Compose now uses `GET /readyz` for the API healthcheck
- `GET /readyz` returns non-200 until the model is actually loaded, so the `api` container can stay `unhealthy` after startup until you promote a model and call `/reload_model`

Airflow reset and startup notes:
- Airflow metadata uses its own PostgreSQL database: `postgresql+psycopg2://airflow:airflow@postgres:5432/airflow`
- If your `pgdata` volume was created before `infra/postgres/init/01-create-airflow-db.sql` existed, the `airflow` role/database will not be created retroactively. Reset the local Postgres volume once before restarting Airflow:

```bash
cd infra
docker compose down -v postgres airflow
docker volume rm infra_pgdata infra_airflow_home 2>/dev/null || true
docker compose up -d --build postgres airflow
docker compose logs -f airflow
```

- The Airflow container now waits for DNS resolution and a real connection to the dedicated `airflow` database before running `airflow db migrate` and `airflow standalone`.

Optional realtime demo:
```bash
cd infra
docker compose up -d --build kafka flink-jobmanager flink-taskmanager realtime-processor api

cd ..
python3 streaming/producer/simulate_events.py --bootstrap-servers localhost:29092 --topic user_events --num-events 5
python3 scripts/verify_realtime_redis.py --redis-host localhost --user-id 1 --item-id 101
```

Recommendation evaluation:
```bash
cd ..
python3 scripts/evaluate_recommendation.py \
  --mlflow-tracking-uri http://localhost:5000 \
  --model-uri models:/xgb-baseline-retailrocket/Production \
  --ks 5,10,20
```

## Minimal Cloud Deployment Path

Implemented cloud path:
- `deploy/gcp/deploy_cloud_run.sh` builds the serving image with Cloud Build
- pushes it to Artifact Registry
- deploys it to Cloud Run
- `.github/workflows/deploy-cloud-run.yml` can trigger that path manually

Scope of this cloud path:
- it deploys the FastAPI inference service only
- it does not deploy MLflow, Airflow, Redis, Feast, Kafka, Prometheus, or Grafana
- it assumes any required backing services already exist and are reachable

External or manual cloud dependencies:
- GCP project, IAM, billing, and `gcloud` authentication
- Artifact Registry and Cloud Run permissions
- reachable `MLFLOW_TRACKING_URI` if the service loads models from MLflow
- reachable Redis-backed online store if Feast-backed online inference is required
- cloud-appropriate Feast configuration instead of the local `redis:6379`
- any separate storage or compute services you may choose to use, such as GCS or Compute Engine

## GitHub Actions

Implemented workflows:
- `ci.yml`: compile/import/syntax validation for the main Python entrypoints
- `manual-train.yml`: rebuild local data and run baseline training, with optional Optuna tuning
- `deploy-cloud-run.yml`: manual Cloud Run deployment workflow

This is enough to support a thesis demo and basic reproducibility, but it is not yet a full multi-environment CI/CD system.

## Monitoring

Implemented:
- FastAPI request and inference metrics exposed at `/metrics`
- Prometheus scrape configuration in `infra/monitoring/prometheus/prometheus.yml`
- Grafana provisioning in `infra/monitoring/grafana/`

Not yet implemented:
- drift detection
- automated retraining triggers from monitoring signals
- production feedback-loop monitoring

## Defense-Friendly Summary

The repository demonstrates an end-to-end MLOps pipeline with honest scope boundaries:
- strong local containerized pipeline for data, features, training, registry, serving, and demo monitoring
- working top-N recommendation through lightweight retrieval plus purchase-probability ranking
- working realtime demo path from Kafka to Redis to inference
- minimal but real Cloud Run deployment path for the serving layer
- clear remaining gaps around fully managed cloud services, stronger streaming semantics, and broader CI/CD

See [docs/realtime_architecture.md](/home/truc/mlops-rs/docs/realtime_architecture.md) and [docs/thesis_scope.md](/home/truc/mlops-rs/docs/thesis_scope.md) for the concise defense-ready architecture and scope notes.
