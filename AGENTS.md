# AGENTS.md

## Project title
Design and implementation of an end-to-end cloud-native MLOps pipeline for purchase probability prediction and product recommendation.

## Thesis goal
This repository should implement a cloud-native end-to-end MLOps pipeline for:
1. data processing
2. feature management
3. model training and hyperparameter tuning
4. experiment tracking and model registry
5. production model serving
6. real-time monitoring
7. cloud deployment
8. CI/CD automation

Business problem:
Given a user's historical interactions with items (click, add-to-cart, favorite, purchase, etc.), the system should:
- predict the probability that a given user will buy a given item at a target time
- recommend the top-N items that the user is most likely to purchase

## Required tech stack
- Feast + Redis: offline/online feature management
- MLflow + PostgreSQL: experiment tracking and model registry
- Apache Airflow + PostgreSQL: pipeline orchestration
- XGBoost + Optuna: model training and hyperparameter tuning
- FastAPI: serving endpoints
- Prometheus + Grafana: monitoring
- Docker + Docker Compose: local containerized stack
- Google Cloud: Cloud Run, Artifact Registry, Compute Engine, Storage Bucket
- GitHub Actions: CI/CD

## What Codex must understand first
Before making changes:
1. read README.md
2. inspect all top-level folders
3. identify the current end-to-end flow
4. compare the implementation against the thesis goal
5. explicitly classify what is already done, partial, missing, or extra
6. avoid assuming that local-only functionality is enough for the thesis goal

## Expected thesis-aligned architecture
The intended architecture is:

1. raw interaction data ingestion and preprocessing
2. dataset construction for future purchase prediction
3. feature definition and management through Feast
4. offline features for training and online features from Redis for serving
5. XGBoost training and Optuna tuning
6. experiment tracking and model registration in MLflow
7. pipeline orchestration with Airflow
8. FastAPI inference service for probability prediction and recommendation
9. Prometheus/Grafana monitoring
10. containerized local stack with Docker Compose
11. deployment path to Google Cloud
12. CI/CD automation with GitHub Actions

## Repository responsibilities
### data/
- preprocessing scripts
- train/validation/test dataset generation
- label construction for future purchase prediction
- any time-based split logic should be preserved carefully

### feature_repo/
- source of truth for feature definitions
- should represent the feature contract between training and serving
- Codex must check offline/online feature parity
- avoid changes that break feature consistency silently

### training/
- baseline model training
- Optuna hyperparameter tuning
- MLflow tracking and model registration/promotion
- preserve reproducibility, metric names, and MLflow logging conventions

### serving/
- FastAPI inference service
- should support purchase probability prediction
- should support top-N recommendation
- online inference should prefer Feast/Redis-backed features instead of requiring all precomputed features directly in the request
- keep endpoint contracts stable unless explicitly asked to change them

### infra/
- local stack definitions
- Docker Compose services
- MLflow, PostgreSQL, Redis, monitoring, and Airflow-related infrastructure
- explain infra impact before making broad changes

### deploy/ or cloud-related files
- Cloud Run deployment path
- Artifact Registry push/build flow
- GCS usage for artifacts or related assets
- Compute Engine usage if present

### .github/workflows/
- CI validation
- deployment automation
- should remain simple and thesis-appropriate

## Recommendation system scope
The recommendation goal is not only binary prediction.
The project should produce top-N items with the highest predicted purchase probability.

Acceptable staged implementation:
- minimum: rerank a candidate set using purchase probability
- better: lightweight candidate generation + ranking
Codex should clearly state which stage is currently implemented.

## Review rules
When reviewing the repository against the thesis goal, classify findings into:
- done
- partial
- missing
- extra

For each finding:
- cite exact files
- explain why it belongs to that category
- propose the smallest correct fix first

## Constraints
- preserve WSL-friendly development
- prefer minimal, targeted changes
- avoid unnecessary overengineering
- do not silently replace the declared tech stack with a different one
- do not describe local-only scripts as a complete cloud-native pipeline unless the cloud and CI/CD parts are actually implemented

## Documentation rules
When updating docs, Codex should:
- distinguish clearly between implemented features and planned/future work
- avoid overstating incomplete parts
- make the architecture easy to explain in a thesis defense