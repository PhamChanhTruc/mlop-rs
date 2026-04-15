# Thesis Scope And Limitations

## Scope Positioning

The repository should be defended as an end-to-end MLOps thesis prototype with:
- a strong local containerized implementation
- a coherent feature, training, registry, serving, and monitoring story
- a lightweight but working realtime demo path
- a minimal cloud deployment path for the inference service

It should not be defended as a fully cloud-managed production platform.

## Done / Partial / Future Work

| Area | Status | Notes |
| --- | --- | --- |
| Data preprocessing and time-based dataset creation | Done | `data/scripts/preprocess_retailrocket.py`, `data/scripts/build_trainset_retailrocket.py` |
| Canonical model feature contract | Done | `feature_repo/feature_definitions.py` |
| Feast + Redis online feature materialization | Done | `feature_repo/build_online_features.py`, `feature_repo/repo.py`, `feature_repo/feature_store.yaml` |
| XGBoost baseline training | Done | `training/train_xgb_baseline.py` |
| Optuna hyperparameter tuning | Done | `training/tune_xgb_optuna.py` |
| MLflow tracking and model promotion | Done | `training/train_xgb_baseline.py`, `training/promote_model.py` |
| Airflow local orchestration | Done | `infra/airflow/dags/retailrocket_baseline_pipeline.py` |
| Purchase probability serving | Done | `serving/app.py`, `serving/realtime_app.py` |
| Top-N recommendation serving | Done | lightweight retrieval + ranking in `serving/recommendation.py` and `serving/app.py`; retrieval uses recent user interactions plus popular-item fallback, then ranks with the existing purchase-probability model |
| Realtime feature refresh | Partial | Kafka to Redis path works, but via Python consumer in `streaming/flink/realtime_feature_job.py`, not a submitted Flink job |
| Monitoring | Partial | Prometheus/Grafana cover API metrics, not full model/data quality monitoring |
| Cloud deployment | Partial | `deploy/gcp/` deploys the inference service only |
| CI/CD automation | Partial | `.github/workflows/` provides CI, manual training, and manual deploy workflows |
| Cloud-managed backing services | Future work | external MLflow, Redis, Feast, storage, and broader deployment automation are still manual |
| Advanced streaming semantics | Future work | no checkpointing, windowed state, or exactly-once guarantees |
| Stronger retrieval/recommendation stack | Future work | the current recommendation component is done for thesis scope, but retrieval is intentionally simple and can later be replaced with a stronger candidate-generation model |

## Minimal Defense Position

The cleanest thesis position is:

"This project implements an end-to-end MLOps pipeline primarily as a local cloud-native prototype. It covers data processing, feature management, model training and tuning, experiment tracking, model promotion, online inference, local monitoring, and a reproducible containerized stack. It also includes a minimal Cloud Run deployment path for the inference service and a lightweight realtime feature update demo. The remaining gaps are mainly in production-grade cloud operations, richer streaming guarantees, and broader CI/CD automation."

## What Not To Overclaim In Defense

- Do not say the project fully deploys the entire MLOps platform on Google Cloud.
- Do not say the realtime pipeline is a full Flink job if you are using the current implementation.
- Do not say recommendations come from a dedicated large-scale retrieval system.
- Do say the implemented recommendation stage is a lightweight retrieval + ranking pipeline.
- Do not say monitoring covers drift, fairness, or full model observability.
- Do not say CI/CD is fully automated across environments.
- Do not say Feast alone powers all online updates; realtime Redis writes are also used directly.
