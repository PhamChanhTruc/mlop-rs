from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import mlflow
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.operators.empty import EmptyOperator
from mlflow.tracking import MlflowClient

REPO_ROOT = Path("/opt/mlops-rs")
FEATURE_REPO_ROOT = REPO_ROOT / "feature_repo"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
BASELINE_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "xgb-baseline-retailrocket")
TUNING_EXPERIMENT = os.getenv("AIRFLOW_TUNING_EXPERIMENT", "xgb-optuna-retailrocket")
BASELINE_MODEL_NAME = os.getenv("AIRFLOW_MODEL_NAME", "xgb-baseline-retailrocket")
TUNED_MODEL_NAME = os.getenv("AIRFLOW_TUNED_MODEL_NAME", "xgb-optuna-retailrocket")
MODEL_ARTIFACT_PATH = os.getenv("AIRFLOW_MODEL_ARTIFACT_PATH", "model")
TUNED_MODEL_ARTIFACT_PATH = os.getenv("AIRFLOW_TUNED_MODEL_ARTIFACT_PATH", "best_model_train_val")
ENABLE_OPTUNA_TUNING = os.getenv("AIRFLOW_ENABLE_OPTUNA_TUNING", "false").lower() == "true"
OPTUNA_TRIALS = int(os.getenv("AIRFLOW_OPTUNA_TRIALS", "30"))
OPTUNA_TIMEOUT_SEC = os.getenv("AIRFLOW_OPTUNA_TIMEOUT_SEC")
PROMOTE_TUNED_IF_AVAILABLE = os.getenv("AIRFLOW_PROMOTE_TUNED_IF_AVAILABLE", "false").lower() == "true"
MIN_BASELINE_VAL_PR_AUC = float(os.getenv("AIRFLOW_MIN_BASELINE_VAL_PR_AUC", "0.0"))
MIN_TUNED_FINAL_TEST_PR_AUC = float(os.getenv("AIRFLOW_MIN_TUNED_FINAL_TEST_PR_AUC", "0.0"))
RELOAD_API_AFTER_PROMOTION = os.getenv("AIRFLOW_RELOAD_API_AFTER_PROMOTION", "true").lower() == "true"
AIRFLOW_API_BASE_URL = os.getenv("AIRFLOW_API_BASE_URL", "http://api:8000")
AIRFLOW_API_RELOAD_URL = os.getenv("AIRFLOW_API_RELOAD_URL")
AIRFLOW_API_RELOAD_TIMEOUT_SEC = os.getenv("AIRFLOW_API_RELOAD_TIMEOUT_SEC", "10")
AIRFLOW_API_RELOAD_MAX_ATTEMPTS = os.getenv("AIRFLOW_API_RELOAD_MAX_ATTEMPTS", "3")
AIRFLOW_API_RELOAD_RETRY_DELAY_SEC = os.getenv("AIRFLOW_API_RELOAD_RETRY_DELAY_SEC", "2")


def _repo_env() -> dict[str, str]:
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    return env


def run_repo_script(*args: str, cwd: Path = REPO_ROOT, extra_env: Optional[dict[str, str]] = None) -> None:
    env = _repo_env()
    if extra_env:
        env.update(extra_env)
    subprocess.run(["python3", *args], cwd=cwd, env=env, check=True)


def run_repo_module(module_name: str, *args: str, extra_env: Optional[dict[str, str]] = None) -> None:
    run_repo_script("-m", module_name, *args, extra_env=extra_env)


def run_feature_repo_command(*args: str) -> None:
    subprocess.run(args, cwd=FEATURE_REPO_ROOT, env=_repo_env(), check=True)


def latest_run_info(experiment_name: str) -> Any:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise AirflowFailException(f"MLflow experiment not found: {experiment_name}")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise AirflowFailException(f"No MLflow runs found in experiment: {experiment_name}")
    return runs[0]


def build_model_candidate(
    run_info: Any,
    *,
    mode: str,
    experiment_name: str,
    model_name: str,
    artifact_path: str,
    promotion_metric_name: str,
    minimum_metric: float,
) -> dict[str, Any]:
    metric_value = run_info.data.metrics.get(promotion_metric_name)
    if metric_value is None:
        raise AirflowFailException(
            f"Run {run_info.info.run_id} in experiment {experiment_name} is missing metric "
            f"'{promotion_metric_name}' required for promotion"
        )
    return {
        "mode": mode,
        "run_id": run_info.info.run_id,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "artifact_path": artifact_path,
        "promotion_metric_name": promotion_metric_name,
        "promotion_metric_value": float(metric_value),
        "minimum_metric": float(minimum_metric),
    }


with DAG(
    dag_id="retailrocket_training_feature_pipeline",
    description="Local thesis-aligned orchestration for preprocessing, Feast materialization, training, tuning, validation, and promotion",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "retailrocket", "airflow", "training", "feast"],
) as dag:
    start = EmptyOperator(task_id="start")
    finish = EmptyOperator(task_id="finish")

    @task(task_id="preprocess_events")
    def preprocess_events() -> None:
        run_repo_script("data/scripts/preprocess_retailrocket.py")

    @task(task_id="build_trainset")
    def build_trainset() -> None:
        run_repo_script("data/scripts/build_trainset_retailrocket.py")

    @task(task_id="build_online_feature_sources")
    def build_online_feature_sources() -> None:
        run_repo_script("build_online_features.py", cwd=FEATURE_REPO_ROOT)

    @task(task_id="feast_apply")
    def feast_apply() -> None:
        run_feature_repo_command("feast", "apply")

    @task(task_id="feast_materialize_online_features")
    def feast_materialize_online_features() -> str:
        end_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        run_feature_repo_command("feast", "materialize-incremental", end_ts)
        return end_ts

    @task(task_id="train_baseline_model")
    def train_baseline_model() -> dict[str, Any]:
        run_repo_module(
            "training.train_xgb_baseline",
            extra_env={"MLFLOW_EXPERIMENT": BASELINE_EXPERIMENT},
        )
        run_info = latest_run_info(BASELINE_EXPERIMENT)
        return build_model_candidate(
            run_info,
            mode="baseline",
            experiment_name=BASELINE_EXPERIMENT,
            model_name=BASELINE_MODEL_NAME,
            artifact_path=MODEL_ARTIFACT_PATH,
            promotion_metric_name="val_pr_auc",
            minimum_metric=MIN_BASELINE_VAL_PR_AUC,
        )

    @task(task_id="tune_optuna_model")
    def tune_optuna_model() -> Optional[dict[str, Any]]:
        if not ENABLE_OPTUNA_TUNING:
            return None

        args = [
            "--n-trials",
            str(OPTUNA_TRIALS),
            "--tracking-uri",
            MLFLOW_TRACKING_URI,
            "--experiment",
            TUNING_EXPERIMENT,
        ]
        if OPTUNA_TIMEOUT_SEC:
            args.extend(["--timeout-sec", OPTUNA_TIMEOUT_SEC])

        run_repo_module(
            "training.tune_xgb_optuna",
            *args,
            extra_env={"MLFLOW_EXPERIMENT": TUNING_EXPERIMENT},
        )
        run_info = latest_run_info(TUNING_EXPERIMENT)
        return build_model_candidate(
            run_info,
            mode="tuned",
            experiment_name=TUNING_EXPERIMENT,
            model_name=TUNED_MODEL_NAME,
            artifact_path=TUNED_MODEL_ARTIFACT_PATH,
            promotion_metric_name="final_test_pr_auc",
            minimum_metric=MIN_TUNED_FINAL_TEST_PR_AUC,
        )

    @task(task_id="select_model_for_promotion")
    def select_model_for_promotion(
        baseline_candidate: dict[str, Any], tuned_candidate: Optional[dict[str, Any]]
    ) -> dict[str, Any]:
        if PROMOTE_TUNED_IF_AVAILABLE and tuned_candidate is not None:
            return tuned_candidate
        return baseline_candidate

    @task(task_id="validate_model_candidate")
    def validate_model_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        metric_value = float(candidate["promotion_metric_value"])
        minimum_metric = float(candidate["minimum_metric"])
        if metric_value < minimum_metric:
            raise AirflowFailException(
                f"Candidate run {candidate['run_id']} failed validation: "
                f"{candidate['promotion_metric_name']}={metric_value:.6f} < {minimum_metric:.6f}"
            )

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        artifact_uri = f"runs:/{candidate['run_id']}/{candidate['artifact_path']}"
        downloaded_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
        candidate["downloaded_artifact_path"] = downloaded_path
        return candidate

    @task(task_id="promote_model")
    def promote_model(candidate: dict[str, Any]) -> None:
        args = [
            "--run-id",
            candidate["run_id"],
            "--model-name",
            candidate["model_name"],
            "--artifact-path",
            candidate["artifact_path"],
            "--tracking-uri",
            MLFLOW_TRACKING_URI,
        ]
        if RELOAD_API_AFTER_PROMOTION:
            args.append("--reload-api")
            if AIRFLOW_API_RELOAD_URL:
                args.extend(["--reload-url", AIRFLOW_API_RELOAD_URL])
            else:
                args.extend(["--api-base-url", AIRFLOW_API_BASE_URL])
            args.extend(
                [
                    "--reload-timeout-sec",
                    AIRFLOW_API_RELOAD_TIMEOUT_SEC,
                    "--reload-max-attempts",
                    AIRFLOW_API_RELOAD_MAX_ATTEMPTS,
                    "--reload-retry-delay-sec",
                    AIRFLOW_API_RELOAD_RETRY_DELAY_SEC,
                ]
            )
        run_repo_module("training.promote_model", *args)

    preprocess = preprocess_events()
    dataset = build_trainset()
    online_features = build_online_feature_sources()
    applied = feast_apply()
    materialized = feast_materialize_online_features()
    baseline_candidate = train_baseline_model()
    tuned_candidate = tune_optuna_model()
    selected_candidate = select_model_for_promotion(baseline_candidate, tuned_candidate)
    validated_candidate = validate_model_candidate(selected_candidate)
    promoted = promote_model(validated_candidate)

    start >> preprocess >> dataset
    dataset >> online_features >> applied >> materialized >> finish
    dataset >> baseline_candidate >> tuned_candidate >> selected_candidate
    baseline_candidate >> selected_candidate
    selected_candidate >> validated_candidate >> promoted >> finish
