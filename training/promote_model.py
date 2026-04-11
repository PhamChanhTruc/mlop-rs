import argparse

from mlflow import MlflowException
from mlflow.tracking import MlflowClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register and promote an MLflow run model to Production.")
    parser.add_argument("--run-id", required=True, help="Run ID containing the logged model artifact.")
    parser.add_argument(
        "--model-name",
        default="xgb-optuna-retailrocket",
        help="Registered model name.",
    )
    parser.add_argument(
        "--artifact-path",
        default="best_model_train_val",
        help="Artifact path inside the run (e.g. model, best_model_train_val).",
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking URI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = MlflowClient(tracking_uri=args.tracking_uri)

    try:
        client.create_registered_model(args.model_name)
        print(f"Created registered model: {args.model_name}")
    except MlflowException:
        print(f"Registered model already exists: {args.model_name}")

    source = f"runs:/{args.run_id}/{args.artifact_path}"
    mv = client.create_model_version(name=args.model_name, source=source, run_id=args.run_id)
    print(f"Created model version: {args.model_name} v{mv.version}")

    # Keep compatibility with existing API URI format: models:/<name>/Production
    client.transition_model_version_stage(
        name=args.model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Promoted to Production: {args.model_name} v{mv.version}")
    print(f"Model URI: models:/{args.model_name}/Production")


if __name__ == "__main__":
    main()
