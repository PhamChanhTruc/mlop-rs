from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote an MLflow model and then trigger the serving API reload endpoint.",
    )
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
    parser.add_argument(
        "--api-base-url",
        default="http://localhost:8000",
        help="Base serving API URL. Used with /reload_model when --reload-url is not provided.",
    )
    parser.add_argument(
        "--reload-url",
        default=None,
        help="Serving API reload endpoint.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=30.0,
        help="Timeout in seconds for the reload API call.",
    )
    parser.add_argument(
        "--reload-max-attempts",
        type=int,
        default=3,
        help="Maximum reload attempts when the API cannot be reached.",
    )
    parser.add_argument(
        "--reload-retry-delay-sec",
        type=float,
        default=2.0,
        help="Delay between reload retries after connection failures.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used to run training.promote_model.",
    )
    return parser.parse_args()


def run_promotion(args: argparse.Namespace) -> None:
    command = [
        args.python_bin,
        "-m",
        "training.promote_model",
        "--run-id",
        args.run_id,
        "--model-name",
        args.model_name,
        "--artifact-path",
        args.artifact_path,
        "--tracking-uri",
        args.tracking_uri,
        "--reload-api",
        "--reload-timeout-sec",
        str(args.timeout_sec),
        "--reload-max-attempts",
        str(args.reload_max_attempts),
        "--reload-retry-delay-sec",
        str(args.reload_retry_delay_sec),
    ]
    if args.reload_url:
        command.extend(["--reload-url", args.reload_url])
    elif args.api_base_url:
        command.extend(["--api-base-url", args.api_base_url])

    print("Promoting model and reloading the serving API...")
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)


def main() -> None:
    args = parse_args()

    try:
        run_promotion(args)
    except subprocess.CalledProcessError as exc:
        print("Promotion and reload failed.", file=sys.stderr)
        if exc.stdout:
            print(exc.stdout.strip(), file=sys.stderr)
        if exc.stderr:
            print(exc.stderr.strip(), file=sys.stderr)
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
