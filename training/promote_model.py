import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

from mlflow import MlflowException
from mlflow.tracking import MlflowClient


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    reload_group = parser.add_mutually_exclusive_group()
    reload_group.add_argument(
        "--reload-api",
        dest="reload_api",
        action="store_true",
        help="Call the serving API reload endpoint after successful promotion.",
    )
    reload_group.add_argument(
        "--no-reload-api",
        dest="reload_api",
        action="store_false",
        help="Skip the serving API reload step even if PROMOTE_RELOAD_API=true.",
    )
    parser.set_defaults(reload_api=_env_flag("PROMOTE_RELOAD_API", False))
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("PROMOTE_API_BASE_URL"),
        help="Base serving API URL. Used with /reload_model when --reload-url is not provided.",
    )
    parser.add_argument(
        "--reload-url",
        default=os.getenv("PROMOTE_RELOAD_URL"),
        help="Full serving API reload endpoint. Overrides --api-base-url when set.",
    )
    parser.add_argument(
        "--reload-timeout-sec",
        type=float,
        default=float(os.getenv("PROMOTE_RELOAD_TIMEOUT_SEC", "10")),
        help="Timeout in seconds for each reload API call.",
    )
    parser.add_argument(
        "--reload-max-attempts",
        type=int,
        default=int(os.getenv("PROMOTE_RELOAD_MAX_ATTEMPTS", "3")),
        help="Maximum reload attempts when the API cannot be reached.",
    )
    parser.add_argument(
        "--reload-retry-delay-sec",
        type=float,
        default=float(os.getenv("PROMOTE_RELOAD_RETRY_DELAY_SEC", "2")),
        help="Delay between reload retries after connection failures.",
    )
    return parser.parse_args()


def _promote_model(args: argparse.Namespace) -> str:
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
    return f"models:/{args.model_name}/Production"


def _resolve_reload_url(args: argparse.Namespace) -> str:
    if args.reload_url:
        return args.reload_url
    if args.api_base_url:
        return f"{args.api_base_url.rstrip('/')}/reload_model"
    raise ValueError(
        "Promotion reload was requested, but no API endpoint was configured. "
        "Set --reload-url or --api-base-url, or PROMOTE_RELOAD_URL/PROMOTE_API_BASE_URL."
    )


def _call_reload_endpoint(reload_url: str, timeout_sec: float) -> tuple[int, str, dict]:
    request = urllib.request.Request(
        reload_url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        body = response.read().decode("utf-8", errors="replace")
        payload = json.loads(body) if body else {}
        return response.status, body, payload


def _reload_api(args: argparse.Namespace) -> dict:
    if args.reload_timeout_sec <= 0:
        raise ValueError("--reload-timeout-sec must be greater than 0")
    if args.reload_max_attempts < 1:
        raise ValueError("--reload-max-attempts must be at least 1")
    if args.reload_retry_delay_sec < 0:
        raise ValueError("--reload-retry-delay-sec must be non-negative")

    reload_url = _resolve_reload_url(args)
    for attempt in range(1, args.reload_max_attempts + 1):
        print(f"Reloading serving API via {reload_url} (attempt {attempt}/{args.reload_max_attempts})")
        try:
            status, _, payload = _call_reload_endpoint(reload_url, args.reload_timeout_sec)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = (
                f"Promotion succeeded, but API reload failed with HTTP {exc.code} at {reload_url}."
            )
            if body:
                message = f"{message} Response body: {body}"
            raise RuntimeError(message) from exc
        except urllib.error.URLError as exc:
            if attempt >= args.reload_max_attempts:
                raise RuntimeError(
                    f"Promotion succeeded, but API reload request could not reach {reload_url} "
                    f"after {attempt} attempt(s): {exc}"
                ) from exc
            print(
                f"Reload attempt {attempt}/{args.reload_max_attempts} could not reach {reload_url}: {exc}. "
                f"Retrying in {args.reload_retry_delay_sec} seconds..."
            )
            time.sleep(args.reload_retry_delay_sec)
            continue
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Promotion succeeded, but API reload response from {reload_url} was not valid JSON: {exc}"
            ) from exc

        print(f"Reload HTTP {status}")
        print(json.dumps(payload, indent=2, sort_keys=True))
        if payload.get("reloaded") is True:
            return payload
        raise RuntimeError(
            "Promotion succeeded, but reload response did not confirm success: "
            f"{json.dumps(payload, sort_keys=True)}"
        )

    raise RuntimeError("Promotion succeeded, but the API reload loop ended unexpectedly")


def main() -> None:
    args = parse_args()

    try:
        _promote_model(args)
    except Exception as exc:
        print(f"Promotion failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if not args.reload_api:
        return

    try:
        payload = _reload_api(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    loaded_model_uri = payload.get("model_uri_loaded")
    print(
        "Promotion and reload succeeded."
        f"{f' Serving model: {loaded_model_uri}' if loaded_model_uri else ''}"
    )


if __name__ == "__main__":
    main()
