from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request


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
        "--reload-url",
        default="http://localhost:8000/reload_model",
        help="Serving API reload endpoint.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=30.0,
        help="Timeout in seconds for the reload API call.",
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
    ]

    print("[1/2] Promoting model in MLflow...")
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


def call_reload(args: argparse.Namespace) -> dict:
    print(f"[2/2] Calling reload endpoint: {args.reload_url}")
    request = urllib.request.Request(
        args.reload_url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=args.timeout_sec) as response:
        body = response.read().decode("utf-8")
        payload = json.loads(body) if body else {}
        print(f"Reload HTTP {response.status}")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload


def main() -> None:
    args = parse_args()

    try:
        run_promotion(args)
    except subprocess.CalledProcessError as exc:
        print("Promotion failed.", file=sys.stderr)
        if exc.stdout:
            print(exc.stdout.strip(), file=sys.stderr)
        if exc.stderr:
            print(exc.stderr.strip(), file=sys.stderr)
        raise SystemExit(exc.returncode) from exc

    try:
        payload = call_reload(args)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(
            f"Promotion succeeded, but reload failed with HTTP {exc.code} at {args.reload_url}.",
            file=sys.stderr,
        )
        if body:
            print(body, file=sys.stderr)
        raise SystemExit(1) from exc
    except urllib.error.URLError as exc:
        print(
            f"Promotion succeeded, but reload request could not reach {args.reload_url}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    loaded_model_uri = payload.get("model_uri_loaded")
    if payload.get("reloaded") is True:
        print(
            "Promotion and reload succeeded."
            f"{f' Serving model: {loaded_model_uri}' if loaded_model_uri else ''}"
        )
        return

    print(
        "Promotion succeeded, but reload response did not confirm success.",
        file=sys.stderr,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
