from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List
import sys

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.recommendation import (
    CandidateGenerator,
    build_point_in_time_feature_rows,
    load_events_frame,
    resolve_recommendation_events_path,
)

DEFAULT_TEST_SPLIT_PATH = REPO_ROOT / "data/processed/dataset_retailrocket/test.parquet"
DEFAULT_MODEL_URI = "models:/xgb-baseline-retailrocket/Production"
DEFAULT_MLFLOW_TRACKING_URI = "http://localhost:5000"
DEFAULT_EXPERIMENT = "xgb-baseline-retailrocket"
DEFAULT_ARTIFACT_PATH = "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline evaluation for the retrieval + ranking recommendation pipeline.",
    )
    parser.add_argument(
        "--events-path",
        type=Path,
        default=REPO_ROOT / "data/processed/events_retailrocket.parquet",
        help="Processed events parquet used for retrieval history and ground truth.",
    )
    parser.add_argument(
        "--test-split-path",
        type=Path,
        default=DEFAULT_TEST_SPLIT_PATH,
        help="Test candidate parquet used to determine the evaluation cutoff.",
    )
    parser.add_argument(
        "--model-uri",
        default=DEFAULT_MODEL_URI,
        help="MLflow model URI used for ranking.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=DEFAULT_MLFLOW_TRACKING_URI,
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=DEFAULT_EXPERIMENT,
        help="Experiment name used when --fallback-to-latest-run is enabled.",
    )
    parser.add_argument(
        "--model-artifact-path",
        default=DEFAULT_ARTIFACT_PATH,
        help="Artifact path within the fallback MLflow run.",
    )
    parser.add_argument(
        "--fallback-to-latest-run",
        action="store_true",
        help="Use the latest run artifact if the configured model URI cannot be loaded.",
    )
    parser.add_argument(
        "--ks",
        default="5,10,20",
        help="Comma-separated K values for recommendation metrics.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Optional cap on the number of evaluated users for a faster notebook/demo run.",
    )
    return parser.parse_args()


def _latest_run_model_uri(
    tracking_uri: str,
    experiment_name: str,
    artifact_path: str,
) -> str | None:
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    return f"runs:/{runs[0].info.run_id}/{artifact_path}"


def load_ranking_model(args: argparse.Namespace):
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    primary_error = None
    try:
        return mlflow.xgboost.load_model(args.model_uri), args.model_uri
    except Exception as exc:
        primary_error = exc

    if args.fallback_to_latest_run:
        fallback_uri = _latest_run_model_uri(
            args.mlflow_tracking_uri,
            args.mlflow_experiment,
            args.model_artifact_path,
        )
        if fallback_uri is not None:
            return mlflow.xgboost.load_model(fallback_uri), fallback_uri

    raise RuntimeError(
        f"Could not load ranking model from {args.model_uri}: {primary_error}"
    ) from primary_error


def _metric_at_k(recommended: List[int], ground_truth: set[int], k: int) -> Dict[str, float]:
    top_k = recommended[:k]
    hits = [1 if item_id in ground_truth else 0 for item_id in top_k]
    hit_count = sum(hits)

    precision = hit_count / float(k)
    recall = hit_count / float(len(ground_truth)) if ground_truth else 0.0
    hit_rate = 1.0 if hit_count > 0 else 0.0

    precision_sum = 0.0
    hits_so_far = 0
    for idx, is_hit in enumerate(hits, start=1):
        if is_hit:
            hits_so_far += 1
            precision_sum += hits_so_far / float(idx)
    average_precision = precision_sum / float(min(len(ground_truth), k)) if ground_truth else 0.0

    return {
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"hitrate@{k}": hit_rate,
        f"map@{k}": average_precision,
    }


def _mean_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in keys
    }


def main() -> None:
    args = parse_args()
    ks = sorted({int(value.strip()) for value in args.ks.split(",") if value.strip()})
    if not ks:
        raise ValueError("At least one K value is required")

    events_path = resolve_recommendation_events_path(str(args.events_path))
    if not args.test_split_path.exists():
        raise FileNotFoundError(
            f"Missing {args.test_split_path}. Run data/scripts/build_trainset_retailrocket.py first."
        )

    model, loaded_model_uri = load_ranking_model(args)

    events = load_events_frame(events_path)
    test_split = pd.read_parquet(args.test_split_path, columns=["event_ts"])
    test_split["event_ts"] = pd.to_datetime(test_split["event_ts"], utc=True, errors="coerce")
    test_split = test_split.dropna(subset=["event_ts"])
    if test_split.empty:
        raise ValueError(f"No valid timestamps found in {args.test_split_path}")

    eval_start = test_split["event_ts"].min()
    eval_end = test_split["event_ts"].max()

    history = events[events["event_ts"] < eval_start].copy()
    future_transactions = events[
        (events["event_ts"] >= eval_start)
        & (events["event_ts"] <= eval_end)
        & (events["event_type"] == "transaction")
    ].copy()
    future_transactions = future_transactions.drop_duplicates(["user_id", "item_id"])

    if history.empty:
        raise ValueError("No historical events found before the test cutoff")
    if future_transactions.empty:
        raise ValueError("No future transaction labels found in the evaluation window")

    generator = CandidateGenerator.from_events(history, events_path=events_path)
    truth_by_user = future_transactions.groupby("user_id")["item_id"].apply(
        lambda s: {int(item_id) for item_id in s.tolist()}
    )

    if args.max_users is not None:
        truth_by_user = truth_by_user.head(args.max_users)

    metric_rows_by_k: Dict[int, List[Dict[str, float]]] = {k: [] for k in ks}
    users_evaluated = 0
    max_k = max(ks)

    for user_id, ground_truth in truth_by_user.items():
        candidate_item_ids = generator.generate_for_user(int(user_id), max_k)
        if not candidate_item_ids:
            continue

        feature_rows = build_point_in_time_feature_rows(
            history,
            user_id=int(user_id),
            item_ids=candidate_item_ids,
            prediction_time=eval_start,
            is_addtocart=0,
        )
        scores = model.predict_proba(feature_rows)[:, 1]
        ranked = [
            item_id
            for item_id, _ in sorted(
                zip(candidate_item_ids, scores),
                key=lambda pair: pair[1],
                reverse=True,
            )
        ]

        users_evaluated += 1
        for k in ks:
            metric_rows_by_k[k].append(_metric_at_k(ranked, ground_truth, k))

    if users_evaluated == 0:
        raise ValueError("No users could be evaluated with the current retrieval history and ground truth")

    print("Recommendation offline evaluation")
    print("retrieval_strategy=recent_user_interactions + popular_items_fallback")
    print("ranking_strategy=existing_purchase_probability_model")
    print(f"events_path={events_path}")
    print(f"test_split_path={args.test_split_path}")
    print(f"ranking_model_uri={loaded_model_uri}")
    print(f"evaluation_window_start={eval_start.isoformat()}")
    print(f"evaluation_window_end={eval_end.isoformat()}")
    print(f"users_with_ground_truth={len(truth_by_user)}")
    print(f"users_evaluated={users_evaluated}")
    print(f"candidate_pool_limit={generator.candidate_pool_limit(max_k)}")

    for k in ks:
        summary = _mean_metrics(metric_rows_by_k[k])
        metric_line = " ".join(
            f"{metric_name}={metric_value:.4f}"
            for metric_name, metric_value in summary.items()
        )
        print(f"K={k} {metric_line}")


if __name__ == "__main__":
    main()
