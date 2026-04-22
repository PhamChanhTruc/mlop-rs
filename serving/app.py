import os
import logging
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from feature_repo.feature_definitions import MODEL_FEATURES
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

from serving.realtime_features import RealtimeFeatureReader
from serving.recommendation import (
    CandidateGenerator,
    RecommendationCandidate,
    resolve_recommendation_candidates,
    resolve_recommendation_artifact_path,
    resolve_recommendation_events_path,
)

try:
    from feast import FeatureStore
except Exception:
    FeatureStore = None

FEATURES = MODEL_FEATURES
LOGGER = logging.getLogger("serving.app")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_URI = os.getenv("MODEL_URI", "models:/xgb-baseline-retailrocket/Production")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "xgb-baseline-retailrocket")
MODEL_ARTIFACT_PATH = os.getenv("MODEL_ARTIFACT_PATH", "model")
FALLBACK_TO_LATEST_RUN = os.getenv("MODEL_URI_FALLBACK_TO_LATEST_RUN", "false").lower() == "true"
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "/app/feature_repo")
RECOMMENDATION_CANDIDATE_ARTIFACT_PATH = os.getenv(
    "RECOMMENDATION_CANDIDATE_ARTIFACT_PATH",
    "/app/data/processed/recommendation_candidates.json.gz",
)
RECOMMENDATION_CANDIDATE_ALLOW_PARQUET_FALLBACK = (
    os.getenv("RECOMMENDATION_CANDIDATE_ALLOW_PARQUET_FALLBACK", "false").lower() == "true"
)
RECOMMENDATION_EVENTS_PATH = os.getenv(
    "RECOMMENDATION_EVENTS_PATH",
    "/app/data/processed/events_retailrocket.parquet",
)
ONLINE_FEATURE_REFS = [
    "user_stats:user_event_count_prev",
    "item_stats:item_event_count_prev",
    "user_item_stats:user_item_event_count_prev",
    "user_stats:user_last_event_ts",
    "item_stats:item_last_event_ts",
    "user_item_stats:user_item_last_event_ts",
]

app = FastAPI(title="RetailRocket Inference API", version="0.1.0")
_model = None
_loaded_model_uri: Optional[str] = None
_model_load_error: Optional[str] = None
_feature_store = None
_feature_store_error: Optional[str] = None
_realtime_feature_reader: Optional[RealtimeFeatureReader] = None
_realtime_feature_error: Optional[str] = None
_candidate_generator = None
_candidate_generator_error: Optional[str] = None

REQ_COUNTER = Counter("inference_requests_total", "Total inference requests")
ERROR_COUNTER = Counter("inference_errors_total", "Total inference errors")
LATENCY = Histogram("inference_latency_seconds", "Inference latency in seconds")
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=("path", "method", "status"),
)
HTTP_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    labelnames=("path", "method"),
)
PROBA_HIST = Histogram(
    "model_output_proba",
    "Distribution of model output probabilities",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
LAST_PROBA = Gauge("model_output_proba_last", "Last predicted probability")
POSITIVE_COUNTER = Counter(
    "model_output_positive_total",
    "Number of predictions above threshold",
)


class PredictRequest(BaseModel):
    user_id: Optional[int] = Field(default=None)
    item_id: Optional[int] = Field(default=None)
    is_addtocart: int = Field(default=0, ge=0, le=1)
    user_event_count_prev: Optional[float] = None
    item_event_count_prev: Optional[float] = None
    user_item_event_count_prev: Optional[float] = None
    user_time_since_prev_event_sec: Optional[float] = None
    item_time_since_prev_event_sec: Optional[float] = None
    user_item_time_since_prev_event_sec: Optional[float] = None


class CandidateRequest(BaseModel):
    item_id: int
    is_addtocart: int = Field(default=0, ge=0, le=1)
    user_event_count_prev: Optional[float] = None
    item_event_count_prev: Optional[float] = None
    user_item_event_count_prev: Optional[float] = None
    user_time_since_prev_event_sec: Optional[float] = None
    item_time_since_prev_event_sec: Optional[float] = None
    user_item_time_since_prev_event_sec: Optional[float] = None


class RecommendRequest(BaseModel):
    user_id: Optional[int] = Field(default=None)
    top_k: int = Field(default=10, ge=1, le=100)
    is_addtocart: int = Field(default=0, ge=0, le=1)
    candidate_item_ids: Optional[List[int]] = Field(default=None, min_length=1, max_length=1000)
    candidates: Optional[List[CandidateRequest]] = Field(default=None, min_length=1, max_length=1000)


def _latest_run_model_uri() -> Optional[str]:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if exp is None:
        return None
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    run_id = runs[0].info.run_id
    return f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"


def _resolve_downloaded_model_dir(downloaded_path: str) -> str:
    path = Path(downloaded_path)
    if path.is_file():
        path = path.parent

    if (path / "MLmodel").exists():
        return str(path)

    matches = sorted({candidate.parent for candidate in path.rglob("MLmodel")})
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        raise RuntimeError(
            f"Downloaded artifacts under {path} contained multiple MLmodel files: "
            f"{[str(match) for match in matches]}"
        )
    raise RuntimeError(f"Downloaded artifacts under {path} do not contain an MLmodel file")


def _load_model_from_uri(model_uri: str):
    print(f"[model-load] downloading artifacts for: {model_uri}")
    downloaded_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    resolved_model_dir = _resolve_downloaded_model_dir(downloaded_path)
    print(
        f"[model-load] downloaded_path={downloaded_path}; resolved_model_dir={resolved_model_dir}"
    )
    model = mlflow.xgboost.load_model(resolved_model_dir)
    return model, resolved_model_dir


def _load_model() -> None:
    global _model, _loaded_model_uri, _model_load_error
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    primary_error: Optional[str] = None

    try:
        _model, resolved_model_dir = _load_model_from_uri(MODEL_URI)
        _loaded_model_uri = MODEL_URI
        _model_load_error = None
        print(f"[model-load] loaded model_uri={MODEL_URI} from local_dir={resolved_model_dir}")
        return
    except Exception as exc:
        primary_error = f"primary({MODEL_URI}) failed: {exc}"

    if FALLBACK_TO_LATEST_RUN:
        try:
            fallback_uri = _latest_run_model_uri()
        except Exception as exc:
            fallback_uri = None
            primary_error = f"{primary_error}; latest-run lookup failed: {exc}"

        if fallback_uri:
            try:
                _model, resolved_model_dir = _load_model_from_uri(fallback_uri)
                _loaded_model_uri = fallback_uri
                _model_load_error = None
                print(
                    f"[model-load] loaded fallback_uri={fallback_uri} from local_dir={resolved_model_dir}"
                )
                return
            except Exception as exc:
                primary_error = f"{primary_error}; fallback({fallback_uri}) failed: {exc}"

    _model = None
    _loaded_model_uri = None
    _model_load_error = primary_error


def _load_feature_store() -> None:
    global _feature_store, _feature_store_error
    if FeatureStore is None:
        _feature_store = None
        _feature_store_error = "Feast is not installed in the current runtime"
        return

    try:
        _feature_store = FeatureStore(repo_path=FEAST_REPO_PATH)
        _feature_store_error = None
    except Exception as exc:
        _feature_store = None
        _feature_store_error = str(exc)


def _load_realtime_feature_reader() -> None:
    global _realtime_feature_reader, _realtime_feature_error
    try:
        reader = RealtimeFeatureReader.from_env()
        reader.ping()
        _realtime_feature_reader = reader
        _realtime_feature_error = None
    except Exception as exc:
        _realtime_feature_reader = None
        _realtime_feature_error = str(exc)


def _load_candidate_generator() -> None:
    global _candidate_generator, _candidate_generator_error
    try:
        artifact_path = resolve_recommendation_artifact_path(RECOMMENDATION_CANDIDATE_ARTIFACT_PATH)
        _candidate_generator = CandidateGenerator.from_artifact(artifact_path)
        _candidate_generator_error = None
        return
    except Exception as exc:
        artifact_error = exc

    if RECOMMENDATION_CANDIDATE_ALLOW_PARQUET_FALLBACK:
        try:
            events_path = resolve_recommendation_events_path(RECOMMENDATION_EVENTS_PATH)
            _candidate_generator = CandidateGenerator.from_parquet(events_path)
            _candidate_generator_error = None
            return
        except Exception as exc:
            _candidate_generator = None
            _candidate_generator_error = (
                f"Recommendation candidate artifact load failed: {artifact_error}; "
                f"parquet fallback also failed: {exc}"
            )
            return

    _candidate_generator = None
    _candidate_generator_error = (
        f"Recommendation candidate artifact load failed: {artifact_error}. "
        "Build it with `python data/scripts/build_recommendation_candidates.py`, "
        "or set RECOMMENDATION_CANDIDATE_ALLOW_PARQUET_FALLBACK=true to temporarily rebuild from the "
        "processed events parquet at startup."
    )


def _user_item_key(user_id: int, item_id: int) -> str:
    return f"{int(user_id)}:{int(item_id)}"


def _payload_has_manual_features(payload: PredictRequest) -> bool:
    return all(getattr(payload, name) is not None for name in FEATURES)


def _feature_dict_value(values: Dict[str, List], feature_ref: str):
    candidates = [feature_ref, feature_ref.split(":")[-1]]
    for key in candidates:
        if key in values and values[key]:
            return values[key][0]
    return None


def _coerce_online_timestamp(value) -> Optional[pd.Timestamp]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        abs_value = abs(float(value))
        if abs_value < 1e11:
            return pd.to_datetime(value, unit="s", utc=True, errors="coerce")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _seconds_since(ts: Optional[pd.Timestamp]) -> float:
    if ts is None:
        return -1.0
    delta = (pd.Timestamp.now(tz="UTC") - ts).total_seconds()
    return float(max(delta, 0.0))


def _build_manual_feature_row(payload: PredictRequest) -> Dict[str, float]:
    return {name: float(getattr(payload, name)) for name in FEATURES}


def _build_realtime_feature_row(payload: PredictRequest) -> Dict[str, float]:
    if payload.user_id is None or payload.item_id is None:
        raise ValueError("user_id and item_id are required for realtime Redis retrieval")
    if _realtime_feature_reader is None:
        raise RuntimeError(_realtime_feature_error or "Realtime Redis feature reader is not available")
    return _realtime_feature_reader.build_feature_row(
        user_id=int(payload.user_id),
        item_id=int(payload.item_id),
        is_addtocart=int(payload.is_addtocart),
    )


def _build_online_feature_row(payload: PredictRequest) -> Dict[str, float]:
    if payload.user_id is None or payload.item_id is None:
        raise ValueError("user_id and item_id are required for Feast online retrieval")
    if _feature_store is None:
        raise RuntimeError(_feature_store_error or "Feast feature store is not available")

    response = _feature_store.get_online_features(
        features=ONLINE_FEATURE_REFS,
        entity_rows=[
            {
                "user_id": int(payload.user_id),
                "item_id": int(payload.item_id),
                "user_item_key": _user_item_key(payload.user_id, payload.item_id),
            }
        ],
    )
    values = response.to_dict()

    user_last_event_ts = _coerce_online_timestamp(
        _feature_dict_value(values, "user_stats:user_last_event_ts")
    )
    item_last_event_ts = _coerce_online_timestamp(
        _feature_dict_value(values, "item_stats:item_last_event_ts")
    )
    user_item_last_event_ts = _coerce_online_timestamp(
        _feature_dict_value(values, "user_item_stats:user_item_last_event_ts")
    )

    return {
        "is_addtocart": float(payload.is_addtocart),
        "user_event_count_prev": float(
            _feature_dict_value(values, "user_stats:user_event_count_prev") or 0.0
        ),
        "item_event_count_prev": float(
            _feature_dict_value(values, "item_stats:item_event_count_prev") or 0.0
        ),
        "user_item_event_count_prev": float(
            _feature_dict_value(values, "user_item_stats:user_item_event_count_prev") or 0.0
        ),
        "user_time_since_prev_event_sec": _seconds_since(user_last_event_ts),
        "item_time_since_prev_event_sec": _seconds_since(item_last_event_ts),
        "user_item_time_since_prev_event_sec": _seconds_since(user_item_last_event_ts),
    }


def _resolve_predict_features(payload: PredictRequest) -> tuple[Dict[str, float], str]:
    if payload.user_id is not None and payload.item_id is not None:
        try:
            return _build_realtime_feature_row(payload), "redis_realtime"
        except Exception as exc:
            LOGGER.info(
                "[features] realtime Redis lookup unavailable for user_id=%s item_id=%s: %s",
                payload.user_id,
                payload.item_id,
                exc,
            )
        try:
            return _build_online_feature_row(payload), "feast_online"
        except Exception as exc:
            LOGGER.info(
                "[features] Feast online lookup unavailable for user_id=%s item_id=%s: %s",
                payload.user_id,
                payload.item_id,
                exc,
            )
            if _payload_has_manual_features(payload):
                return _build_manual_feature_row(payload), "manual_fallback"
            raise

    if _payload_has_manual_features(payload):
        return _build_manual_feature_row(payload), "manual_fallback"

    raise ValueError(
        "Provide either user_id + item_id for realtime Redis / Feast retrieval, or all model features in the payload"
    )


def _candidate_to_predict_request(
    user_id: Optional[int], candidate: RecommendationCandidate
) -> PredictRequest:
    return PredictRequest(
        user_id=user_id,
        item_id=candidate.item_id,
        is_addtocart=candidate.is_addtocart,
        user_event_count_prev=candidate.user_event_count_prev,
        item_event_count_prev=candidate.item_event_count_prev,
        user_item_event_count_prev=candidate.user_item_event_count_prev,
        user_time_since_prev_event_sec=candidate.user_time_since_prev_event_sec,
        item_time_since_prev_event_sec=candidate.item_time_since_prev_event_sec,
        user_item_time_since_prev_event_sec=candidate.user_item_time_since_prev_event_sec,
    )


def _predict_proba_values(X: pd.DataFrame) -> np.ndarray:
    if hasattr(_model, "predict_proba"):
        values = _model.predict_proba(X)[:, 1]
    else:
        values = np.asarray(_model.predict(X)).reshape(-1)
    return np.clip(values.astype(float), 0.0, 1.0)


def _observe_output_metrics(values: np.ndarray) -> None:
    if values.size == 0:
        return
    for value in values:
        PROBA_HIST.observe(float(value))
    LAST_PROBA.set(float(values[-1]))
    positive_count = int((values >= PREDICTION_THRESHOLD).sum())
    if positive_count > 0:
        POSITIVE_COUNTER.inc(positive_count)


@app.on_event("startup")
def startup_event() -> None:
    _load_model()
    _load_feature_store()
    _load_realtime_feature_reader()
    _load_candidate_generator()
    if _model is None:
        print(f"[startup] model not loaded: {_model_load_error}")
    else:
        print(f"[startup] model loaded from: {_loaded_model_uri}")
    if _feature_store is None:
        print(f"[startup] feast feature store not loaded: {_feature_store_error}")
    else:
        print(f"[startup] feast feature store loaded from: {FEAST_REPO_PATH}")
    if _realtime_feature_reader is None:
        print(f"[startup] realtime feature reader not loaded: {_realtime_feature_error}")
    else:
        print(
            "[startup] realtime feature reader loaded from: "
            f"{_realtime_feature_reader.host}:{_realtime_feature_reader.port} db={_realtime_feature_reader.db}"
        )
    if _candidate_generator is None:
        print(f"[startup] candidate generator not loaded: {_candidate_generator_error}")
    else:
        print(
            f"[startup] candidate generator loaded from: {_candidate_generator.events_path}"
        )


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = perf_counter() - start
        path = request.url.path
        method = request.method
        HTTP_REQUESTS.labels(path=path, method=method, status=str(status_code)).inc()
        HTTP_LATENCY.labels(path=path, method=method).observe(elapsed)


@app.get("/")
def root() -> dict:
    return {
        "status": "ok",
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_uri_config": MODEL_URI,
        "model_uri_loaded": _loaded_model_uri,
        "model_loaded": _model is not None,
        "feast_repo_path": FEAST_REPO_PATH,
        "feast_loaded": _feature_store is not None,
        "realtime_feature_reader_loaded": _realtime_feature_reader is not None,
        "candidate_generation_loaded": _candidate_generator is not None,
        "candidate_generation_source": (
            str(_candidate_generator.source_path) if _candidate_generator is not None else None
        ),
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"alive": True}


@app.get("/readyz")
def readyz() -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail={"ready": False, "error": _model_load_error})
    return {"ready": True, "model_uri_loaded": _loaded_model_uri}


@app.post("/predict_proba")
def predict_proba(payload: PredictRequest) -> dict:
    REQ_COUNTER.inc()
    start = perf_counter()
    if _model is None:
        ERROR_COUNTER.inc()
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model is not loaded",
                "model_uri_config": MODEL_URI,
                "model_uri_loaded": _loaded_model_uri,
                "tracking_uri": MLFLOW_TRACKING_URI,
                "load_error": _model_load_error,
            },
        )

    try:
        row, feature_source = _resolve_predict_features(payload)
        X = pd.DataFrame([row], columns=FEATURES)
        values = _predict_proba_values(X)
        _observe_output_metrics(values)
        value = float(values[0])

        return {
            "user_id": payload.user_id,
            "item_id": payload.item_id,
            "model_uri": _loaded_model_uri,
            "feature_source": feature_source,
            "proba": value,
            "threshold": PREDICTION_THRESHOLD,
        }
    except Exception as exc:
        ERROR_COUNTER.inc()
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    finally:
        LATENCY.observe(perf_counter() - start)


@app.post("/recommend")
def recommend(payload: RecommendRequest) -> dict:
    REQ_COUNTER.inc()
    start = perf_counter()
    if _model is None:
        ERROR_COUNTER.inc()
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model is not loaded",
                "model_uri_config": MODEL_URI,
                "model_uri_loaded": _loaded_model_uri,
                "tracking_uri": MLFLOW_TRACKING_URI,
                "load_error": _model_load_error,
            },
        )

    try:
        if (
            payload.candidate_item_ids is None
            and payload.candidates is None
            and _candidate_generator is None
        ):
            _load_candidate_generator()
        manual_candidates = (
            [candidate.model_dump() for candidate in payload.candidates]
            if payload.candidates is not None
            else None
        )
        candidates, request_mode = resolve_recommendation_candidates(
            user_id=payload.user_id,
            top_k=payload.top_k,
            is_addtocart=payload.is_addtocart,
            candidate_item_ids=payload.candidate_item_ids,
            manual_candidates=manual_candidates,
            candidate_generator=_candidate_generator,
            candidate_generator_error=_candidate_generator_error,
        )

        rows = []
        item_ids = []
        feature_sources = []
        for cand in candidates:
            predict_payload = _candidate_to_predict_request(payload.user_id, cand)
            row, feature_source = _resolve_predict_features(predict_payload)
            item_ids.append(cand.item_id)
            feature_sources.append(feature_source)
            rows.append(row)

        X = pd.DataFrame(rows, columns=FEATURES)
        values = _predict_proba_values(X)
        _observe_output_metrics(values)

        scored = [
            {
                "item_id": item_id,
                "proba": float(proba),
                "feature_source": feature_source,
            }
            for item_id, proba, feature_source in zip(item_ids, values, feature_sources)
        ]
        scored.sort(key=lambda x: x["proba"], reverse=True)
        top_items = scored[: payload.top_k]

        return {
            "user_id": payload.user_id,
            "top_k": payload.top_k,
            "num_candidates": len(scored),
            "request_mode": request_mode,
            "model_uri": _loaded_model_uri,
            "threshold": PREDICTION_THRESHOLD,
            "items": top_items,
        }
    except Exception as exc:
        ERROR_COUNTER.inc()
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc
    finally:
        LATENCY.observe(perf_counter() - start)


@app.post("/reload_model")
def reload_model() -> dict:
    _load_model()
    _load_feature_store()
    _load_realtime_feature_reader()
    _load_candidate_generator()
    if _model is None:
        raise HTTPException(status_code=500, detail={"reloaded": False, "error": _model_load_error})
    return {"reloaded": True, "model_uri_loaded": _loaded_model_uri}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type="text/plain; version=0.0.4; charset=utf-8")



