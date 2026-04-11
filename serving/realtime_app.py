import logging
from typing import Dict, Optional

import serving.app as base
from fastapi import HTTPException

from serving.realtime_features import RealtimeFeatureReader

LOGGER = logging.getLogger("serving.realtime_app")
app = base.app
_realtime_feature_reader: Optional[RealtimeFeatureReader] = None
_realtime_feature_error: Optional[str] = None


def _load_realtime_feature_reader() -> None:
    global _realtime_feature_reader, _realtime_feature_error
    try:
        reader = RealtimeFeatureReader.from_env()
        reader.ping()
        _realtime_feature_reader = reader
        _realtime_feature_error = None
        LOGGER.info(
            "[startup] realtime Redis feature reader loaded from %s:%s db=%s",
            reader.host,
            reader.port,
            reader.db,
        )
    except Exception as exc:
        _realtime_feature_reader = None
        _realtime_feature_error = str(exc)
        LOGGER.info("[startup] realtime Redis feature reader not loaded: %s", exc)


def _build_realtime_feature_row(payload: base.PredictRequest) -> Dict[str, float]:
    if payload.user_id is None or payload.item_id is None:
        raise ValueError("user_id and item_id are required for realtime Redis retrieval")
    if _realtime_feature_reader is None:
        raise RuntimeError(_realtime_feature_error or "Realtime Redis feature reader is not available")
    return _realtime_feature_reader.build_feature_row(
        user_id=int(payload.user_id),
        item_id=int(payload.item_id),
        is_addtocart=int(payload.is_addtocart),
    )


def _resolve_predict_features(payload: base.PredictRequest) -> tuple[Dict[str, float], str]:
    if payload.user_id is not None and payload.item_id is not None:
        try:
            row = _build_realtime_feature_row(payload)
            LOGGER.info(
                "[features] using redis_realtime for user_id=%s item_id=%s",
                payload.user_id,
                payload.item_id,
            )
            return row, "redis_realtime"
        except Exception as exc:
            LOGGER.info(
                "[features] realtime Redis lookup unavailable for user_id=%s item_id=%s: %s",
                payload.user_id,
                payload.item_id,
                exc,
            )
        try:
            row = base._build_online_feature_row(payload)
            LOGGER.info(
                "[features] using feast_online for user_id=%s item_id=%s",
                payload.user_id,
                payload.item_id,
            )
            return row, "feast_online"
        except Exception as exc:
            LOGGER.info(
                "[features] Feast online lookup unavailable for user_id=%s item_id=%s: %s",
                payload.user_id,
                payload.item_id,
                exc,
            )
            if base._payload_has_manual_features(payload):
                LOGGER.info(
                    "[features] using manual_fallback for user_id=%s item_id=%s",
                    payload.user_id,
                    payload.item_id,
                )
                return base._build_manual_feature_row(payload), "manual_fallback"
            raise

    if base._payload_has_manual_features(payload):
        LOGGER.info("[features] using manual_payload")
        return base._build_manual_feature_row(payload), "manual_payload"

    raise ValueError(
        "Provide either user_id + item_id for realtime Redis / Feast retrieval, or all model features in the payload"
    )


def _reload_model() -> dict:
    base._load_model()
    base._load_feature_store()
    if hasattr(base, "_load_candidate_generator"):
        base._load_candidate_generator()
    _load_realtime_feature_reader()
    if base._model is None:
        raise HTTPException(status_code=500, detail={"reloaded": False, "error": base._model_load_error})
    return {"reloaded": True, "model_uri_loaded": base._loaded_model_uri}


@app.on_event("startup")
def startup_realtime_features() -> None:
    _load_realtime_feature_reader()


for route in app.router.routes:
    if getattr(route, "path", None) == "/reload_model":
        route.endpoint = _reload_model
        if hasattr(route, "dependant"):
            route.dependant.call = _reload_model

base._resolve_predict_features = _resolve_predict_features
