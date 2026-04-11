# Feast Feature Repo

This is the canonical Feast repository for the project.

It preserves training/serving feature parity for the current thesis-stage model:
- `is_addtocart` remains request-time context supplied by the API caller
- historical aggregate features are fetched online from Feast + Redis

Current online feature set:
- `user_stats:user_event_count_prev`
- `item_stats:item_event_count_prev`
- `user_item_stats:user_item_event_count_prev`
- `user_stats:user_last_event_ts`
- `item_stats:item_last_event_ts`
- `user_item_stats:user_item_last_event_ts`

Offline and online now share one canonical feature module:
- `feature_repo/feature_definitions.py` defines `MODEL_FEATURES` and `LABEL_COLUMN`
- `feature_repo/feature_definitions.py` also contains the leakage-safe offline
  feature engineering used by `data/scripts/build_trainset_retailrocket.py`
- `feature_repo/build_online_features.py` builds the corresponding online
  aggregate sources for Feast materialization to Redis

Parity contract for the current model:
- Offline training features are built from processed events through
  `feature_repo.feature_definitions.build_labeled_candidate_frame(...)`
- Online serving features use the same feature names, with `is_addtocart` from
  the request, counts from Redis-backed sources, and recency reconstructed from
  the stored last-event timestamps

This keeps the canonical feature definitions, offline dataset construction, and
online Redis-backed serving story in one place while keeping the local stack
minimal and thesis-friendly.
