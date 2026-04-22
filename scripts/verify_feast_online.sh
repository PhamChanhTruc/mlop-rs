#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFRA_DIR="${REPO_ROOT}/infra"
API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
FEATURE_REPO_DIR="${REPO_ROOT}/feature_repo"
PROCESSED_EVENTS_PATH="${REPO_ROOT}/data/processed/events_retailrocket.parquet"
ONLINE_FEATURE_REFS="user_stats:user_event_count_prev,item_stats:item_event_count_prev,user_item_stats:user_item_event_count_prev,user_stats:user_last_event_ts,item_stats:item_last_event_ts,user_item_stats:user_item_last_event_ts"

if [[ ! -f "${PROCESSED_EVENTS_PATH}" ]]; then
  echo "ERROR: Missing processed events parquet: ${PROCESSED_EVENTS_PATH}" >&2
  echo "Run data/scripts/preprocess_retailrocket.py first." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not available in PATH." >&2
  exit 1
fi

echo "[1/5] Verifying required Compose services are running..."
(
  cd "${INFRA_DIR}"
  docker compose ps --status running api redis mlflow >/dev/null
)

echo "[2/5] Building Feast online feature source files inside the Compose network..."
(
  cd "${INFRA_DIR}"
  docker compose exec -T api bash -lc "cd /app && python -m feature_repo.build_online_features"
)

for required_path in \
  "${FEATURE_REPO_DIR}/data/user_stats.parquet" \
  "${FEATURE_REPO_DIR}/data/item_stats.parquet" \
  "${FEATURE_REPO_DIR}/data/user_item_stats.parquet"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "ERROR: Expected generated Feast source file is missing: ${required_path}" >&2
    exit 1
  fi
done
echo "Generated Feast source parquet files are present under feature_repo/data/."

echo "[3/5] Applying Feast registry definitions inside the Compose network..."
(
  cd "${INFRA_DIR}"
  docker compose exec -T api bash -lc "cd /app/feature_repo && feast apply"
)

if [[ ! -f "${FEATURE_REPO_DIR}/data/registry.db" ]]; then
  echo "ERROR: Feast registry file was not created at ${FEATURE_REPO_DIR}/data/registry.db" >&2
  exit 1
fi
echo "Feast registry file is present at feature_repo/data/registry.db."

echo "[4/5] Materializing Feast online features to Redis..."
END_TS="$(date -u +%Y-%m-%dT%H:%M:%S)"
(
  cd "${INFRA_DIR}"
  docker compose exec -T api bash -lc "cd /app/feature_repo && feast materialize-incremental ${END_TS}"
)

echo "Selecting a processed user/item pair without realtime Redis hashes..."
SELECTED_PAIR="$(
  cd "${REPO_ROOT}" && python3 - <<'PY'
from pathlib import Path

import pandas as pd

events = pd.read_parquet(
    Path("data/processed/events_retailrocket.parquet"),
    columns=["user_id", "item_id"],
)
events = events.dropna(subset=["user_id", "item_id"]).drop_duplicates().head(500)
for row in events.itertuples(index=False):
    print(f"{int(row.user_id)} {int(row.item_id)}")
PY
)"

SELECTED_USER_ID=""
SELECTED_ITEM_ID=""
while IFS=' ' read -r user_id item_id; do
  [[ -z "${user_id}" ]] && continue
  exists_count="$(
    cd "${INFRA_DIR}" &&
      docker compose exec -T redis redis-cli EXISTS \
        "user:${user_id}" \
        "item:${item_id}" \
        "user_item:${user_id}:${item_id}"
  )"
  if [[ "${exists_count}" == "0" ]]; then
    SELECTED_USER_ID="${user_id}"
    SELECTED_ITEM_ID="${item_id}"
    break
  fi
done <<< "${SELECTED_PAIR}"

if [[ -z "${SELECTED_USER_ID}" || -z "${SELECTED_ITEM_ID}" ]]; then
  echo "ERROR: Could not find a processed user/item pair without realtime Redis hashes." >&2
  echo "Stop the realtime processor or clear those demo hashes, then rerun this check." >&2
  exit 1
fi

echo "Using user_id=${SELECTED_USER_ID} item_id=${SELECTED_ITEM_ID} for Feast verification."

echo "[5/5] Verifying Feast online lookup directly inside the API container..."
(
  cd "${INFRA_DIR}"
  docker compose exec -T api bash -lc "
    cd /app && python - <<'PY'
import json
from feast import FeatureStore

user_id = int(${SELECTED_USER_ID})
item_id = int(${SELECTED_ITEM_ID})
store = FeatureStore(repo_path='/app/feature_repo')
response = store.get_online_features(
    features='${ONLINE_FEATURE_REFS}'.split(','),
    entity_rows=[{
        'user_id': user_id,
        'item_id': item_id,
        'user_item_key': f'{user_id}:{item_id}',
    }],
).to_dict()

required_refs = [
    'user_stats:user_event_count_prev',
    'item_stats:item_event_count_prev',
    'user_item_stats:user_item_event_count_prev',
]
missing = [feature_ref for feature_ref in required_refs if not response.get(feature_ref) or response[feature_ref][0] is None]
if missing:
    raise SystemExit(f'ERROR: Feast online lookup returned missing values for {missing}: {json.dumps(response, default=str)}')

print(json.dumps(response, indent=2, default=str))
PY"
)

echo "Direct Feast online lookup passed."

echo "Checking /predict_proba for feature_source=feast_online when the API can score requests..."
HTTP_BODY="$(mktemp)"
HTTP_CODE="$(
  curl -sS -o "${HTTP_BODY}" -w "%{http_code}" \
    -X POST "${API_BASE_URL}/predict_proba" \
    -H "Content-Type: application/json" \
    -d "{\"user_id\": ${SELECTED_USER_ID}, \"item_id\": ${SELECTED_ITEM_ID}, \"is_addtocart\": 0}"
)"

if [[ "${HTTP_CODE}" == "200" ]]; then
  FEATURE_SOURCE="$(python3 - "${HTTP_BODY}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(payload.get("feature_source", ""))
PY
)"
  cat "${HTTP_BODY}"
  echo
  if [[ "${FEATURE_SOURCE}" != "feast_online" ]]; then
    echo "ERROR: /predict_proba did not use Feast online features. feature_source=${FEATURE_SOURCE}" >&2
    rm -f "${HTTP_BODY}"
    exit 1
  fi
  echo "PASS: /predict_proba returned feature_source=feast_online."
elif [[ "${HTTP_CODE}" == "503" ]]; then
  cat "${HTTP_BODY}"
  echo
  echo "NOTE: Feast online lookup is ready, but the API did not have a loaded model, so /predict_proba could not score yet." >&2
  echo "Promote a model and call POST /reload_model, then rerun this helper to assert feature_source=feast_online at the API layer." >&2
else
  cat "${HTTP_BODY}"
  echo
  echo "ERROR: /predict_proba returned unexpected HTTP ${HTTP_CODE}." >&2
  rm -f "${HTTP_BODY}"
  exit 1
fi

rm -f "${HTTP_BODY}"

echo
echo "Feast online state is explicitly verified."
echo "Pass signals:"
echo "  - feature_repo/data/*.parquet exists"
echo "  - feature_repo/data/registry.db exists"
echo "  - direct Feast online lookup returns non-null online feature values"
echo "  - /predict_proba returns feature_source=feast_online when the API model is loaded"
