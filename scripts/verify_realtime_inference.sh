#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
KAFKA_BOOTSTRAP_SERVERS="${KAFKA_BOOTSTRAP_SERVERS:-localhost:29092}"
TOPIC="${TOPIC:-user_events}"
USER_ID="${USER_ID:-1}"
ITEM_ID="${ITEM_ID:-101}"
TOP_K="${TOP_K:-3}"
NUM_EVENTS="${NUM_EVENTS:-5}"
SLEEP_SEC="${SLEEP_SEC:-0.2}"

echo "Publishing ${NUM_EVENTS} demo events for user ${USER_ID}, item ${ITEM_ID}..."
python3 streaming/producer/simulate_events.py \
  --bootstrap-servers "${KAFKA_BOOTSTRAP_SERVERS}" \
  --topic "${TOPIC}" \
  --num-events "${NUM_EVENTS}" \
  --sleep-sec "${SLEEP_SEC}" \
  --user-id-start "${USER_ID}" \
  --user-id-end "${USER_ID}" \
  --item-id-start "${ITEM_ID}" \
  --item-id-end "${ITEM_ID}"

echo
echo "Waiting briefly for Kafka -> processor -> Redis..."
sleep 2

echo
echo "Inspecting Redis hashes..."
python3 scripts/verify_realtime_redis.py --user-id "${USER_ID}" --item-id "${ITEM_ID}"

echo
echo "Calling /predict_proba..."
curl -s -X POST "${API_BASE_URL}/predict_proba" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": ${USER_ID}, \"item_id\": ${ITEM_ID}, \"is_addtocart\": 1}"
echo

echo
echo "Calling /recommend..."
curl -s -X POST "${API_BASE_URL}/recommend" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": ${USER_ID}, \"top_k\": ${TOP_K}, \"candidate_item_ids\": [${ITEM_ID}]}"
echo

echo
echo "Expected demo signals:"
echo "- Redis counts for user:${USER_ID}, item:${ITEM_ID}, and user_item:${USER_ID}:${ITEM_ID} should increase"
echo "- /predict_proba should return feature_source=redis_realtime"
echo "- /recommend should include the updated item with feature_source=redis_realtime"
