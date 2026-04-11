#!/usr/bin/env bash
set -euo pipefail

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
: "${GCP_REGION:?Set GCP_REGION}"
: "${ARTIFACT_REGISTRY_REPO:?Set ARTIFACT_REGISTRY_REPO}"
: "${CLOUD_RUN_SERVICE:?Set CLOUD_RUN_SERVICE}"

IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${CLOUD_RUN_SERVICE}:${IMAGE_TAG:-latest}"

printf 'Using image: %s\n' "$IMAGE_URI"

 gcloud config set project "$GCP_PROJECT_ID"
 gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
 gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

 gcloud artifacts repositories describe "$ARTIFACT_REGISTRY_REPO" \
   --location "$GCP_REGION" >/dev/null 2>&1 || \
 gcloud artifacts repositories create "$ARTIFACT_REGISTRY_REPO" \
   --repository-format docker \
   --location "$GCP_REGION" \
   --description "Artifact Registry for thesis inference images"

 gcloud builds submit \
   --tag "$IMAGE_URI" \
   .

ENV_VARS=(
  "MODEL_URI=${MODEL_URI:-models:/xgb-baseline-retailrocket/Production}"
  "MODEL_URI_FALLBACK_TO_LATEST_RUN=false"
  "PREDICTION_THRESHOLD=${PREDICTION_THRESHOLD:-0.5}"
  "FEAST_REPO_PATH=/app/feature_repo"
)

if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
  ENV_VARS+=("MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}")
fi

if [[ -n "${MLFLOW_EXPERIMENT:-}" ]]; then
  ENV_VARS+=("MLFLOW_EXPERIMENT=${MLFLOW_EXPERIMENT}")
fi

 gcloud run deploy "$CLOUD_RUN_SERVICE" \
   --image "$IMAGE_URI" \
   --region "$GCP_REGION" \
   --platform managed \
   --allow-unauthenticated \
   --port 8000 \
   --memory "${CLOUD_RUN_MEMORY:-1Gi}" \
   --cpu "${CLOUD_RUN_CPU:-1}" \
   --min-instances "${CLOUD_RUN_MIN_INSTANCES:-0}" \
   --max-instances "${CLOUD_RUN_MAX_INSTANCES:-2}" \
   --set-env-vars "$(IFS=,; echo "${ENV_VARS[*]}")"

printf '\nCloud Run deployment submitted for %s\n' "$CLOUD_RUN_SERVICE"
printf 'If Feast-backed inference is required in Cloud Run, also provision a reachable Redis-backed Feast online store and update the repo config accordingly.\n'
