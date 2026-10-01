#!/usr/bin/env bash
# FinGuard RAG: private, scale-to-zero, retrieval-only Cloud Run deployment.
#
# Run manually, from the repository root, AFTER completing the one-time
# prerequisites in deploy/DEPLOY.md (project, billing, budget, APIs, login).
#
#   PROJECT_ID=my-project REGION=us-central1 bash deploy/deploy.sh
#
# This script creates billable resources (an Artifact Registry repository and
# a Cloud Run service). It asks for confirmation before pushing and deploying.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-finguard-rag}"
REPO="${REPO:-finguard}"

if [ "$PROJECT_ID" = "PROJECT_ID" ]; then
  echo "Set PROJECT_ID first, e.g. PROJECT_ID=my-project bash deploy/deploy.sh" >&2
  exit 1
fi

IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:${IMAGE_TAG}"

echo "Project : ${PROJECT_ID}"
echo "Region  : ${REGION}"
echo "Service : ${SERVICE}"
echo "Image   : ${IMAGE}"
read -r -p "Create/push/deploy these resources? [y/N] " answer
[ "$answer" = "y" ] || [ "$answer" = "Y" ] || { echo "Aborted."; exit 1; }

# 1. Artifact Registry Docker repository (created only if missing).
if ! gcloud artifacts repositories describe "$REPO" \
      --project "$PROJECT_ID" --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPO" \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --repository-format docker \
    --description "FinGuard RAG images"
fi

# 2. Let Docker push to Artifact Registry with your gcloud credentials.
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# 3. Build for Cloud Run's architecture, then push.
docker buildx build --platform linux/amd64 --load -t "$IMAGE" .
docker push "$IMAGE"

# 4. Deploy: private, scale-to-zero, retrieval-only (no Ollama/LLM variables).
#    HF_HUB_OFFLINE=1: models are baked into the image, so skip Hugging Face
#    network checks at startup.
gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --image "$IMAGE" \
  --no-allow-unauthenticated \
  --min-instances 0 \
  --max-instances 1 \
  --concurrency 4 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --port 8080 \
  --set-env-vars HF_HUB_OFFLINE=1

gcloud run services describe "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format "value(status.url)"
