#!/usr/bin/env bash
# FinGuard RAG: post-interview cleanup.
#
# Deletes the Cloud Run service, the images and the Artifact Registry repository
# created by deploy/deploy.sh. Run manually:
#
#   PROJECT_ID=my-project REGION=us-central1 bash deploy/teardown.sh
#
# Deletion is permanent. The script asks for confirmation first.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-finguard-rag}"
REPO="${REPO:-finguard}"

if [ "$PROJECT_ID" = "PROJECT_ID" ]; then
  echo "Set PROJECT_ID first, e.g. PROJECT_ID=my-project bash deploy/teardown.sh" >&2
  exit 1
fi

REPO_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"

echo "Images currently stored in ${REPO_PATH}:"
gcloud artifacts docker images list "$REPO_PATH" --include-tags || true

read -r -p "Delete service '${SERVICE}' and repository '${REPO}' in ${PROJECT_ID}/${REGION}? [y/N] " answer
[ "$answer" = "y" ] || [ "$answer" = "Y" ] || { echo "Aborted."; exit 1; }

# 1. Cloud Run service.
gcloud run services delete "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --quiet

# 2. Repository (deletes every image version inside it).
gcloud artifacts repositories delete "$REPO" \
  --project "$PROJECT_ID" \
  --location "$REGION" \
  --quiet

echo "Done. Remaining Cloud Run services in ${REGION}:"
gcloud run services list --project "$PROJECT_ID" --region "$REGION"

# To keep the repository but delete only old image versions instead, run:
#
#   gcloud artifacts docker images list "$REPO_PATH/$SERVICE" --include-tags
#   gcloud artifacts docker images delete "$REPO_PATH/$SERVICE@sha256:<DIGEST>" --delete-tags --quiet
