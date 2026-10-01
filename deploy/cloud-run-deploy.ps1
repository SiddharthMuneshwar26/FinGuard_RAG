# Cloud Run deployment template.
# DO NOT run automatically. Fill in the placeholders and execute manually.

$PROJECT_ID = "<YOUR_GCP_PROJECT_ID>"
$REGION = "<YOUR_GCP_REGION>"
$IMAGE = "<YOUR_ARTIFACT_REGISTRY_IMAGE>"

gcloud run deploy fingguard-rag-api `
  --project $PROJECT_ID `
  --region $REGION `
  --image $IMAGE `
  --platform managed `
  --no-allow-unauthenticated `
  --min 0 `
  --max 1 `
  --concurrency 4 `
  --cpu 2 `
  --memory 4Gi `
  --timeout 300 `
  --port 8080
