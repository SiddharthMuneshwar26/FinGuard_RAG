# FinGuard RAG - Cloud Run

## Intended deployment

The FastAPI container is designed for a private Cloud Run service using:

- Min instances: 0
- Max instances: 1
- Concurrency: 4
- CPU: 2
- Memory: 4 GiB
- Request timeout: 300 seconds
- Container port: 8080
- Authentication: required

## Generation behavior

Cloud Run does not run Ollama. Production Cloud Run requests should use generate=false.

Local development can continue using Ollama and generate=true.

## Deployment

Use deploy/cloud-run-deploy.ps1 as the deployment template.

Before deploying, replace:

- <YOUR_GCP_PROJECT_ID>
- <YOUR_GCP_REGION>
- <YOUR_ARTIFACT_REGISTRY_IMAGE>

The deployment template uses --no-allow-unauthenticated.

No GCP authentication, Artifact Registry creation, image push, or Cloud Run deployment is performed automatically.

## Post-deployment verification

After a manual deployment, obtain the service URL:

gcloud run services describe fingguard-rag-api `
  --project <YOUR_GCP_PROJECT_ID> `
  --region <YOUR_GCP_REGION> `
  --format="value(status.url)"

Verify the private health endpoint:

$URL = "<CLOUD_RUN_SERVICE_URL>"
$TOKEN = gcloud auth print-identity-token

curl.exe `
  -H "Authorization: Bearer $TOKEN" `
  "$URL/health"

Expected:

{"status":"ok"}

## Retrieval-only query

Use generate=false because Ollama is not inside the Cloud Run container.

Example request:

{
  "question": "What is SHAP and how does it explain model predictions?",
  "top_k": 4,
  "generate": false
}

Expected:

- HTTP 200
- answer is null
- citations contains retrieved sources
- chunks contains retrieved/reranked evidence
- latency_ms contains timing information

## Teardown

When the service is no longer required:

gcloud run services delete fingguard-rag-api `
  --project <YOUR_GCP_PROJECT_ID> `
  --region <YOUR_GCP_REGION>

Do not execute deployment or teardown commands unless explicitly intended.