# FinGuard RAG: Cloud Run deployment

A **private, scale-to-zero, retrieval-only** deployment of the FinGuard REST API on Google Cloud Run.

- **Private:** every request needs a Google identity token (`--no-allow-unauthenticated`).
- **Scale-to-zero:** `--min-instances 0`, so no instance runs while the service is idle.
- **Retrieval-only:** the service returns reranked evidence and citations with `"generate": false`. Ollama is not part of the image, so `"generate": true` returns HTTP `503`. Local development can still use Ollama with `"generate": true`.

> Status: **not deployed.** Nothing in this repository creates cloud resources automatically. Every command below is run by hand.

Placeholders used throughout:

| Placeholder | Value |
|---|---|
| `PROJECT_ID` | Your GCP project ID |
| `REGION` | A Tier 1 region, for example `us-central1` |
| `SERVICE` | `finguard-rag` |
| `REPO` | `finguard` |

The bash commands work in Linux/macOS shells and in Git Bash on Windows. PowerShell versions are given for the test commands.

---

## 1. One-time prerequisites (do these in order)

1. **Create a GCP project** in the [Cloud Console](https://console.cloud.google.com/projectcreate) and note its project ID.

2. **Attach a billing account** to the project: *Billing → Link a billing account*.

3. **Create a spend-cap budget of about $1 before deploying anything:** *Billing → Budgets & alerts → Create budget*. See [Manage spend cap budgets](https://docs.cloud.google.com/billing/docs/how-to/budgets-spend-caps).
   - On the budget creation page, select **Spend cap enforcement**. Scope it to this project and the **Cloud Run** service; a spend cap covers exactly one project and one eligible service. Set the target amount to about $1.
   - Required role: **Billing Account Administrator**, or **Billing Account Costs Manager** on the billing account plus **Editor** on the project.
   - When the cap is reached, Cloud Run usage in the project is blocked and the service returns 5xx errors. It stays suspended until you manually lift the cap: edit the budget and select *Lift spend cap*.
   - **Plain email alerts do not stop spending.** Only the spend cap enforcement option does.
   - Limits: the feature is marked Preview, enforcement uses estimated costs and is not instant (any overage is billed as normal), and Artifact Registry is not an eligible service, so the cap does not cover image storage.
   - Fallback, only if the option does not appear in your console: [disable billing with budget notifications](https://docs.cloud.google.com/billing/docs/how-to/disable-billing-with-notifications).

4. **Log in and select the project:**

   ```bash
   gcloud auth login
   gcloud config set project PROJECT_ID
   ```

5. **Enable only the two APIs this deployment needs:**

   ```bash
   gcloud services enable run.googleapis.com artifactregistry.googleapis.com --project PROJECT_ID
   ```

---

## 2. Build and push the image

`deploy/deploy.sh` runs steps 2 and 3 for you after a confirmation prompt:

```bash
PROJECT_ID=PROJECT_ID REGION=us-central1 bash deploy/deploy.sh
```

The individual commands, if you prefer to run them one at a time:

```bash
PROJECT_ID=PROJECT_ID
REGION=us-central1
SERVICE=finguard-rag
REPO=finguard
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:$(git rev-parse --short HEAD)"

# Artifact Registry Docker repository
gcloud artifacts repositories create "$REPO" \
  --project "$PROJECT_ID" \
  --location "$REGION" \
  --repository-format docker \
  --description "FinGuard RAG images"

# Docker authentication for Artifact Registry
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# Build for Cloud Run's architecture (linux/amd64), tag and push
docker buildx build --platform linux/amd64 --load -t "$IMAGE" .
docker push "$IMAGE"
```

---

## 3. Deploy

```bash
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
```

PowerShell equivalent, after building and pushing `$IMAGE`:

```powershell
$PROJECT_ID = "PROJECT_ID"; $REGION = "us-central1"; $SERVICE = "finguard-rag"
$IMAGE = "$REGION-docker.pkg.dev/$PROJECT_ID/finguard/${SERVICE}:$(git rev-parse --short HEAD)"

gcloud run deploy $SERVICE `
  --project $PROJECT_ID `
  --region $REGION `
  --image $IMAGE `
  --no-allow-unauthenticated `
  --min-instances 0 `
  --max-instances 1 `
  --concurrency 4 `
  --memory 4Gi `
  --cpu 2 `
  --timeout 300 `
  --port 8080 `
  --set-env-vars HF_HUB_OFFLINE=1
```

- **Why `HF_HUB_OFFLINE=1`:** the models are baked into the image at build time, so offline mode stops the container from calling Hugging Face at startup and avoids cold-start delay or failures.
- **Why `4Gi`:** the BGE embedding model, the CrossEncoder reranker and PyTorch stay loaded in memory, and Cloud Run's writable filesystem also uses instance memory. 4 GiB leaves headroom over the roughly 0.5 GiB measured locally.
- **No Ollama/LLM variables are set,** so the service runs retrieval-only.

Get the service URL:

```bash
gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format "value(status.url)"
```

---

## 4. Test the private service

The service rejects requests without an identity token (HTTP `403`).

**bash:**

```bash
URL=$(gcloud run services describe finguard-rag --project PROJECT_ID --region us-central1 --format "value(status.url)")
TOKEN=$(gcloud auth print-identity-token)

curl -s -H "Authorization: Bearer $TOKEN" "$URL/health"

curl -s -X POST "$URL/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is SHAP?", "top_k": 4, "generate": false}'
```

**PowerShell:**

```powershell
$URL = gcloud run services describe finguard-rag --project PROJECT_ID --region us-central1 --format "value(status.url)"
$TOKEN = gcloud auth print-identity-token

curl.exe -s -H "Authorization: Bearer $TOKEN" "$URL/health"

$body = '{"question": "What is SHAP?", "top_k": 4, "generate": false}'
$body | curl.exe -s -X POST "$URL/query" -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" --data-binary "@-"
```

Expected results: `/health` returns `{"status":"ok"}`. `/query` returns HTTP `200` with `"answer": null`, a non-empty `citations` list, a `chunks` list and `latency_ms`.

---

## 5. Measure cold start

1. Send no requests for at least 15–20 minutes. In the Cloud Console, open *Cloud Run → finguard-rag → Metrics* and wait until **Container instance count** shows 0.
2. Fetch the token first so that it is not part of the timing.
3. Time a first (cold) request, then immediately a second (warm) one.

**bash:**

```bash
TOKEN=$(gcloud auth print-identity-token)
for run in cold warm; do
  curl -s -o /dev/null -w "$run: HTTP %{http_code}, %{time_total}s\n" \
    -H "Authorization: Bearer $TOKEN" "$URL/health"
done
```

**PowerShell:**

```powershell
$TOKEN = gcloud auth print-identity-token
foreach ($run in "cold", "warm") {
  curl.exe -s -o NUL -w "${run}: HTTP %{http_code}, %{time_total}s`n" -H "Authorization: Bearer $TOKEN" "$URL/health"
}
```

The cold time includes starting the instance, pulling the image and loading the models. The container's startup log line `Application startup complete.` appears in *Cloud Run → Logs*.

For reference, running the image locally on Docker Desktop (Cloud Run-style: linux/amd64, read-only root filesystem with `/tmp` writable, non-root user, `HF_HUB_OFFLINE=1`) took about 6–9 seconds from container start to the first successful `/health`. These were local measurements, not Cloud Run results.

---

## 6. Proof

Fill in after a real deployment. Leave empty until then.

| Item | Value |
|---|---|
| Service URL (private) | |
| Image tag | |
| Image size (Artifact Registry) | |
| `/health` curl output | |
| `/query` (`generate=false`) curl output | |
| Cold-start request time (s) | |
| Warm request time (s) | |
| Console screenshot path | |
| Date checked | |

---

## 7. Teardown (post-interview cleanup)

```bash
PROJECT_ID=PROJECT_ID REGION=us-central1 bash deploy/teardown.sh
```

The script lists the stored images, asks for confirmation, then deletes the Cloud Run service and the Artifact Registry repository with all images in it. To delete only old image versions and keep the repository:

```bash
gcloud artifacts docker images list "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}" --include-tags
gcloud artifacts docker images delete "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}@sha256:<DIGEST>" --delete-tags --quiet
```

To remove everything, including the budget, you can also shut down the whole project: *IAM & Admin → Settings → Shut down*.

---

## 8. Costs and safety

- **Scale-to-zero:** with `--min-instances 0` and request-based billing, there is no compute charge while no instance is running.
- **Artifact Registry storage is small but not literally free.** The image is about 4.9 GB uncompressed locally, and the stored (compressed) size is shown in Artifact Registry after the push. Storage above the free tier is billed per GB-month. Delete old image versions, or run the teardown, to stop this charge.
- **Private access means bots cannot wake the service:** unauthenticated requests are rejected before they start an instance.
- **`--max-instances 1`** caps how many instances can run at once.
- **The Cloud Run spend cap is the backstop.** See section 1, step 3. It does not cover Artifact Registry storage, which is why the teardown deletes the repository.
- No secrets are stored in the repository or the image. Authentication uses your own `gcloud` login.

### Optional: smaller image

These changes were not made, because they touch the Dockerfile. They would cut push size and cold-start pull time:

- `chown -R` on `/opt/huggingface` creates a second copy of the models (a layer of about 532 MB). Using `COPY --chown` or downloading the models as `appuser` would avoid it.
- `build-essential` (a layer of about 323 MB) is probably not needed at runtime.
