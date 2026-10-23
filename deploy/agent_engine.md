# Deploying the FinGuard ADK agent to Vertex AI Agent Engine

> **Status: written, not yet run.** Everything below needs your own Google Cloud
> project with billing enabled. Nothing here has been deployed from this repo yet.

Vertex AI Agent Engine is a managed runtime for ADK agents: it hosts the agent,
manages sessions and scales it. The REST API in `api/` keeps deploying to Cloud
Run (see `DEPLOY.md`). This page covers only the agent in `agents/finguard_agent/`.

## What gets deployed

`deploy/agent_engine_deploy.ps1` stages a self-contained folder in
`build/agent_engine/finguard_agent/` containing:

| Item | Why |
|---|---|
| `agents/finguard_agent/*` | agent, tools, guardrails |
| `src/` | the existing retrieval pipeline (FAISS + CrossEncoder), reused unchanged |
| `vectorstore/` | the prebuilt FAISS index |
| `requirements.txt` | generated from `requirements-agent.txt` (minus pytest) |
| `.env` | `FINGUARD_AGENT_MODEL=gemini-2.5-flash`, `GOOGLE_GENAI_USE_VERTEXAI=TRUE` |

It then runs `adk deploy agent_engine ... --otel_to_cloud`, so traces and logs
go to Cloud Trace and Cloud Logging.

The deployed agent uses **Gemini on Vertex AI**. Ollama is not available inside
Agent Engine.

## One-time prerequisites (your Google Cloud account)

```powershell
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# Billing must be enabled on the project. Set a budget alert first.
gcloud services enable aiplatform.googleapis.com cloudtrace.googleapis.com `
    logging.googleapis.com cloudbuild.googleapis.com storage.googleapis.com
```

* Region: use one where Agent Engine and your Gemini model are both available
  (for example `us-central1`).
* The Agent Engine service agent needs permission to call Vertex AI, which it
  has by default. Add `roles/secretmanager.secretAccessor` if you enable the
  Drive tool with a Secret Manager token.

## Deploy

```powershell
.\.venv-agent\Scripts\Activate.ps1
python -m src.build_index            # only if vectorstore\ is missing
.\deploy\agent_engine_deploy.ps1 -ProjectId YOUR_PROJECT_ID -Region us-central1
```

To update an existing instance later, pass `-AgentEngineId <id>`.

## Test the deployed agent

The `adk deploy` output prints the resource name
`projects/<p>/locations/<r>/reasoningEngines/<id>`. Query it from the Vertex AI
console (Agent Engine → your agent → Playground), or from Python with the
Vertex AI SDK (`vertexai.agent_engines.get(<resource name>)`).

## Tear down (stops all charges)

Delete the Agent Engine instance from the Vertex AI console (Agent Engine →
select → Delete). Also delete the staging bucket objects if `adk` created one.

## Known risks / not verified

* **Image size and cold start.** The retrieval stack pulls in PyTorch,
  sentence-transformers and two Hugging Face models (BGE embeddings and the
  CrossEncoder), downloaded on first start. Expect a slow first request, and
  check that the Agent Engine runtime can reach huggingface.co. If it can't, or
  it is too slow, run retrieval as a separate service (the existing Cloud Run
  API with `generate=false`) and point the tool at it with
  `retrieval_tool.set_retriever(...)`.
* **Dependency pins.** `adk deploy agent_engine` adds
  `google-cloud-aiplatform[agent_engines]` to `requirements.txt`. If the
  remote build fails on an `opentelemetry-api` conflict, see the note at the
  top of `requirements-agent.txt`.
* **Gemini Enterprise registration** (publishing the agent to Gemini Enterprise
  / Agentspace) needs a Gemini Enterprise licence and is out of scope here.
