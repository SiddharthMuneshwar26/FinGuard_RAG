# FinGuard ADK agent -> Vertex AI Agent Engine (Windows PowerShell).
#
# Run manually from the repository root, with the AGENT venv active, AFTER the
# one-time prerequisites in deploy/agent_engine.md (billing, APIs, ADC login).
#
#   .\deploy\agent_engine_deploy.ps1 -ProjectId my-project -Region us-central1
#
# Creates billable resources. Asks for confirmation before deploying.

param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [string]$Region = "us-central1",
    [string]$DisplayName = "finguard-agent",
    [string]$Model = "gemini-2.5-flash",
    [string]$AgentEngineId = ""          # set to update an existing instance
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Stage = Join-Path $RepoRoot "build\agent_engine\finguard_agent"

if (-not (Test-Path (Join-Path $RepoRoot "vectorstore\index.faiss"))) {
    throw "vectorstore\index.faiss not found. Build it first: python -m src.build_index"
}

# 1. Stage a self-contained agent folder: agent code + src/ + FAISS index.
#    Agent Engine uploads only this folder, so src/ must live inside it
#    (_bootstrap.py adds the folder to sys.path so `import src...` still works).
if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Path $Stage | Out-Null
Copy-Item (Join-Path $RepoRoot "agents\finguard_agent\*") $Stage -Recurse -Exclude "__pycache__", ".env"
Copy-Item (Join-Path $RepoRoot "src") (Join-Path $Stage "src") -Recurse -Exclude "__pycache__"
Copy-Item (Join-Path $RepoRoot "vectorstore") (Join-Path $Stage "vectorstore") -Recurse
Get-ChildItem $Stage -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

# src/config.py resolves VECTORSTORE_DIR as <parent of src>/vectorstore, which
# is $Stage\vectorstore in the staged layout, so no config change is needed.

# 2. Runtime dependencies (Agent Engine installs requirements.txt from the folder).
Get-Content (Join-Path $RepoRoot "requirements-agent.txt") |
    Where-Object { $_ -notmatch "^\s*(#|$)" -and $_ -notmatch "^pytest" } |
    Set-Content (Join-Path $Stage "requirements.txt") -Encoding utf8

# 3. Environment for the deployed agent: Gemini on Vertex AI (no API key needed).
@(
    "FINGUARD_AGENT_MODEL=$Model",
    "GOOGLE_GENAI_USE_VERTEXAI=TRUE",
    "HF_HUB_DISABLE_TELEMETRY=1"
) | Set-Content (Join-Path $Stage ".env") -Encoding utf8

Write-Host "Project : $ProjectId"
Write-Host "Region  : $Region"
Write-Host "Model   : $Model"
Write-Host "Staged  : $Stage"
$answer = Read-Host "Deploy this agent to Vertex AI Agent Engine? [y/N]"
if ($answer -notin @("y", "Y")) { Write-Host "Aborted."; exit 1 }

# 4. Deploy with ADK's CLI. --otel_to_cloud sends traces/logs to Cloud Trace/Logging.
$deployArgs = @(
    "deploy", "agent_engine",
    "--project", $ProjectId,
    "--region", $Region,
    "--display_name", $DisplayName,
    "--description", "FinGuard: grounded answers over financial AI governance documents",
    "--otel_to_cloud"
)
if ($AgentEngineId) { $deployArgs += @("--agent_engine_id", $AgentEngineId) }
$deployArgs += $Stage

adk @deployArgs
