"""Agent-level configuration. Retrieval settings come from src/config.py."""

from __future__ import annotations

import os

from . import _bootstrap  # noqa: F401  (must run before importing src)
from src.config import ABSTENTION_MESSAGE, RERANK_TOP_N, RETRIEVAL_K  # noqa: F401


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


AGENT_NAME = "finguard_agent"

# Model: "ollama_chat/<model>", "gemini-*", "openai/<model>" or any LiteLLM id.
MODEL_ENV_VAR = "FINGUARD_AGENT_MODEL"
DEFAULT_MODEL = "ollama_chat/qwen3-coder:30b"

# Guardrail limits.
MAX_INPUT_CHARS = _env_int("FINGUARD_MAX_INPUT_CHARS", 2000)
MAX_QUERY_CHARS = _env_int("FINGUARD_MAX_QUERY_CHARS", 500)
MAX_SEARCHES_PER_TURN = _env_int("FINGUARD_MAX_SEARCHES_PER_TURN", 3)

# Optional tools (Phase 2). Off by default so the agent runs with no cloud setup.
ENABLE_DRIVE_TOOL = _env_bool("FINGUARD_ENABLE_DRIVE")
ENABLE_SQL_TOOL = _env_bool("FINGUARD_ENABLE_SQL")

# Tracing to Cloud Trace / Cloud Logging (Phase 2).
ENABLE_CLOUD_TRACE = _env_bool("FINGUARD_CLOUD_TRACE")

# Multimodal input limits (Gemini only).
MAX_ATTACHMENT_BYTES = _env_int("FINGUARD_MAX_ATTACHMENT_BYTES", 10 * 1024 * 1024)
ALLOWED_ATTACHMENT_MIME_PREFIXES = ("image/", "audio/", "application/pdf")

# Session-state keys (scoped per invocation, see guardrails.py).
STATE_TURN = "finguard_turn"

INPUT_BLOCKED_MESSAGE = (
    "I can't process that request. FinGuard only answers questions about the "
    "indexed governance documents."
)
PII_BLOCKED_MESSAGE = (
    "Your message appears to contain personal data ({kinds}). Please remove it "
    "and ask again. FinGuard does not process personal information."
)
TOO_LONG_MESSAGE = (
    "Your message is too long ({length} characters; limit {limit}). Please ask "
    "a shorter, more specific question."
)
