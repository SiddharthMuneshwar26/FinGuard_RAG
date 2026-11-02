"""Environment-driven settings for the conversation service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_list(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # Storage. Empty = in-process fallback (fine for tests and local demos only).
    database_url: str = field(default_factory=lambda: os.getenv("FINGUARD_DATABASE_URL", ""))
    redis_url: str = field(default_factory=lambda: os.getenv("FINGUARD_REDIS_URL", ""))

    # Model routing through LiteLLM: primary model plus ordered fallbacks.
    model: str = field(default_factory=lambda: os.getenv("FINGUARD_CHAT_MODEL", "ollama_chat/qwen3-coder:30b"))
    fallback_models: list[str] = field(default_factory=lambda: _env_list("FINGUARD_FALLBACK_MODELS"))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("FINGUARD_LLM_TIMEOUT_S", 60.0))
    llm_retries: int = field(default_factory=lambda: _env_int("FINGUARD_LLM_RETRIES", 2))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("FINGUARD_LLM_MAX_TOKENS", 512))

    # Caching.
    retrieval_cache_ttl_s: int = field(default_factory=lambda: _env_int("FINGUARD_RETRIEVAL_CACHE_TTL_S", 3600))
    session_ttl_s: int = field(default_factory=lambda: _env_int("FINGUARD_SESSION_TTL_S", 86400))

    channels_file: Path = field(
        default_factory=lambda: Path(os.getenv("FINGUARD_CHANNELS_FILE", str(ROOT_DIR / "config" / "channels.yaml")))
    )
