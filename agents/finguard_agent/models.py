"""Model selection via the FINGUARD_AGENT_MODEL environment variable.

  ollama_chat/qwen3-coder:30b  (default) local Ollama through LiteLLM. Set
                         OLLAMA_API_BASE if Ollama is not on http://localhost:11434.
  gemini-2.5-flash       native Gemini. Uses GOOGLE_API_KEY (AI Studio), or Vertex AI
                         with GOOGLE_GENAI_USE_VERTEXAI=TRUE (or the newer
                         GOOGLE_GENAI_USE_ENTERPRISE=TRUE) + GOOGLE_CLOUD_PROJECT +
                         GOOGLE_CLOUD_LOCATION.
  openai/gpt-4o-mini     OpenAI through LiteLLM (needs OPENAI_API_KEY). A bare
                         "gpt-..." / "o3-..." name is routed to openai/ automatically.

Any other "<provider>/<model>" string is passed to LiteLLM unchanged.
"""

from __future__ import annotations

import os
from typing import Union

from google.adk.models.base_llm import BaseLlm

from .settings import DEFAULT_MODEL, MODEL_ENV_VAR

ModelSpec = Union[str, BaseLlm]

_OPENAI_BARE_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


def configured_model_name() -> str:
    return (os.getenv(MODEL_ENV_VAR) or DEFAULT_MODEL).strip()


def is_gemini(model_name: str) -> bool:
    name = model_name.lower()
    return name.startswith("gemini") or name.startswith("projects/") and "/models/gemini" in name


def resolve_model(model_name: str | None = None) -> ModelSpec:
    """Return what LlmAgent(model=...) expects for the configured model."""
    name = (model_name or configured_model_name()).strip()
    if not name:
        raise ValueError(f"{MODEL_ENV_VAR} is empty")

    if is_gemini(name):
        return name  # ADK's registry resolves Gemini natively (google-genai)

    from google.adk.models.lite_llm import LiteLlm

    if name.startswith("ollama/"):
        # ADK docs: the plain `ollama` provider can loop on tool calls; use ollama_chat.
        name = "ollama_chat/" + name.split("/", 1)[1]
    elif "/" not in name and name.lower().startswith(_OPENAI_BARE_PREFIXES):
        name = "openai/" + name
    elif "/" not in name:
        raise ValueError(
            f"Unrecognised model {name!r}. Use 'ollama_chat/<model>', 'gemini-<version>', "
            "'openai/<model>' or another LiteLLM '<provider>/<model>' id."
        )
    return LiteLlm(model=name)


def model_label(model: ModelSpec) -> str:
    return model if isinstance(model, str) else str(getattr(model, "model", model))
