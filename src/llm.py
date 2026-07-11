"""Small, typed client for the local Ollama chat API."""

from __future__ import annotations

import re
from dataclasses import dataclass

import requests

from .config import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
    OLLAMA_TIMEOUT_SECONDS,
)


@dataclass(frozen=True)
class OllamaClient:
    """Invoke a local Ollama chat model with LangChain-like ergonomics."""

    model: str = OLLAMA_MODEL_NAME
    base_url: str = OLLAMA_BASE_URL

    def invoke(self, messages: list[dict[str, str]]) -> str:
        """Return the model's non-streaming text response."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "think": False,
                    "stream": False,
                    "options": {
                        "temperature": LLM_TEMPERATURE,
                        "num_predict": LLM_MAX_TOKENS,
                    },
                    "messages": messages,
                },
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Ollama request failed at {self.base_url}. Ensure Ollama is running "
                f"and model {self.model!r} is installed."
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Ollama returned an unexpected response") from exc
        return re.sub(r"<think>.*?</think>", "", str(content), flags=re.DOTALL).strip()


def get_llm(model_name: str = OLLAMA_MODEL_NAME) -> OllamaClient:
    """Create a configured local Ollama client."""
    return OllamaClient(model=model_name)
