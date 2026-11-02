"""LLM access through LiteLLM with timeouts, retries and model fallbacks.

The primary model is tried first; on timeout or provider error LiteLLM retries
it (`num_retries`) and then the ordered fallback models are tried. If every model
fails, LLMUnavailable is raised and the conversation graph degrades to a
retrieval-only answer instead of returning an error.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Iterator, Protocol

Messages = list[dict[str, str]]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMUnavailable(RuntimeError):
    """Every configured model failed."""


class ChatModel(Protocol):
    name: str

    def stream(self, messages: Messages) -> Iterator[str]:
        """Yield response text deltas. Raise LLMUnavailable if no model answers."""
        ...


class LiteLLMChatModel:
    def __init__(
        self,
        model: str,
        fallbacks: list[str] | None = None,
        timeout_s: float = 60.0,
        retries: int = 2,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        self.name = model
        self.last_model = model
        self._fallbacks = fallbacks or []
        self._timeout_s = timeout_s
        self._retries = retries
        self._max_tokens = max_tokens
        self._temperature = temperature

    def stream(self, messages: Messages) -> Iterator[str]:
        """Try the primary model, then each fallback in order.

        Each model gets `retries` attempts with a per-call timeout. A model may be
        swapped only before it has produced any text; a failure mid-answer is
        raised as LLMUnavailable rather than splicing two models' outputs.
        (Fallbacks are done here instead of via LiteLLM's `fallbacks=` argument,
        which does not support synchronous streaming.)
        """
        import litellm

        errors: list[str] = []
        for model in [self.name, *self._fallbacks]:
            emitted = False
            try:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    stream=True,
                    timeout=self._timeout_s,
                    num_retries=self._retries,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                for delta in _without_think(response):
                    emitted = True
                    yield delta
                self.last_model = model
                return
            except Exception as exc:
                if emitted:
                    raise LLMUnavailable(f"{model} failed mid-stream: {exc}") from exc
                errors.append(f"{model}: {type(exc).__name__}")
        raise LLMUnavailable("all models failed: " + "; ".join(errors))


def _without_think(response: Iterable[Any]) -> Iterator[str]:
    """Yield content deltas, dropping <think>...</think> blocks from reasoning models.

    Providers split tags across chunks ("<th", "ink>"), so a small tail is held
    back until it can no longer be the start of a tag.
    """
    open_tag, close_tag = "<think>", "</think>"
    buffer, in_think = "", False
    for chunk in response:
        buffer += chunk.choices[0].delta.content or ""
        while True:
            if in_think:
                end = buffer.find(close_tag)
                if end < 0:
                    buffer = buffer[-(len(close_tag) - 1) :]
                    break
                buffer, in_think = buffer[end + len(close_tag) :], False
            else:
                start = buffer.find(open_tag)
                if start >= 0:
                    if start:
                        yield buffer[:start]
                    buffer, in_think = buffer[start + len(open_tag) :], True
                    continue
                keep = _partial_tag_suffix(buffer, open_tag)
                if len(buffer) > keep:
                    yield buffer[: len(buffer) - keep]
                    buffer = buffer[len(buffer) - keep :]
                break
    if buffer and not in_think:
        yield buffer


def _partial_tag_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of `tag`."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-size:]):
            return size
    return 0


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()
