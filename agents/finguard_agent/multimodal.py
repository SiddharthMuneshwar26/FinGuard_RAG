"""Multimodal input: images (charts, scanned pages) and audio questions.

Gemini models receive image/audio parts natively (attach them in `adk web`, or
build a message with `build_user_content`). Text-only models (Ollama, OpenAI via
this agent) get a text fallback: attachments are removed from the request and
replaced by a short note so the model can tell the user to use a Gemini model.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Callable, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from .settings import ALLOWED_ATTACHMENT_MIME_PREFIXES, MAX_ATTACHMENT_BYTES

TEXT_ONLY_NOTE = (
    "[An attachment ({mime}) was removed: the current model is text-only. "
    "Set FINGUARD_AGENT_MODEL to a Gemini model to use image or audio input.]"
)
REJECTED_NOTE = "[An attachment ({mime}) was rejected: {reason}.]"


def _attachment(part: types.Part) -> Optional[tuple[str, int]]:
    """Return (mime_type, size_bytes) for a non-text media part, else None."""
    if part.inline_data is not None:
        data = part.inline_data.data or b""
        return part.inline_data.mime_type or "application/octet-stream", len(data)
    if part.file_data is not None:
        return part.file_data.mime_type or "application/octet-stream", 0
    return None


def make_attachment_callback(model_supports_media: bool) -> Callable[[CallbackContext, LlmRequest], Optional[LlmResponse]]:
    """before_model callback: validate attachments, or strip them for text-only models."""

    def attachment_guardrail(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
        for content in llm_request.contents or []:
            if content.role != "user" or not content.parts:
                continue
            new_parts: list[types.Part] = []
            for part in content.parts:
                info = _attachment(part)
                if info is None:
                    new_parts.append(part)
                    continue
                mime, size = info
                if not model_supports_media:
                    new_parts.append(types.Part(text=TEXT_ONLY_NOTE.format(mime=mime)))
                elif not mime.startswith(ALLOWED_ATTACHMENT_MIME_PREFIXES):
                    new_parts.append(types.Part(text=REJECTED_NOTE.format(mime=mime, reason="unsupported type")))
                elif size > MAX_ATTACHMENT_BYTES:
                    new_parts.append(types.Part(text=REJECTED_NOTE.format(mime=mime, reason="file too large")))
                else:
                    new_parts.append(part)
            content.parts = new_parts
        return None

    return attachment_guardrail


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        raise ValueError(f"Cannot determine the file type of {path}")
    return mime


def build_user_content(
    text: str = "",
    image_path: str | Path | None = None,
    audio_path: str | Path | None = None,
) -> types.Content:
    """Build a user message with optional image and/or audio parts (for Gemini)."""
    parts: list[types.Part] = []
    for path in (image_path, audio_path):
        if path is None:
            continue
        path = Path(path)
        data = path.read_bytes()
        if len(data) > MAX_ATTACHMENT_BYTES:
            raise ValueError(f"{path.name} is larger than {MAX_ATTACHMENT_BYTES} bytes")
        parts.append(types.Part.from_bytes(data=data, mime_type=_guess_mime(path)))
    if text.strip():
        parts.append(types.Part(text=text.strip()))
    elif audio_path is not None:
        parts.append(types.Part(text="Answer the question asked in the audio."))
    if not parts:
        raise ValueError("Provide text, an image, or an audio file.")
    return types.Content(role="user", parts=parts)
