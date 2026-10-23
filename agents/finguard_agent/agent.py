"""FinGuard ADK agent definition. `root_agent` is what `adk web` / `adk run` load."""

from __future__ import annotations

from typing import Any, Optional

from . import _bootstrap  # noqa: F401
from google.adk.agents import LlmAgent
from google.genai import types

from . import settings
from .guardrails import grounding_guardrail, input_guardrail, tool_guardrail
from .models import ModelSpec, is_gemini, model_label, resolve_model
from .multimodal import make_attachment_callback
from .prompts import build_instruction
from .retrieval_tool import search_governance_documents
from .tracing import maybe_enable_cloud_tracing


def build_agent(
    model: Optional[ModelSpec] = None,
    enable_drive: Optional[bool] = None,
    enable_sql: Optional[bool] = None,
) -> LlmAgent:
    """Create the FinGuard agent. Arguments default to the environment settings."""
    model = model if model is not None else resolve_model()
    enable_drive = settings.ENABLE_DRIVE_TOOL if enable_drive is None else enable_drive
    enable_sql = settings.ENABLE_SQL_TOOL if enable_sql is None else enable_sql

    tools: list[Any] = [search_governance_documents]
    if enable_drive:
        from .workspace_tool import search_google_drive

        tools.append(search_google_drive)
    if enable_sql:
        from .sql_tool import describe_internal_database, query_internal_database

        tools.extend([describe_internal_database, query_internal_database])

    supports_media = is_gemini(model_label(model))

    return LlmAgent(
        name=settings.AGENT_NAME,
        model=model,
        description="Answers financial AI governance questions from indexed documents, with page citations.",
        instruction=build_instruction(drive_enabled=enable_drive, sql_enabled=enable_sql),
        tools=tools,
        generate_content_config=types.GenerateContentConfig(temperature=0.2),
        before_model_callback=[input_guardrail, make_attachment_callback(supports_media)],
        before_tool_callback=tool_guardrail,
        after_model_callback=grounding_guardrail,
    )


maybe_enable_cloud_tracing()
root_agent = build_agent()
