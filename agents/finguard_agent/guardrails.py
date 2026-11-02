"""The three FinGuard guardrail callbacks.

a. input_guardrail      (before_model_callback): blocks prompt injection, PII
                        and over-long input before the model ever sees it.
b. tool_guardrail       (before_tool_callback):  validates tool arguments and
                        caps searches at MAX_SEARCHES_PER_TURN per user turn.
c. grounding_guardrail  (after_model_callback):  strips citations that point
                        to chunks/pages that were never retrieved this turn,
                        and abstains when no grounded content remains.

Returning a value from a before_* callback short-circuits the model/tool call
(ADK then skips after_model for that response); returning None lets it run.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .settings import (
    ABSTENTION_MESSAGE,
    INPUT_BLOCKED_MESSAGE,
    MAX_INPUT_CHARS,
    MAX_QUERY_CHARS,
    MAX_SEARCHES_PER_TURN,
    PII_BLOCKED_MESSAGE,
    TOO_LONG_MESSAGE,
)
from .turn_state import get_turn, increment_searches

# Detection helpers live in src/safety.py so the LangGraph conversation service
# shares exactly the same rules. Re-exported here for existing imports/tests.
from src.safety import _inside, _luhn_ok, detect_pii, detect_prompt_injection  # noqa: E402,F401


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _latest_user_text(llm_request: LlmRequest) -> str:
    """Text of the most recent user message (tool responses are skipped)."""
    for content in reversed(llm_request.contents or []):
        if content.role != "user" or not content.parts:
            continue
        texts = [part.text for part in content.parts if part.text]
        if texts:
            return "\n".join(texts)
        if any(part.function_response for part in content.parts):
            continue
    return ""


# --------------------------------------------------------------------------
# a. Input guardrail
# --------------------------------------------------------------------------

def input_guardrail(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    text = _latest_user_text(llm_request)
    if not text:
        return None
    if len(text) > MAX_INPUT_CHARS:
        return _text_response(TOO_LONG_MESSAGE.format(length=len(text), limit=MAX_INPUT_CHARS))
    if detect_prompt_injection(text):
        return _text_response(INPUT_BLOCKED_MESSAGE)
    pii = detect_pii(text)
    if pii:
        return _text_response(PII_BLOCKED_MESSAGE.format(kinds=", ".join(pii)))
    return None


# --------------------------------------------------------------------------
# b. Tool guardrail
# --------------------------------------------------------------------------

SEARCH_TOOLS = {"search_governance_documents", "search_google_drive"}
SQL_TOOLS = {"query_internal_database"}
MAX_SQL_CHARS = 2000


def _tool_error(message: str) -> dict[str, Any]:
    return {"status": "error", "error_message": message, "results": []}


def tool_guardrail(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext) -> Optional[dict[str, Any]]:
    if tool.name in SEARCH_TOOLS:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return _tool_error("The search query must be a non-empty string.")
        if len(query) > MAX_QUERY_CHARS:
            return _tool_error(f"The search query is too long (limit {MAX_QUERY_CHARS} characters). Shorten it.")
        if detect_prompt_injection(query) or detect_pii(query):
            return _tool_error("The search query was rejected by the input policy.")
        if get_turn(tool_context)["searches"] >= MAX_SEARCHES_PER_TURN:
            return _tool_error(
                f"Search limit reached ({MAX_SEARCHES_PER_TURN} per question). Answer from the "
                f"results you already have, or reply exactly: {ABSTENTION_MESSAGE}"
            )
        increment_searches(tool_context)
        return None

    if tool.name in SQL_TOOLS:
        from .sql_tool import validate_select  # local import keeps sql_tool optional

        sql = args.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return _tool_error("The SQL query must be a non-empty string.")
        if len(sql) > MAX_SQL_CHARS:
            return _tool_error(f"The SQL query is too long (limit {MAX_SQL_CHARS} characters).")
        problem = validate_select(sql)
        if problem:
            return _tool_error(problem)
        parameters = args.get("parameters") or []
        if not isinstance(parameters, list) or len(parameters) > 20:
            return _tool_error("parameters must be a list of at most 20 values.")
    return None


# --------------------------------------------------------------------------
# c. Grounding guardrail
# --------------------------------------------------------------------------

# Citation-stripping logic lives in src/grounding.py (shared with conversation/).
from src.grounding import strip_ungrounded_citations  # noqa: E402,F401


def grounding_guardrail(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    if llm_response.partial:
        return None  # only the final (aggregated) response is checked
    content = llm_response.content
    if content is None or not content.parts:
        return None
    if any(part.function_call for part in content.parts):
        return None  # intermediate tool-calling step, nothing to ground yet

    text = "".join(part.text for part in content.parts if part.text and not part.thought)
    if not text.strip() or text.strip() == ABSTENTION_MESSAGE:
        return None

    record = get_turn(callback_context)
    cleaned, valid = strip_ungrounded_citations(text, set(record["citations"]), set(record["pages"]))

    if valid == 0 or not cleaned:
        return llm_response.model_copy(update={"content": types.Content(role="model", parts=[types.Part(text=ABSTENTION_MESSAGE)])})
    # A grounded answer with the abstention sentence tacked on contradicts itself; keep the answer.
    cleaned = re.sub(r"[ \t]*" + re.escape(ABSTENTION_MESSAGE), "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if cleaned == text.strip():
        return None
    return llm_response.model_copy(update={"content": types.Content(role="model", parts=[types.Part(text=cleaned)])})
