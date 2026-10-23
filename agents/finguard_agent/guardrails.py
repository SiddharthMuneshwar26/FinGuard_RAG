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

# --------------------------------------------------------------------------
# Detection helpers (pure functions, unit-testable without ADK)
# --------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|your|system)\b[^.\n]{0,30}\b(instructions?|prompts?|rules?|guidelines?|directions?)\b",
    r"\b(reveal|show|print|repeat|leak|output)\b[^.\n]{0,30}\b(system|hidden|initial|original)\s+(prompt|instructions?|message)\b",
    r"\byou\s+are\s+now\b",
    r"\b(act|pretend|behave)\s+as\b[^.\n]{0,40}\b(unrestricted|without\s+(any\s+)?(rules|restrictions|limits|filters))\b",
    r"\b(developer|god|dan|jailbreak)\s+mode\b",
    r"\bjailbreak\b",
    r"\bdo\s+anything\s+now\b",
    r"<\|?(im_start|im_end|system|endoftext)\|?>",
    r"^\s*#{2,}\s*(system|instruction)s?\b",
    r"\bnew\s+(system\s+)?instructions?\s*:",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _INJECTION_PATTERNS]

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_AADHAAR_RE = re.compile(r"(?<![\d])[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?![\d])")
_CARD_CANDIDATE_RE = re.compile(r"(?<![\d])(?:\d[ -]?){12,18}\d(?![\d])")
_PHONE_RE = re.compile(r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?\(?\d{2,5}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,5}(?![\w])")


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def detect_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _INJECTION_RE)


def detect_pii(text: str) -> list[str]:
    """Return the kinds of personal data found (empty list if none)."""
    kinds: list[str] = []
    if _EMAIL_RE.search(text):
        kinds.append("email address")
    if _PAN_RE.search(text):
        kinds.append("PAN-like ID")
    card_spans: list[tuple[int, int]] = []
    for match in _CARD_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            card_spans.append(match.span())
    if card_spans:
        kinds.append("card-like number")
    if any(not _inside(m.span(), card_spans) for m in _AADHAAR_RE.finditer(text)):
        kinds.append("Aadhaar-like ID")
    for match in _PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 10 <= len(digits) <= 13 and not _inside(match.span(), card_spans):
            if "Aadhaar-like ID" in kinds and len(digits) == 12:
                continue
            kinds.append("phone number")
            break
    return kinds


def _inside(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(start <= span[0] and span[1] <= end for start, end in spans)


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

_BRACKET_RE = re.compile(r"\[([^\[\]\n]{1,200})\](?!\()")
_CHUNK_TOKEN_RE = re.compile(r"^[A-Za-z0-9_]+-p\d+-c\d+$")
_PAGE_TOKEN_RE = re.compile(r"^([A-Za-z0-9_]+)[,\s]+(?:p\.?|pp\.?|page)\s*(\d+)$", re.IGNORECASE)
_TOOL_TOKEN_RE = re.compile(r"^(?:drive|sql):[A-Za-z0-9_\-]+$")
def _split_sentences(line: str) -> list[str]:
    """Split on whitespace after . ! ? -- but never inside [...] (e.g. "[SHARC, p. 9]")."""
    sentences, start, depth = [], 0, 0
    for index, char in enumerate(line):
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        elif char.isspace() and depth == 0 and index > start and line[index - 1] in ".!?":
            sentences.append(line[start:index])
            start = index + 1
    sentences.append(line[start:])
    return [s for s in sentences if s.strip()] or [line]


def _classify(token: str) -> Optional[str]:
    if _CHUNK_TOKEN_RE.match(token):
        return "chunk"
    if _PAGE_TOKEN_RE.match(token):
        return "page"
    if _TOOL_TOKEN_RE.match(token):
        return "tool"
    return None


def _is_valid(token: str, kind: str, citations: set[str], pages: set[str]) -> bool:
    if kind in {"chunk", "tool"}:
        return token in citations
    match = _PAGE_TOKEN_RE.match(token)
    assert match is not None
    return f"{match.group(1).upper()}:{int(match.group(2))}" in pages


def strip_ungrounded_citations(text: str, citations: set[str], pages: set[str]) -> tuple[str, int]:
    """Remove citations not backed by this turn's tool results.

    A sentence whose citations are *all* invalid is dropped (it is an ungrounded
    claim). Returns the cleaned text and the number of valid citations kept.
    """
    pages = {page.upper() for page in pages}
    valid_total = 0
    out_lines: list[str] = []

    for line in text.split("\n"):
        kept_sentences: list[str] = []
        for sentence in _split_sentences(line):
            found = valid = 0

            def replace(match: re.Match[str]) -> str:
                nonlocal found, valid
                tokens = [token.strip() for token in re.split(r"[;,](?![\s]*(?:p\.?|pp\.?|page)\s*\d)", match.group(1))]
                kinds = [_classify(token) for token in tokens]
                if not any(kinds):
                    return match.group(0)  # not a citation, leave untouched
                good = [t for t, k in zip(tokens, kinds) if k and _is_valid(t, k, citations, pages)]
                found += sum(1 for k in kinds if k)
                valid += len(good)
                return f"[{', '.join(good)}]" if good else ""

            cleaned = _BRACKET_RE.sub(replace, sentence)
            if found and not valid:
                continue  # every citation in this sentence was fabricated
            valid_total += valid
            kept_sentences.append(re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned).rstrip())
        out_lines.append(" ".join(s for s in kept_sentences if s))

    cleaned_text = re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()
    return cleaned_text, valid_total


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
